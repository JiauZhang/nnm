import math

import torch
from conippets.config import Config
from torch import nn

from nnm.layers.embedding import merge_placeholders
from nnm.layers.norm import RMSNorm
from nnm.layers.rope import RoPE, RoPE2D, yarn_inv_freq
from nnm.models.clip import CLIPVisionModel
from nnm.models.pretrained import PretrainedModel
from nnm.models.sam import SamVisionEncoder
from nnm.models.unlimited_ocr.decoder import DecoderLayer
from nnm.models.unlimited_ocr.sparse import build_page_layout


class UnlimitedOCRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        sparse = config.sparse_attention_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.get("pad_token_id", 0))
        self.layers = nn.ModuleList(
            [DecoderLayer(config=config, layer_idx=index, sparse=sparse) for index in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        vision = config.vision_config
        sam_config = vision["width"]["sam_vit_b"]
        clip_config = vision["width"]["clip-l-14-224"]
        out_chans = sam_config.get("out_chans", 256)
        downsample_channels = sam_config.get("downsample_channels", (512, 1024))
        self.sam_model = SamVisionEncoder(
            Config(
                hidden_size=sam_config.width,
                num_attention_heads=sam_config.heads,
                image_size=sam_config.get("image_size", vision.get("image_size", 1024)),
                patch_size=sam_config.get("patch_size", 16),
                num_hidden_layers=sam_config.layers,
                mlp_ratio=sam_config.get("mlp_ratio", 4.0),
                output_channels=out_chans,
                window_size=sam_config.get("window_size", 14),
                global_attn_indexes=list(sam_config.get("global_attn_indexes", ())),
                layer_norm_eps=sam_config.get("norm_eps", 1e-6),
            )
        )
        self.sam_downsample_2x = nn.Conv2d(out_chans, downsample_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.sam_downsample_4x = nn.Conv2d(
            downsample_channels[0], downsample_channels[1], kernel_size=3, stride=2, padding=1, bias=False
        )
        self.vision_model = CLIPVisionModel(
            Config(
                hidden_size=clip_config.width,
                num_attention_heads=clip_config.heads,
                num_hidden_layers=clip_config.layers,
                intermediate_size=clip_config.get(
                    "ffn_hidden_size", int(clip_config.width * clip_config.get("mlp_ratio", 4.0))
                ),
                image_size=clip_config.get("image_size", 224),
                patch_size=clip_config.get("patch_size", 14),
                layer_norm_eps=clip_config.get("layernorm_epsilon", 1e-5),
                hidden_act="quick_gelu",
            ),
            post_layernorm=False,
        )
        self.vision_model.pre_layrnorm = nn.LayerNorm(clip_config.width, eps=clip_config.get("pre_layernorm_epsilon", 1e-5))
        projector_config = config.projector_config
        self.projector = nn.Linear(projector_config.input_dim, projector_config.n_embed)
        n_embed = projector_config.n_embed
        embed_std = 1 / math.sqrt(n_embed)
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

        self.block_size = int(sparse["sparse_block_size"])
        self.local_window = max(1, int(sparse["sparse_local_block"])) * self.block_size
        self.image_token_id = int(sparse["image_token_id"])
        self.page_token_id = int(sparse["page_token_id"])
        self.per_page_image_tokens = int(sparse.get("per_page_image_tokens", 273))
        self.page_rope_2d = None
        self.page_mod = 0
        if str(sparse.get("sparse_main_rope_mode", "1d")).lower() == "page2d":
            head_dim = config.hidden_size // config.num_attention_heads
            half = head_dim // 2
            page_pairs = int(sparse.get("sparse_rope_page_pairs", 8))
            n_tok = half - page_pairs
            positions = torch.arange(0, n_tok, dtype=torch.float32)
            tok_theta = float(sparse.get("sparse_rope_tok_theta", config.rope_theta))
            if str(sparse.get("sparse_main_rope_tok_freq_mode", "resample")).lower() == "truncate":
                inv_tok = 1.0 / (tok_theta ** (positions * 2 / head_dim))
            else:
                inv_tok = 1.0 / (tok_theta ** (positions / n_tok))
            page_theta = float(sparse.get("sparse_rope_page_theta", 256.0))
            inv_page = 1.0 / (page_theta ** (torch.arange(0, page_pairs, dtype=torch.float32) / page_pairs))
            self.page_rope_2d = RoPE2D([inv_tok, inv_page])
            self.page_mod = int(sparse.get("sparse_rope_page_mod", 0))
        scaling = config.get("rope_scaling", {}) or {}
        factor = float(scaling.get("factor", 1.0))
        self.yarn_rope = RoPE(
            yarn_inv_freq(
                config.hidden_size // config.num_attention_heads,
                float(config.rope_theta),
                factor,
                int(scaling.get("original_max_position_embeddings", config.max_position_embeddings)),
                float(scaling.get("beta_fast", 32)),
                float(scaling.get("beta_slow", 1)),
            ),
            scaling=float(scaling.get("attention_factor", 1.0)),
        )

    def encode_page(self, pixel_values):
        pixel_values = pixel_values.to(self.image_newline.dtype)
        sam_features = self.sam_downsample_4x(self.sam_downsample_2x(self.sam_model(pixel_values)))
        clip_features = self.vision_model(pixel_values, patch_embeds=sam_features)["last_hidden_state"]
        merged = torch.cat((clip_features[:, 1:], sam_features.flatten(2).transpose(1, 2)), dim=-1)
        features = self.projector(merged)[0]
        height, width = sam_features.shape[2], sam_features.shape[3]
        features = features.view(height, width, -1)
        features = torch.cat([features, self.image_newline[None, None, :].expand(height, 1, -1)], dim=1).reshape(
            -1, features.shape[-1]
        )
        return torch.cat([features, self.view_seperator[None, :]], dim=0)

    def _page_context(self, input_ids, position_ids, dtype_reference, seq_len):
        layout = build_page_layout(input_ids, self.image_token_id, self.page_token_id, self.per_page_image_tokens)
        if layout.rope_pages is not None and self.page_rope_2d is not None:
            page = layout.rope_pages
            if self.page_mod > 0:
                page = torch.remainder(page, self.page_mod)
            cos, sin = self.page_rope_2d(torch.stack([layout.rope_tokens, page], dim=-1))
        else:
            cos, sin = self.yarn_rope(position_ids)
        num_key_blocks = (seq_len + self.block_size - 1) // self.block_size
        block_candidate = None
        if layout.image_pages is not None and layout.query_pages is not None:
            chunks = []
            for start in range(0, seq_len, 2048):
                end = min(start + 2048, seq_len)
                chunks.append(layout.candidate_mask(position_ids[:, start:end], num_key_blocks, self.block_size))
            block_candidate = torch.cat(chunks, dim=2)[:, 0]
        image_blocks = None
        if layout.image_mask is not None:
            image_blocks = layout.image_blocks(num_key_blocks, self.block_size)
        return cos, sin, block_candidate, image_blocks

    def forward(self, input_ids=None, inputs_embeds=None, pixel_values=None, attention_mask=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if pixel_values is not None:
            page_features = torch.cat(
                [self.encode_page(pixel_values[index : index + 1]) for index in range(pixel_values.shape[0])], dim=0
            )
            image_positions = input_ids == self.image_token_id
            if int(image_positions.sum()) != page_features.shape[0]:
                raise ValueError(
                    f"image token count {int(image_positions.sum())} does not match page features {page_features.shape[0]}"
                )
            inputs_embeds = merge_placeholders(inputs_embeds, image_positions, page_features)

        batch, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
        position_ids = torch.arange(seq_len, device=inputs_embeds.device)[None].expand(batch, -1)
        cos, sin, block_candidate, image_blocks = self._page_context(input_ids, position_ids, inputs_embeds, seq_len)
        additive_mask = None
        if attention_mask is not None:
            additive_mask = torch.where(attention_mask > 0, 0.0, float("-inf"))[:, None, None, :].to(inputs_embeds.dtype)

        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, position_ids, block_candidate, image_blocks, additive_mask)
        return self.norm(hidden)


class UnlimitedOCR(PretrainedModel):
    weight_pattern = "model*.safetensors"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = UnlimitedOCRModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, inputs_embeds=None, pixel_values=None, attention_mask=None):
        hidden = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        return self.lm_head(hidden)
