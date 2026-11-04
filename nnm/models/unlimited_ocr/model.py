import math

import torch
from torch import nn

from nnm.layers.embedding import merge_placeholders
from nnm.layers.norm import RMSNorm
from nnm.layers.rope import RoPE, RoPE2D, yarn_inv_freq
from nnm.models.pretrained import PretrainedModel
from nnm.models.unlimited_ocr.clip import ClipEncoder
from nnm.models.unlimited_ocr.decoder import DecoderLayer
from nnm.models.unlimited_ocr.projector import MlpProjector
from nnm.models.unlimited_ocr.sam import SamEncoder
from nnm.models.unlimited_ocr.sparse import (
    build_candidate_mask,
    build_page_aware_ids,
    build_page_rope_pos,
    image_block_mask,
)


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
        self.sam_model = SamEncoder(
            image_size=sam_config.get("image_size", vision.get("image_size", 1024)),
            patch_size=sam_config.get("patch_size", 16),
            width=sam_config.width,
            depth=sam_config.layers,
            num_heads=sam_config.heads,
            mlp_ratio=sam_config.get("mlp_ratio", 4.0),
            out_chans=sam_config.get("out_chans", 256),
            window_size=sam_config.get("window_size", 14),
            global_attn_indexes=sam_config.get("global_attn_indexes", ()),
            downsample_channels=sam_config.get("downsample_channels", (512, 1024)),
            norm_eps=sam_config.get("norm_eps", 1e-6),
        )
        self.vision_model = ClipEncoder(
            width=clip_config.width,
            image_size=clip_config.get("image_size", 224),
            patch_size=clip_config.get("patch_size", 14),
            depth=clip_config.layers,
            num_heads=clip_config.heads,
            hidden_dim=clip_config.get("ffn_hidden_size", int(clip_config.width * clip_config.get("mlp_ratio", 4.0))),
            pre_layernorm_eps=clip_config.get("pre_layernorm_epsilon", 1e-5),
            norm_eps=clip_config.get("layernorm_epsilon", 1e-5),
        )
        projector_config = config.projector_config
        self.projector = MlpProjector(input_dim=projector_config.input_dim, n_embed=projector_config.n_embed)
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
        sam_features = self.sam_model(pixel_values)
        clip_features = self.vision_model(pixel_values, sam_features)
        merged = torch.cat((clip_features[:, 1:], sam_features.flatten(2).transpose(1, 2)), dim=-1)
        features = self.projector(merged)[0]
        height, width = sam_features.shape[2], sam_features.shape[3]
        features = features.view(height, width, -1)
        features = torch.cat([features, self.image_newline[None, None, :].expand(height, 1, -1)], dim=1).reshape(
            -1, features.shape[-1]
        )
        return torch.cat([features, self.view_seperator[None, :]], dim=0)

    def _page_context(self, input_ids, position_ids, dtype_reference, seq_len):
        page_pos = None
        if input_ids is not None and self.page_rope_2d is not None:
            page_pos = build_page_rope_pos(input_ids, self.image_token_id, self.page_token_id)
        if page_pos is not None:
            page = page_pos[1]
            if self.page_mod > 0:
                page = torch.remainder(page, self.page_mod)
            positions = torch.stack([page_pos[0], page], dim=-1)
            cos, sin = self.page_rope_2d(positions)
        else:
            cos, sin = self.yarn_rope(position_ids)
        block_candidate = None
        image_blocks = None
        num_key_blocks = (seq_len + self.block_size - 1) // self.block_size
        if input_ids is not None:
            image_mask, image_pages, query_pages = build_page_aware_ids(
                input_ids, self.image_token_id, self.page_token_id, self.per_page_image_tokens
            )
            if image_pages is not None and query_pages is not None:
                chunks = []
                for start in range(0, seq_len, 2048):
                    end = min(start + 2048, seq_len)
                    chunks.append(
                        build_candidate_mask(
                            position_ids[:, start:end],
                            num_key_blocks,
                            image_mask,
                            image_pages,
                            query_pages,
                            self.block_size,
                        )
                    )
                block_candidate = torch.cat(chunks, dim=2)[:, 0]
            if image_mask is not None:
                image_blocks = image_block_mask(image_mask, num_key_blocks, self.block_size)
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
