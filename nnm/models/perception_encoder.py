import torch
from functools import partial
from torch import nn

from nnm.layers.attention import RotarySelfAttention
from nnm.layers.attention_mask import make_causal_mask
from nnm.layers.mlp import GatedMLP, MLP
from nnm.layers.rope import RoPE, RopePositionEmbedding, rotary_inv_freq
from nnm.layers.transformer import RotaryVisionBlock, TransformerBlock
from nnm.layers.vit import PatchEmbed2D
from nnm.models.pretrained import PretrainedModel


def _ffn(config, hidden_size):
    ffn_layer = config.get("ffn_layer", "mlp")
    ratio = config.get("ffn_ratio", 4.0)
    bias = config.get("ffn_bias", config.get("mlp_bias", True))
    intermediate = config.get("intermediate_size")
    if ffn_layer != "mlp" or config.get("use_gated_mlp", False):
        align_to = int(ffn_layer.removeprefix("swiglu") or 8)
        size = int(hidden_size * ratio * 2 / 3)
        return GatedMLP(hidden_size, size + (-size % align_to), bias)
    size = intermediate if intermediate is not None else int(hidden_size * ratio)
    return MLP(hidden_size, size, config.get("hidden_act", "gelu"), bias)


class PerceptionEncoderVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("embed_dim", config.get("hidden_size"))
        num_heads = config.get("num_heads", config.get("num_attention_heads"))
        num_layers = config.get("depth", config.get("num_hidden_layers"))
        head_dim = hidden_size // num_heads
        patch_size = config.get("patch_size", 14)
        norm_eps = config.get("norm_eps", config.get("layer_norm_eps", 1e-6))
        layerscale_init = config.get("layerscale_init", config.get("layerscale_value", 1e-5))
        num_storage_tokens = config.get("n_storage_tokens", config.get("num_register_tokens", 0))
        self.num_storage_tokens = num_storage_tokens
        self.use_cls_token = config.get("use_cls_token", True)
        self.patch_embed = PatchEmbed2D(
            kernel_size=patch_size,
            in_channels=config.get("in_chans", config.get("num_channels", 3)),
            embed_dim=hidden_size,
            bias=True,
            norm=False,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size)) if self.use_cls_token else None
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.storage_tokens = nn.Parameter(torch.zeros(1, num_storage_tokens, hidden_size)) if num_storage_tokens else None
        self.rope_embed = RopePositionEmbedding(
            hidden_size,
            num_heads,
            base=config.get("pos_embed_rope_base", config.get("rope_theta", 100.0)),
            min_period=config.get("pos_embed_rope_min_period"),
            max_period=config.get("pos_embed_rope_max_period"),
            normalize_coords=config.get("pos_embed_rope_normalize_coords", "separate"),
        )
        qkv_bias = config.get("qkv_bias", False)
        self.layers = nn.ModuleList(
            [
                RotaryVisionBlock(
                    hidden_size,
                    RotarySelfAttention(
                        hidden_size,
                        num_heads,
                        head_dim=head_dim,
                        qkv_bias=(
                            config.get("query_bias", qkv_bias),
                            config.get("key_bias", qkv_bias),
                            config.get("value_bias", qkv_bias),
                        ),
                        o_bias=config.get("proj_bias", True),
                        causal=False,
                    ),
                    _ffn(config, hidden_size),
                    norm_layer=partial(nn.LayerNorm, eps=norm_eps),
                    layerscale_init=layerscale_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)

    def forward(self, pixel_values):
        patches, (height, width) = self.patch_embed(pixel_values)
        tokens = []
        if self.cls_token is not None:
            tokens.append(self.cls_token.expand(patches.shape[0], -1, -1))
        if self.storage_tokens is not None:
            tokens.append(self.storage_tokens.expand(patches.shape[0], -1, -1))
        hidden = torch.cat([*tokens, patches], dim=1)
        cos, sin = self.rope_embed(height, width)
        prefix = (1 if self.use_cls_token else 0) + self.num_storage_tokens
        cos = torch.cat([cos.new_ones(prefix, cos.shape[-1]), cos], dim=0)
        sin = torch.cat([sin.new_zeros(prefix, sin.shape[-1]), sin], dim=0)
        for layer in self.layers:
            hidden = layer(hidden, rotary_embeddings=(cos, sin))
        return self.norm(hidden)

    def forward_features(self, pixel_values):
        hidden = self.forward(pixel_values)
        prefix = (1 if self.use_cls_token else 0) + self.num_storage_tokens
        features = {
            "x_norm_patchtokens": hidden[:, prefix:],
            "x_norm_clstoken": hidden[:, 0] if self.use_cls_token else hidden[:, prefix - self.num_storage_tokens],
        }
        if self.num_storage_tokens:
            offset = 1 if self.use_cls_token else 0
            features["x_storage_tokens"] = hidden[:, offset:prefix]
        return features


class PerceptionEncoderText(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get(
            "text_hidden_size", config.get("text_embed_dim", config.get("embed_dim", config.get("hidden_size")))
        )
        num_heads = config.get(
            "text_num_heads", config.get("text_num_attention_heads", config.get("num_heads", config.get("num_attention_heads")))
        )
        num_layers = config.get("text_depth", config.get("text_num_hidden_layers", 12))
        intermediate_size = config.get("text_intermediate_size", hidden_size * 4)
        norm_eps = config.get("norm_eps", config.get("layer_norm_eps", 1e-6))
        head_dim = hidden_size // num_heads
        self.rotary = RoPE(rotary_inv_freq(head_dim, config.get("text_rope_base", 10000.0)))
        self.token_embedding = nn.Embedding(config.get("text_vocab_size", config.get("vocab_size", 32000)), hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    RotarySelfAttention(
                        hidden_size,
                        num_heads,
                        head_dim=head_dim,
                        qkv_bias=True,
                        o_bias=True,
                        causal=True,
                    ),
                    MLP(hidden_size, intermediate_size, config.get("hidden_act", "gelu"), bias=True),
                    norm_layer=partial(nn.LayerNorm, eps=norm_eps),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.head = nn.Linear(hidden_size, config.get("projection_dim", hidden_size), bias=True)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.token_embedding(input_ids)
        mask = None
        if attention_mask is not None:
            mask = make_causal_mask(
                hidden.shape[1],
                hidden.shape[1],
                0,
                hidden.device,
                hidden.dtype,
                attn_mask=attention_mask,
            )
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        rotary_embeddings = self.rotary(positions)
        for layer in self.layers:
            hidden = layer(hidden, rotary_embeddings=rotary_embeddings, attn_mask=mask)
        hidden = self.norm(hidden)
        return self.head(hidden[:, 0])


class PerceptionEncoder(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.vision_model = PerceptionEncoderVision(config)
        self.text_model = PerceptionEncoderText(config) if config.get("use_text", True) else None
        self.logit_scale = nn.Parameter(torch.zeros(1))
        self.logit_bias = nn.Parameter(torch.zeros(1))

    def get_image_features(self, pixel_values):
        return self.vision_model.forward_features(pixel_values)["x_norm_clstoken"]

    def get_text_features(self, input_ids, attention_mask=None):
        return self.text_model(input_ids, attention_mask)

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        features = self.vision_model.forward_features(pixel_values)
        if self.text_model is None or input_ids is None:
            return features
        image_features = features["x_norm_clstoken"]
        text_features = self.get_text_features(input_ids, attention_mask)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.t() + self.logit_bias
        return {"logits_per_image": logits_per_image, "logits_per_text": logits_per_image.t(), **features}
