import torch
from functools import partial
from torch import nn

from nnm.layers.attention import RotarySelfAttention
from nnm.layers.mlp import GatedMLP, MLP
from nnm.layers.rope import RopePositionEmbedding
from nnm.layers.transformer import RotaryVisionBlock
from nnm.layers.vit import PatchEmbed2D
from nnm.models.pretrained import PretrainedModel


def build_block_ffn(config, hidden_size):
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


def build_block(config, hidden_size, num_heads, head_dim, layerscale_init, drop_path_rate):
    qkv_bias = config.get("qkv_bias", False)
    attention = RotarySelfAttention(
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
    )
    return RotaryVisionBlock(
        hidden_size,
        attention,
        build_block_ffn(config, hidden_size),
        norm_layer=partial(nn.LayerNorm, eps=config.get("norm_eps", config.get("layer_norm_eps", 1e-6))),
        layerscale_init=layerscale_init,
        drop_path_rate=drop_path_rate,
    )


class DINOv3(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        num_heads = config.get("num_attention_heads", config.get("num_heads"))
        num_layers = config.get("num_hidden_layers", config.get("depth"))
        head_dim = hidden_size // num_heads
        patch_size = config.get("patch_size", 16)
        num_register_tokens = config.get("num_register_tokens", config.get("n_storage_tokens", 0))
        norm_eps = config.get("norm_eps", config.get("layer_norm_eps", 1e-6))
        layerscale_init = config.get("layerscale_init", config.get("layerscale_value", 1e-5))
        drop_path_rate = config.get("drop_path_rate", 0.0)
        self.num_register_tokens = num_register_tokens
        self.patch_embed = PatchEmbed2D(
            kernel_size=patch_size,
            in_channels=config.get("in_chans", config.get("num_channels", 3)),
            embed_dim=hidden_size,
            bias=True,
            norm=False,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.storage_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, hidden_size)) if num_register_tokens else None
        self.rope_embed = RopePositionEmbedding(
            hidden_size,
            num_heads,
            base=config.get("pos_embed_rope_base", config.get("rope_theta", 100.0)),
            min_period=config.get("pos_embed_rope_min_period"),
            max_period=config.get("pos_embed_rope_max_period"),
            normalize_coords=config.get("pos_embed_rope_normalize_coords", "separate"),
        )
        drop_path_rates = (
            [drop_path_rate * index / max(num_layers - 1, 1) for index in range(num_layers)]
            if drop_path_rate
            else [0.0] * num_layers
        )
        self.layers = nn.ModuleList(
            [
                build_block(config, hidden_size, num_heads, head_dim, layerscale_init, drop_path_rates[index])
                for index in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)

    def forward(self, pixel_values, bool_masked_pos=None):
        patches, (height, width) = self.patch_embed(pixel_values)
        if bool_masked_pos is not None:
            patches = torch.where(bool_masked_pos.unsqueeze(-1), self.mask_token.to(patches.dtype), patches)
        tokens = [self.cls_token.expand(patches.shape[0], -1, -1)]
        if self.storage_tokens is not None:
            tokens.append(self.storage_tokens.expand(patches.shape[0], -1, -1))
        hidden = torch.cat([*tokens, patches], dim=1)
        cos, sin = self.rope_embed(height, width)
        prefix = 1 + self.num_register_tokens
        cos = torch.cat([cos.new_ones(prefix, cos.shape[-1]), cos], dim=0)
        sin = torch.cat([sin.new_zeros(prefix, sin.shape[-1]), sin], dim=0)
        for layer in self.layers:
            hidden = layer(hidden, rotary_embeddings=(cos, sin))
        return self.norm(hidden)

    def forward_features(self, pixel_values):
        hidden = self.forward(pixel_values)
        prefix = 1 + self.num_register_tokens
        return {
            "x_norm_clstoken": hidden[:, 0],
            "x_storage_tokens": hidden[:, 1:prefix],
            "x_norm_patchtokens": hidden[:, prefix:],
        }
