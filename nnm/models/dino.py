import torch
from functools import partial
from torch import nn

from nnm.layers.attention import RotarySelfAttention, SelfAttention
from nnm.layers.grid import PatchEmbed2D
from nnm.layers.mlp import GatedMLP, MLP
from nnm.layers.rope import RopePositionEmbedding
from nnm.layers.transformer import LayerScaleBlock, RotaryLayerScaleBlock
from nnm.pretrained import PretrainedModel


def build_block_ffn(config, hidden_size):
    ffn_layer = config.get("ffn_layer", "mlp")
    ratio = config.get("ffn_ratio", 4.0)
    bias = config.get("ffn_bias", True)
    intermediate = config.get("intermediate_size")
    if ffn_layer != "mlp" or config.get("use_gated_mlp", False):
        align_to = int(ffn_layer.removeprefix("swiglu") or 8)
        size = int(hidden_size * ratio)
        return GatedMLP(hidden_size, size + (-size % align_to), bias)
    size = intermediate if intermediate is not None else int(hidden_size * ratio)
    return MLP(hidden_size, size, config.get("hidden_act", "gelu"), bias)


def build_dinov2_block(config, hidden_size, num_heads, head_dim, layerscale_init, drop_path_rate):
    qkv_bias = config.get("qkv_bias", True)
    attention = SelfAttention(
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
    return LayerScaleBlock(
        hidden_size,
        attention,
        build_block_ffn(config, hidden_size),
        norm_layer=partial(nn.LayerNorm, eps=config.get("layer_norm_eps", 1e-6)),
        layerscale_init=layerscale_init,
        drop_path_rate=drop_path_rate,
    )


def build_dinov3_block(config, hidden_size, num_heads, head_dim, layerscale_init, drop_path_rate):
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
    return RotaryLayerScaleBlock(
        hidden_size,
        attention,
        build_block_ffn(config, hidden_size),
        norm_layer=partial(nn.LayerNorm, eps=config.get("layer_norm_eps", 1e-6)),
        layerscale_init=layerscale_init,
        drop_path_rate=drop_path_rate,
    )


class DINOv2(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_layers = config.num_hidden_layers
        head_dim = hidden_size // num_heads
        patch_size = config.get("patch_size", 14)
        image_size = config.get("image_size", 518)
        num_register_tokens = config.get("num_register_tokens", 0)
        norm_eps = config.get("layer_norm_eps", 1e-6)
        layerscale_init = config.get("layerscale_init", 1e-5)
        drop_path_rate = config.get("drop_path_rate", 0.0)
        self.num_register_tokens = num_register_tokens
        self.interpolate_offset = config.get("interpolate_offset", 0.1)
        self.interpolate_antialias = config.get("interpolate_antialias", False)
        self.patch_embed = PatchEmbed2D(
            kernel_size=patch_size,
            in_channels=config.get("num_channels", 3),
            embed_dim=hidden_size,
            bias=True,
            norm=False,
        )
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, hidden_size)) if num_register_tokens else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        drop_path_rates = (
            [drop_path_rate * index / max(num_layers - 1, 1) for index in range(num_layers)]
            if drop_path_rate
            else [0.0] * num_layers
        )
        self.blocks = nn.ModuleList(
            [
                build_dinov2_block(config, hidden_size, num_heads, head_dim, layerscale_init, drop_path_rates[index])
                for index in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)

    def interpolate_positions(self, width, height):
        num_patches = self.pos_embed.shape[1] - 1
        grid = int(num_patches**0.5)
        pos = self.pos_embed.float()
        cls_position = pos[:, 0]
        patch_positions = pos[:, 1:].reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
        kernel = self.patch_embed.proj.kernel_size[0]
        w0 = width // kernel + self.interpolate_offset
        h0 = height // kernel + self.interpolate_offset
        scale = (w0 / grid, h0 / grid)
        patch_positions = nn.functional.interpolate(
            patch_positions, scale_factor=scale, mode="bicubic", antialias=self.interpolate_antialias
        )
        patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(1, -1, pos.shape[-1])
        return torch.cat([cls_position.unsqueeze(0), patch_positions], dim=1).to(self.pos_embed.dtype)

    def prepare_tokens(self, pixel_values, bool_masked_pos=None):
        patches, (height, width) = self.patch_embed(pixel_values)
        if bool_masked_pos is not None:
            patches = torch.where(bool_masked_pos.unsqueeze(-1), self.mask_token.to(patches.dtype), patches)
        batch = patches.shape[0]
        position = self.pos_embed
        if patches.shape[1] + 1 != position.shape[1]:
            pixel_h, pixel_w = height * self.patch_embed.proj.kernel_size[0], width * self.patch_embed.proj.kernel_size[0]
            position = self.interpolate_positions(pixel_w, pixel_h)
        tokens = torch.cat([self.cls_token.expand(batch, -1, -1), patches], dim=1) + position
        if self.register_tokens is not None:
            register = self.register_tokens.expand(batch, -1, -1)
            tokens = torch.cat([tokens[:, :1], register, tokens[:, 1:]], dim=1)
        return tokens

    def forward(self, pixel_values, bool_masked_pos=None):
        hidden = self.prepare_tokens(pixel_values, bool_masked_pos)
        for layer in self.blocks:
            hidden = layer(hidden)
        return self.norm(hidden)

    def get_intermediate_layers(self, pixel_values, indices, return_class_token=False, norm=True):
        hidden = self.prepare_tokens(pixel_values)
        prefix = 1 + self.num_register_tokens
        outputs = []
        for index, layer in enumerate(self.blocks):
            hidden = layer(hidden)
            if index in indices:
                out = self.norm(hidden) if norm else hidden
                if return_class_token:
                    outputs.append((out[:, prefix:], out[:, 0]))
                else:
                    outputs.append(out[:, prefix:])
        return outputs


class DINOv3(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_layers = config.num_hidden_layers
        head_dim = hidden_size // num_heads
        patch_size = config.get("patch_size", 16)
        num_register_tokens = config.get("num_register_tokens", 0)
        norm_eps = config.get("layer_norm_eps", 1e-6)
        layerscale_init = config.get("layerscale_init", 1e-5)
        drop_path_rate = config.get("drop_path_rate", 0.0)
        self.num_register_tokens = num_register_tokens
        self.patch_embed = PatchEmbed2D(
            kernel_size=patch_size,
            in_channels=config.get("num_channels", 3),
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
            base=config.get("rope_theta", 100.0),
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
                build_dinov3_block(config, hidden_size, num_heads, head_dim, layerscale_init, drop_path_rates[index])
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
