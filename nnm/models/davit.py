import torch
from torch import nn

from nnm.layers.attention import ChannelAttention, WindowAttention
from nnm.layers.drop import DropPath
from nnm.layers.mlp import MLP
from nnm.models.pretrained import PretrainedModel


def _stage(value, index, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


class DaViTPatchEmbed(nn.Module):
    def __init__(self, config, stage):
        super().__init__()
        patch_size = _stage(config.get("patch_size", [7, 3, 3, 3]), stage, 3)
        embed_dim = _stage(config.get("embed_dim", config.get("hidden_size", [128, 256, 512, 1024])), stage, 128)
        in_channels = (
            config.get("in_channels", config.get("in_chans", config.get("num_channels", 3)))
            if stage == 0
            else _stage(config.get("embed_dim", config.get("hidden_size", [128, 256, 512, 1024])), stage - 1, 128)
        )
        stride = _stage(config.get("patch_stride", [4, 2, 2, 2]), stage, 2)
        padding = _stage(config.get("patch_padding", [3, 1, 1, 1]), stage, 1)
        pre_norm = _stage(config.get("patch_prenorm", [False, True, True, True]), stage, True)
        self.pre_norm = pre_norm
        self.conv = nn.Conv2d(in_channels, embed_dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(in_channels if pre_norm else embed_dim)

    def forward(self, hidden):
        if self.pre_norm:
            hidden = self.norm(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        hidden = self.conv(hidden)
        if not self.pre_norm:
            hidden = self.norm(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return hidden


class DaViTSpatialBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio, qkv_bias, drop_path_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.window_attn = WindowAttention(dim, num_heads, window_size, qkv_bias=qkv_bias)
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, int(dim * mlp_ratio), "gelu", bias=True)
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, hidden):
        batch, channels, height, width = hidden.shape
        hidden = self.conv1(hidden) + hidden
        hidden = hidden.flatten(2).transpose(1, 2)
        residual = hidden
        hidden, _ = self.window_attn(self.norm1(hidden), (height, width))
        hidden = residual + self.drop_path1(hidden)
        hidden = hidden.transpose(1, 2).view(batch, channels, height, width)
        hidden = self.conv2(hidden) + hidden
        hidden = hidden.flatten(2).transpose(1, 2)
        residual = hidden
        hidden = residual + self.drop_path2(self.ffn(self.norm2(hidden)))
        return hidden.transpose(1, 2).view(batch, channels, height, width)


class DaViTChannelBlock(nn.Module):
    def __init__(self, dim, groups, mlp_ratio, qkv_bias, drop_path_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.channel_attn = ChannelAttention(dim, groups, qkv_bias=qkv_bias)
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, int(dim * mlp_ratio), "gelu", bias=True)
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, hidden):
        batch, channels, height, width = hidden.shape
        hidden = self.conv1(hidden) + hidden
        hidden = hidden.flatten(2).transpose(1, 2)
        residual = hidden
        hidden, _ = self.channel_attn(self.norm1(hidden), (height, width))
        hidden = residual + self.drop_path1(hidden)
        hidden = hidden.transpose(1, 2).view(batch, channels, height, width)
        hidden = self.conv2(hidden) + hidden
        hidden = hidden.flatten(2).transpose(1, 2)
        residual = hidden
        hidden = residual + self.drop_path2(self.ffn(self.norm2(hidden)))
        return hidden.transpose(1, 2).view(batch, channels, height, width)


class DaViTBlock(nn.Module):
    def __init__(self, dim, num_heads, groups, window_size, mlp_ratio, qkv_bias, spatial_drop_path, channel_drop_path):
        super().__init__()
        self.spatial_block = DaViTSpatialBlock(dim, num_heads, window_size, mlp_ratio, qkv_bias, spatial_drop_path)
        self.channel_block = DaViTChannelBlock(dim, groups, mlp_ratio, qkv_bias, channel_drop_path)

    def forward(self, hidden):
        return self.channel_block(self.spatial_block(hidden))


class DaViT(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        depths = config.get("depths", config.get("num_hidden_layers", [1, 1, 9, 1]))
        embed_dims = config.get("embed_dim", config.get("hidden_size", [128, 256, 512, 1024]))
        num_heads = config.get("num_heads", [4, 8, 16, 32])
        num_groups = config.get("num_groups", [4, 8, 16, 32])
        window_size = config.get("window_size", 12)
        mlp_ratio = config.get("mlp_ratio", 4.0)
        qkv_bias = config.get("qkv_bias", True)
        drop_path_rate = config.get("drop_path_rate", 0.0)
        dpr = torch.linspace(0, drop_path_rate, sum(depths) * 2)
        offset = 0
        self.convs = nn.ModuleList([DaViTPatchEmbed(config, stage) for stage in range(len(depths))])
        blocks = []
        for stage, depth in enumerate(depths):
            blocks.append(
                nn.ModuleList(
                    [
                        DaViTBlock(
                            _stage(embed_dims, stage),
                            _stage(num_heads, stage),
                            _stage(num_groups, stage),
                            window_size,
                            mlp_ratio,
                            qkv_bias,
                            dpr[offset + index * 2].item(),
                            dpr[offset + index * 2 + 1].item(),
                        )
                        for index in range(depth)
                    ]
                )
            )
            offset += depth * 2
        self.blocks = nn.ModuleList(blocks)
        self.num_features = _stage(embed_dims, len(depths) - 1)

    def forward_features(self, pixel_values):
        hidden = pixel_values
        features = []
        for conv, blocks in zip(self.convs, self.blocks):
            hidden = conv(hidden)
            for block in blocks:
                hidden = block(hidden)
            features.append(hidden)
        return hidden, features

    def forward(self, pixel_values):
        hidden, _ = self.forward_features(pixel_values)
        return hidden
