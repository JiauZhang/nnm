import torch
from torch import nn
from torch.nn import functional as F_

from nnm.layers.attention import CrossAttention
from nnm.layers.mlp import MLP


def resize_abs_pos(abs_pos, size, mode="bicubic"):
    dtype = abs_pos.dtype
    hidden = abs_pos.permute(0, 3, 1, 2).to(torch.float32)
    hidden = F_.interpolate(hidden, size=(size, size), mode=mode, antialias=True, align_corners=False).to(dtype)
    return hidden.permute(0, 2, 3, 1)


def patchify(input, patch_size):
    n, c, h, w = input.shape
    patches = input.reshape(n, c, h // patch_size, patch_size, w // patch_size, patch_size)
    return patches.permute(0, 2, 4, 3, 5, 1).reshape(n, -1, patch_size * patch_size * c)


def unpatchify(input, output_size, patch_size):
    n, _, _ = input.shape
    h, w = output_size
    c = input.shape[-1] // (patch_size * patch_size)
    patches = input.reshape(n, h // patch_size, w // patch_size, patch_size, patch_size, c)
    return patches.permute(0, 5, 1, 3, 2, 4).reshape(n, c, h, w)


class PatchEmbed2D(nn.Module):
    def __init__(self, *, kernel_size, in_channels, embed_dim, stride=None, padding=0, bias=False, norm=True, norm_eps=1e-6):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps) if norm else None

    def forward(self, input):
        output = self.proj(input)
        N, C, H, W = output.shape
        output = output.reshape(N, C, -1).permute(0, 2, 1)
        if self.norm is not None:
            output = self.norm(output)
        return output, (H, W)


class LayerScale(nn.Module):
    def __init__(self, dim, init_scale=1e-6, spatial=False):
        super().__init__()
        shape = (dim, 1, 1) if spatial else (dim,)
        self.gamma = nn.Parameter(torch.full(shape, init_scale))

    def forward(self, hidden):
        return self.gamma * hidden


class AttentionPool(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, activation, norm_eps):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.attention = CrossAttention(
            hidden_size,
            num_heads,
            head_dim=hidden_size // num_heads,
            qkv_bias=True,
            o_bias=True,
        )
        self.layernorm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size, activation, bias=True)

    def forward(self, hidden):
        probe = self.probe.expand(hidden.shape[0], -1, -1)
        pooled = self.attention(probe, hidden)
        return (pooled + self.mlp(self.layernorm(pooled)))[:, 0]


class PatchEmbed3D(nn.Module):
    def __init__(self, *, in_channels, embed_dim, kernel_size, stride=None, padding=0, bias=False):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        N, C, D, H, W = x.shape
        x = x.reshape(N, C, -1).permute(0, 2, 1)
        return x, (D, H, W)
