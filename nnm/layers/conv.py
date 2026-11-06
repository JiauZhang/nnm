import torch
import torch.nn.functional as F
from torch import nn

from nnm.layers.norm import LayerNorm2d


class CXBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim if use_dwconv else 1)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None

    def forward(self, hidden):
        residual = hidden
        hidden = self.dwconv(hidden)
        hidden = self.norm(hidden).permute(0, 2, 3, 1)
        hidden = self.pwconv2(self.act(self.pwconv1(hidden)))
        if self.gamma is not None:
            hidden = hidden * self.gamma
        return residual + hidden.permute(0, 3, 1, 2)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, hidden):
        return self.act(self.bn(self.conv(hidden)))


class ConvFFN(nn.Module):
    def __init__(self, channels, hidden_channels, dropout=0.0, norm=True):
        super().__init__()
        self.conv = (
            ConvNormAct(channels, channels, 7, groups=channels, act=False)
            if norm
            else nn.Sequential(nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False))
        )
        self.fc1 = nn.Conv2d(channels, hidden_channels, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden):
        hidden = self.dropout(self.act(self.fc1(self.conv(hidden))))
        return self.dropout(self.fc2(hidden))


class ResBlock2D(nn.Module):
    def __init__(self, channels, norm_eps=1e-6):
        super().__init__()
        self.norm1 = LayerNorm2d(channels, eps=norm_eps)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = LayerNorm2d(channels, eps=norm_eps)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        hidden = F.silu(self.norm1(x))
        hidden = self.conv1(hidden)
        hidden = F.silu(self.norm2(hidden))
        return x + self.conv2(hidden)


class ShortConv(nn.Module):
    def __init__(self, hidden_size, kernel_size, bias=False, use_cache=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.use_cache = use_cache
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, hidden_states, cache=None, attn_mask=None):
        seq_len = hidden_states.shape[1]
        if attn_mask is not None:
            hidden_states = hidden_states * attn_mask[:, :, None].to(hidden_states.dtype)
        gate, scale, value = self.in_proj(hidden_states).transpose(-1, -2).chunk(3, dim=-2)
        hidden = gate * value
        if attn_mask is not None:
            hidden = hidden * attn_mask[:, None, :].to(hidden.dtype)

        if self.use_cache and cache is not None:
            previous = None if cache.is_empty() else cache.state
            combined = hidden if previous is None else torch.cat([previous, hidden], dim=-1)
            combined = F.pad(combined, (self.kernel_size - 1, 0))
            conv_out = F.conv1d(combined, self.conv.weight, self.conv.bias, groups=self.hidden_size)
            conv_out = conv_out[..., -seq_len:]
            cache.update(hidden)
        else:
            conv_out = self.conv(hidden)[..., :seq_len]

        return self.out_proj((scale * conv_out).transpose(-1, -2).contiguous())
