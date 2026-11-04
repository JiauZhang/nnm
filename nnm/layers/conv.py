import torch
import torch.nn.functional as F
from torch import nn


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
