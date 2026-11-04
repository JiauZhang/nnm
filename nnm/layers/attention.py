import torch
from torch import nn
import torch.nn.functional as F

from nnm.backends.sdpa import scaled_dot_product_attention
from nnm.layers.attention_mask import make_causal_mask
from nnm.layers.rope import apply_rope


def causal_attention(
    query,
    key,
    value,
    *,
    cos=None,
    sin=None,
    cache=None,
    attn_mask=None,
    sliding_window=None,
    scale=None,
    rope_dim=None,
    dropout_p=0.0,
    causal=True,
):
    if cos is not None:
        query = apply_rope(query, cos, sin, dim=rope_dim)
        key = apply_rope(key, cos, sin, dim=rope_dim)

    seq_len = query.shape[-2]
    if cache is not None and attn_mask is not None and attn_mask.dim() == 2 and attn_mask.shape[-1] < cache.kv_len + seq_len:
        padding = torch.ones(
            attn_mask.shape[0], cache.kv_len + seq_len - attn_mask.shape[-1], device=attn_mask.device, dtype=attn_mask.dtype
        )
        attn_mask = torch.cat([padding, attn_mask], dim=-1)

    def build_mask(kv_len, cache_len):
        if attn_mask is not None and attn_mask.dim() == 4:
            return attn_mask
        return make_causal_mask(
            seq_len,
            kv_len,
            cache_len,
            query.device,
            query.dtype,
            attn_mask=attn_mask,
            sliding_window=sliding_window,
        )

    if cache is not None:
        cache_len = cache.kv_len
        key, value = cache.update(key, value)
        mask = build_mask(key.shape[-2], cache_len)
        is_causal = False
    elif sliding_window is not None or attn_mask is not None:
        mask = build_mask(key.shape[-2], 0)
        is_causal = False
    else:
        mask = None
        is_causal = causal

    output = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        enable_gqa=True,
        scale=scale,
    )
    return output.transpose(1, 2).flatten(2)


def ragged_attention(query, key, value, lengths, *, scale=None, enable_gqa=False, causal=False):
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.tolist()
    lengths = [length for length in lengths if length > 0]
    if not lengths:
        return query.new_zeros((0, query.shape[1] * query.shape[2]))

    heads_dim = query.shape[1] * query.shape[2]
    if len(set(lengths)) == 1:
        size = lengths[0]
        query = query.reshape(-1, size, query.shape[1], query.shape[2]).transpose(1, 2)
        key = key.reshape(-1, size, key.shape[1], key.shape[2]).transpose(1, 2)
        value = value.reshape(-1, size, value.shape[1], value.shape[2]).transpose(1, 2)
        output = scaled_dot_product_attention(query, key, value, is_causal=causal, enable_gqa=enable_gqa, scale=scale)
        return output.transpose(1, 2).reshape(-1, heads_dim)

    outputs = []
    for query_i, key_i, value_i in zip(*[torch.split(tensor, lengths) for tensor in (query, key, value)]):
        output = scaled_dot_product_attention(
            query_i.transpose(0, 1)[None],
            key_i.transpose(0, 1)[None],
            value_i.transpose(0, 1)[None],
            is_causal=causal,
            enable_gqa=enable_gqa,
            scale=scale,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0).reshape(-1, heads_dim)


class ChannelAttention(nn.Module):
    def __init__(self, dim, groups, qkv_bias=True):
        super().__init__()
        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        B, L, C = x.shape
        assert C % self.groups == 0, "input feature has wrong channel"
        # [B, L, C] --> [B, L, 3C] --> [B, L, 3, g, C//g] --> [3, B, g, L, C//g]
        qkv = self.qkv(x).reshape(B, L, 3, self.groups, C // self.groups).permute(2, 0, 3, 1, 4)
        # [B, g, L, C//g]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * (float(L) ** -0.5)
        # [B, g, C//g, C//g]
        attention = q.transpose(-1, -2) @ k
        attention = attention.softmax(dim=-1)
        # [B, g, C//g, C//g] @ [B, g, C//g, L] --> [B, g, C//g, L] --> [B, g, L, C//g]
        x = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        # [B, g, L, C//g] --> [B, L, g, C//g] --> [B, L, C]
        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x, size


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    # [B, H//w, w, W//w, w, C] --> [B, H//w, W//w, w, w, C] --> [-1, w * w, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size: int, output_size):
    B, H, W, C = output_size
    x = windows.reshape(-1, window_size, window_size, C)
    x = x.reshape(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(head_dim) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.reshape(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)

        B_, L, C = x.shape
        qkv = self.qkv(x).reshape(B_, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, L, C)
        x = self.proj(x)

        x = window_reverse(x, self.window_size, (B, Hp, Wp, C))

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.reshape(B, H * W, C)

        return x, size
