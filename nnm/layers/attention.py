import torch
from torch import nn
import torch.nn.functional as F

from nnm.backends.sdpa import scaled_dot_product_attention
from nnm.layers.attention_mask import make_causal_mask
from nnm.layers.rope import apply_rope, apply_rope_pairwise


def causal_attention(
    query,
    key,
    value,
    *,
    cache=None,
    attn_mask=None,
    sliding_window=None,
    scale=None,
    dropout_p=0.0,
    causal=True,
):
    seq_len = query.shape[-2]
    if cache is not None and attn_mask is not None and attn_mask.dim() == 2 and attn_mask.shape[-1] < cache.kv_len + seq_len:
        padding = torch.ones(
            attn_mask.shape[0], cache.kv_len + seq_len - attn_mask.shape[-1], device=attn_mask.device, dtype=attn_mask.dtype
        )
        attn_mask = torch.cat([padding, attn_mask], dim=-1)

    cached = cache is not None
    if cached:
        cache_len = cache.kv_len
        key, value = cache.update(key, value)
    if cached or sliding_window is not None or attn_mask is not None:
        if attn_mask is not None and attn_mask.dim() == 4:
            mask = attn_mask
        else:
            mask = make_causal_mask(
                seq_len,
                key.shape[-2],
                cache_len if cached else 0,
                query.device,
                query.dtype,
                attn_mask=attn_mask,
                sliding_window=sliding_window,
            )
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


def split_qkv_bias(qkv_bias):
    if isinstance(qkv_bias, tuple):
        if len(qkv_bias) != 3:
            raise ValueError("qkv_bias tuple must contain (q, k, v)")
        return qkv_bias
    return qkv_bias, qkv_bias, qkv_bias


def split_qk_pos(qk_pos):
    if qk_pos is None:
        return None, None
    if isinstance(qk_pos, tuple):
        return qk_pos
    return qk_pos, qk_pos


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads=None,
        head_dim=None,
        v_head_dim=None,
        *,
        qkv_bias=False,
        o_bias=False,
        qk_norm=False,
        norm_layer=None,
        sliding_window=None,
        causal=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = (hidden_size // num_heads) if head_dim is None else head_dim
        self.v_head_dim = self.head_dim if v_head_dim is None else v_head_dim
        self.scaling = self.head_dim**-0.5
        self.sliding_window = sliding_window
        self.causal = causal

        q_bias, k_bias, v_bias = split_qkv_bias(qkv_bias)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=q_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=k_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.v_head_dim, bias=v_bias)
        self.o_proj = nn.Linear(num_heads * self.v_head_dim, hidden_size, bias=o_bias)

        if qk_norm and norm_layer is None:
            raise ValueError("qk_norm requires norm_layer")
        self.q_norm = norm_layer(self.head_dim) if qk_norm else None
        self.k_norm = norm_layer(self.head_dim) if qk_norm else None

    def project(self, x, query_pos=None, key_pos=None):
        query_input = x if query_pos is None else x + query_pos
        key_input = x if key_pos is None else x + key_pos
        return self.q_proj(query_input), self.k_proj(key_input), self.v_proj(x)

    def split_heads(self, query, key, value, *, lengths=None):
        batch, seq_len = query.shape[:2]
        query = query.view(batch, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch, seq_len, self.num_kv_heads, self.v_head_dim)
        query = self.q_norm(query) if self.q_norm is not None else query
        key = self.k_norm(key) if self.k_norm is not None else key

        if lengths is not None:
            return (
                query.reshape(-1, self.num_heads, self.head_dim),
                key.reshape(-1, self.num_kv_heads, self.head_dim),
                value.reshape(-1, self.num_kv_heads, self.v_head_dim),
            )
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    def attend_heads(self, query, key, value, *, cache=None, attn_mask=None, lengths=None):
        if lengths is not None:
            return ragged_attention(query, key, value, lengths, scale=self.scaling, enable_gqa=True, causal=self.causal)
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_mask = torch.where(attn_mask, 0.0, float("-inf"))[None, None]
        return causal_attention(
            query,
            key,
            value,
            cache=cache,
            attn_mask=attn_mask,
            sliding_window=self.sliding_window,
            scale=self.scaling,
            causal=self.causal,
        )

    def forward_heads(self, x, qk_pos=None, cache=None, attn_mask=None, lengths=None):
        query_pos, key_pos = split_qk_pos(qk_pos)
        query, key, value = self.project(x, query_pos, key_pos)
        query, key, value = self.split_heads(query, key, value, lengths=lengths)
        return self.attend_heads(query, key, value, cache=cache, attn_mask=attn_mask, lengths=lengths)

    def forward(self, x, qk_pos=None, cache=None, attn_mask=None, lengths=None):
        return self.o_proj(self.forward_heads(x, qk_pos=qk_pos, cache=cache, attn_mask=attn_mask, lengths=lengths))


class RotarySelfAttention(SelfAttention):
    def apply_rotary(self, query, key, rotary_embeddings, *, lengths=None):
        if rotary_embeddings is None:
            return query, key
        cos, sin = rotary_embeddings
        dim = 1 if lengths is not None else None
        return (
            apply_rope(query, cos, sin, dim=dim),
            apply_rope(key, cos, sin, dim=dim),
        )

    def forward_heads(self, x, rotary_embeddings=None, cache=None, attn_mask=None, lengths=None):
        query, key, value = self.project(x)
        query, key, value = self.split_heads(query, key, value, lengths=lengths)
        query, key = self.apply_rotary(query, key, rotary_embeddings, lengths=lengths)
        return self.attend_heads(query, key, value, cache=cache, attn_mask=attn_mask, lengths=lengths)

    def forward(self, x, rotary_embeddings=None, cache=None, attn_mask=None, lengths=None):
        return self.o_proj(
            self.forward_heads(
                x,
                rotary_embeddings=rotary_embeddings,
                cache=cache,
                attn_mask=attn_mask,
                lengths=lengths,
            )
        )


class InterleavedRotarySelfAttention(RotarySelfAttention):
    def apply_rotary(self, query, key, rotary_embeddings, *, lengths=None):
        if rotary_embeddings is None:
            return query, key
        cos, sin = rotary_embeddings
        dim = 1 if lengths is not None else None
        return (
            apply_rope_pairwise(query, cos, sin, dim=dim),
            apply_rope_pairwise(key, cos, sin, dim=dim),
        )


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim=None,
        v_head_dim=None,
        *,
        qkv_bias=False,
        o_bias=False,
        qk_norm=False,
        norm_layer=None,
        return_attention=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = (hidden_size // num_heads) if head_dim is None else head_dim
        self.v_head_dim = self.head_dim if v_head_dim is None else v_head_dim
        self.scaling = self.head_dim**-0.5
        self.return_attention = return_attention

        q_bias, k_bias, v_bias = split_qkv_bias(qkv_bias)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=q_bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=k_bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.v_head_dim, bias=v_bias)
        self.o_proj = nn.Linear(num_heads * self.v_head_dim, hidden_size, bias=o_bias)

        if qk_norm and norm_layer is None:
            raise ValueError("qk_norm requires norm_layer")
        self.q_norm = norm_layer(self.head_dim) if qk_norm else None
        self.k_norm = norm_layer(self.head_dim) if qk_norm else None

    def project(self, x, memory, query_pos=None, key_pos=None):
        batch, length = x.shape[:2]
        query_input = x if query_pos is None else x + query_pos
        key_input = memory if key_pos is None else memory + key_pos
        query = self.q_proj(query_input).view(batch, length, self.num_heads, self.head_dim)
        key = self.k_proj(key_input).view(batch, -1, self.num_heads, self.head_dim)
        value = self.v_proj(memory).view(batch, -1, self.num_heads, self.v_head_dim)
        return query, key, value

    def split_heads(self, query, key, value):
        query = self.q_norm(query).transpose(1, 2) if self.q_norm is not None else query.transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2) if self.k_norm is not None else key.transpose(1, 2)
        return query, key, value.transpose(1, 2)

    def attend_heads(self, query, key, value, *, attn_mask=None):
        batch, length = query.shape[0], query.shape[-2]
        if self.return_attention:
            weights = (query @ key.transpose(-1, -2) * self.scaling).softmax(-1)
            self.attention = weights.detach()
            output = (weights @ value).transpose(1, 2).reshape(batch, length, -1)
        else:
            output = scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=self.scaling
            )
            output = output.transpose(1, 2).reshape(batch, length, -1)
        return output

    def forward_heads(self, x, memory, qk_pos=None, attn_mask=None):
        query_pos, key_pos = split_qk_pos(qk_pos)
        query, key, value = self.project(x, memory, query_pos, key_pos)
        query, key, value = self.split_heads(query, key, value)
        return self.attend_heads(query, key, value, attn_mask=attn_mask)

    def forward(self, x, memory, qk_pos=None, attn_mask=None):
        return self.o_proj(self.forward_heads(x, memory, qk_pos=qk_pos, attn_mask=attn_mask))


class RotaryCrossAttention(CrossAttention):
    def forward_heads(self, x, memory, rotary_embeddings=None, attn_mask=None):
        query, key, value = self.project(x, memory)
        query, key, value = self.split_heads(query, key, value)
        if rotary_embeddings is not None:
            cos, sin = rotary_embeddings
            query = apply_rope(query, cos, sin)
            key = apply_rope(key, cos, sin)
        return self.attend_heads(query, key, value, attn_mask=attn_mask)

    def forward(self, x, memory, rotary_embeddings=None, attn_mask=None):
        return self.o_proj(self.forward_heads(x, memory, rotary_embeddings=rotary_embeddings, attn_mask=attn_mask))


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


def window_partition_padded(x, window_size: int):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = H + pad_h, W + pad_w
    x = x.view(B, padded_h // window_size, window_size, padded_w // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (padded_h, padded_w)


def window_unpartition_padded(windows, window_size: int, padded_hw, hw):
    padded_h, padded_w = padded_hw
    height, width = hw
    batch = windows.shape[0] // (padded_h * padded_w // window_size // window_size)
    x = windows.view(batch, padded_h // window_size, padded_w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, padded_h, padded_w, -1)
    return x[:, :height, :width].contiguous()


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
