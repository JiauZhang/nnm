import torch
from torch import nn

from nnm.backends.sdpa import scaled_dot_product_attention
from nnm.cache.attention import SlidingWindowKVCache
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
    query_length = query.shape[-2]
    if cache is not None:
        key, value = cache.update(key, value)
    key_length = key.shape[-2]
    stored_before = key_length - query_length
    window = sliding_window
    if window is None and isinstance(cache, SlidingWindowKVCache):
        window = cache.window_size

    decode_fast_path = (
        cache is not None
        and query_length == 1
        and attn_mask is None
        and (window is None or isinstance(cache, SlidingWindowKVCache))
    )
    training_fast_path = cache is None and attn_mask is None and window is None

    if training_fast_path:
        mask = None
        is_causal = causal
    elif decode_fast_path:
        mask = None
        is_causal = False
    elif attn_mask is not None and attn_mask.dim() == 4:
        mask = attn_mask
        is_causal = False
    else:
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(query.dtype)
            if attn_mask.dim() == 2:
                if attn_mask.shape[-1] < key_length:
                    padding = torch.ones(
                        attn_mask.shape[0],
                        key_length - attn_mask.shape[-1],
                        device=attn_mask.device,
                        dtype=attn_mask.dtype,
                    )
                    attn_mask = torch.cat([padding, attn_mask], dim=-1)
                elif attn_mask.shape[-1] > key_length:
                    attn_mask = attn_mask[..., -key_length:]
        mask = make_causal_mask(
            query_length,
            key_length,
            stored_before,
            query.device,
            query.dtype,
            attn_mask=attn_mask,
            sliding_window=window,
            causal=causal,
        )
        is_causal = False

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
            attn_mask = torch.where(attn_mask, 0.0, float("-inf"))
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None]
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
