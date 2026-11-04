import torch
from torch import nn

from nnm.backends.sdpa import scaled_dot_product_attention
from nnm.layers.activation import get_activation
from nnm.layers.attention import causal_attention, ragged_attention
from nnm.layers.norm import RMSNorm
from nnm.layers.rope import apply_rope


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation="gelu", bias=True):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


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
        q_norm=False,
        k_norm=False,
        norm_eps=1e-6,
        norm_factory=None,
        output_gate=None,
        query_gate=False,
        sliding_window=None,
        causal=True,
        rope=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = (hidden_size // num_heads) if head_dim is None else head_dim
        self.v_head_dim = self.head_dim if v_head_dim is None else v_head_dim
        self.scaling = self.head_dim**-0.5
        self.sliding_window = sliding_window
        self.causal = causal
        self.query_gate = query_gate
        self.output_gate = output_gate
        self.rope = rope

        self.q_proj = nn.Linear(hidden_size, (2 if query_gate else 1) * num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.v_head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * self.v_head_dim, hidden_size, bias=o_bias)

        factory = norm_factory or (lambda dim: RMSNorm(dim, norm_eps))
        self.q_norm = factory(self.head_dim) if q_norm else None
        self.k_norm = factory(self.head_dim) if k_norm else None
        self.gate_proj = (
            nn.Linear(hidden_size, num_heads if output_gate == "head" else num_heads * self.v_head_dim, bias=False)
            if output_gate
            else None
        )

    def project(self, x):
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def forward(self, x, cos=None, sin=None, cache=None, attn_mask=None, lengths=None):
        batch, seq_len = x.shape[:2]
        query, key, value = self.project(x)

        if self.query_gate:
            query = query.view(batch, seq_len, self.num_heads, 2 * self.head_dim)
            query, query_gate = query.chunk(2, dim=-1)
            query_gate = query_gate.reshape(batch, seq_len, -1)
        else:
            query = query.view(batch, seq_len, self.num_heads, self.head_dim)
            query_gate = None
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch, seq_len, self.num_kv_heads, self.v_head_dim)

        query = self.q_norm(query) if self.q_norm is not None else query
        key = self.k_norm(key) if self.k_norm is not None else key

        if lengths is not None:
            query = query.reshape(-1, self.num_heads, self.head_dim)
            key = key.reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.reshape(-1, self.num_kv_heads, self.v_head_dim)
        else:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        if cos is None and self.rope is not None and lengths is None:
            start = cache.kv_len if cache is not None and not cache.is_empty() else 0
            cos, sin = self.rope(torch.arange(start, start + seq_len, device=x.device))
        if cos is not None:
            query = apply_rope(query, cos, sin, dim=1 if lengths is not None else None)
            key = apply_rope(key, cos, sin, dim=1 if lengths is not None else None)

        if lengths is not None:
            output = ragged_attention(query, key, value, lengths, scale=self.scaling, enable_gqa=True, causal=self.causal)
        else:
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                attn_mask = torch.where(attn_mask, 0.0, float("-inf"))[None, None]
            output = causal_attention(
                query,
                key,
                value,
                cache=cache,
                attn_mask=attn_mask,
                sliding_window=self.sliding_window,
                scale=self.scaling,
                causal=self.causal,
            )

        if query_gate is not None:
            output = output * torch.sigmoid(query_gate)
        if self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(x))
            if gate.shape[-1] == self.num_heads:
                output = (output.view(batch, seq_len, self.num_heads, self.v_head_dim) * gate.unsqueeze(-1)).reshape(
                    batch, seq_len, -1
                )
            else:
                output = output * gate
        return self.o_proj(output)


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
        q_norm=False,
        k_norm=False,
        norm_eps=1e-6,
        norm_factory=None,
        output_gate=None,
        return_attention=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = (hidden_size // num_heads) if head_dim is None else head_dim
        self.v_head_dim = self.head_dim if v_head_dim is None else v_head_dim
        self.scaling = self.head_dim**-0.5
        self.return_attention = return_attention
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.v_head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * self.v_head_dim, hidden_size, bias=o_bias)
        factory = norm_factory or (lambda dim: RMSNorm(dim, norm_eps))
        self.q_norm = factory(self.head_dim) if q_norm else None
        self.k_norm = factory(self.head_dim) if k_norm else None
        self.gate_proj = (
            nn.Linear(hidden_size, num_heads if output_gate == "head" else num_heads * self.v_head_dim, bias=False)
            if output_gate
            else None
        )

    def forward(self, x, memory):
        batch, length = x.shape[:2]
        query = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim)
        key = self.k_proj(memory).view(batch, -1, self.num_heads, self.head_dim)
        value = self.v_proj(memory).view(batch, -1, self.num_heads, self.v_head_dim)
        query = self.q_norm(query).transpose(1, 2) if self.q_norm is not None else query.transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2) if self.k_norm is not None else key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.return_attention:
            weights = (query @ key.transpose(-1, -2) * self.scaling).softmax(-1)
            self.attention = weights.detach()
            output = (weights @ value).transpose(1, 2).reshape(batch, length, -1)
        else:
            output = scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, scale=self.scaling)
            output = output.transpose(1, 2).reshape(batch, length, -1)
        if self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(x))
            if gate.shape[-1] == self.num_heads:
                output = (output.view(batch, length, self.num_heads, self.v_head_dim) * gate.unsqueeze(-1)).reshape(
                    batch, length, -1
                )
            else:
                output = output * gate
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, norm_eps, self_attn, mlp, norm_factory=None):
        super().__init__()
        factory = norm_factory or (lambda dim: RMSNorm(dim, norm_eps))
        self.input_layernorm = factory(hidden_size)
        self.post_attention_layernorm = factory(hidden_size)
        self.self_attn = self_attn
        self.mlp = mlp

    def forward(self, x, **kwargs):
        x = x + self.self_attn(self.input_layernorm(x), **kwargs)
        return x + self.mlp(self.post_attention_layernorm(x))


class VisionBlock(nn.Module):
    def __init__(self, hidden_size, norm_eps, attn, mlp, norm_factory=None):
        super().__init__()
        factory = norm_factory or (lambda dim: RMSNorm(dim, norm_eps))
        self.norm_1 = factory(hidden_size)
        self.norm_2 = factory(hidden_size)
        self.attn = attn
        self.mlp = mlp

    def forward(self, x, **kwargs):
        x = x + self.attn(self.norm_1(x), **kwargs)
        return x + self.mlp(self.norm_2(x))
