import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nnm.layers.attention import CrossAttention, RotarySelfAttention

HADA_PERM_SEEDS = (11, 13)
HADA_COND_RANK = 8


def hada_perms(n, split=False):
    if split:
        half = n // 2
        return tuple(
            torch.from_numpy(
                np.concatenate(
                    [np.random.RandomState(seed).permutation(half), half + np.random.RandomState(seed + 977).permutation(half)]
                )
            )
            for seed in HADA_PERM_SEEDS
        )
    return tuple(torch.from_numpy(np.random.RandomState(seed).permutation(n)) for seed in HADA_PERM_SEEDS)


def _hada_blocks(n):
    block = 1 << ((n - 1).bit_length() // 2)
    return block, n // block


def _kron_apply(z, a, b):
    lead = z.shape[:-1]
    z = z.reshape(*lead, a.shape[0], b.shape[0])
    return torch.einsum("...ij,ik,jl->...kl", z, a, b).reshape(*lead, a.shape[0] * b.shape[0])


class HadamardMLP(nn.Module):
    def __init__(self, d_model, hada_n, split_perms=False):
        super().__init__()
        self.d_model, self.hada_n = d_model, hada_n
        block_a, block_b = _hada_blocks(hada_n)
        self.split_perms = split_perms
        for stage in (1, 2, 3):
            self.register_parameter(f"w{stage}a", nn.Parameter(torch.zeros(block_a, block_a)))
            self.register_parameter(f"w{stage}b", nn.Parameter(torch.zeros(block_b, block_b)))
        self.d1 = nn.Parameter(torch.zeros(hada_n))
        self.d2 = nn.Parameter(torch.zeros(hada_n))
        self.b2 = nn.Parameter(torch.zeros(hada_n))
        self.d3 = nn.Parameter(torch.zeros(hada_n))
        self.d4 = nn.Parameter(torch.zeros(hada_n))
        self.cond_v = nn.Parameter(torch.zeros(d_model, HADA_COND_RANK))
        self.cond_u = nn.Parameter(torch.zeros(HADA_COND_RANK, hada_n))
        p1, p2 = hada_perms(hada_n, split_perms)
        self.register_buffer("perm1", p1, persistent=False)
        self.register_buffer("perm2", p2, persistent=False)

    def forward(self, x):
        n = self.hada_n
        cond = 1 + torch.softmax(x @ self.cond_v, -1) @ self.cond_u
        pad = n - self.d_model
        z = torch.nn.functional.pad(x, (0, pad)) if pad else x
        z = _kron_apply(self.d1 * z, self.w1a, self.w1b)[..., self.perm1]
        z = _kron_apply(torch.nn.functional.silu(self.d2 * cond * z + self.b2), self.w2a, self.w2b)[..., self.perm2]
        z = _kron_apply(self.d3 * z, self.w3a, self.w3b)
        return (self.d4 * z)[..., : self.d_model]


class WhistleSelfAttention(RotarySelfAttention):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, v_head_dim, **kwargs):
        super().__init__(hidden_size, num_heads, num_kv_heads, head_dim, v_head_dim, **kwargs)
        self.gate_proj = nn.Linear(hidden_size, num_heads * v_head_dim, bias=False)

    def forward(self, x, rotary_embeddings=None, attn_mask=None, cache=None):
        output = self.forward_heads(
            x,
            rotary_embeddings=rotary_embeddings,
            attn_mask=attn_mask,
            cache=cache,
        )
        return self.o_proj(output * torch.sigmoid(self.gate_proj(x)))


class ConvSelfAttention(WhistleSelfAttention):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, v_head_dim, qkv_conv, **kwargs):
        super().__init__(hidden_size, num_heads, num_kv_heads, head_dim, v_head_dim, **kwargs)
        self.q_conv = nn.Conv1d(num_heads * head_dim, num_heads * head_dim, qkv_conv, groups=num_heads * head_dim, bias=False)
        self.k_conv = nn.Conv1d(
            num_kv_heads * head_dim, num_kv_heads * head_dim, qkv_conv, groups=num_kv_heads * head_dim, bias=False
        )
        self.v_conv = nn.Conv1d(
            num_kv_heads * v_head_dim, num_kv_heads * v_head_dim, qkv_conv, groups=num_kv_heads * v_head_dim, bias=False
        )

    def project(self, x, query_pos=None, key_pos=None):
        query, key, value = super().project(x, query_pos, key_pos)
        return _apply_conv(query, self.q_conv), _apply_conv(key, self.k_conv), _apply_conv(value, self.v_conv)


class WhistleCrossAttention(CrossAttention):
    def __init__(self, hidden_size, num_heads, head_dim, v_head_dim, **kwargs):
        super().__init__(hidden_size, num_heads, head_dim, v_head_dim, **kwargs)
        self.gate_proj = nn.Linear(hidden_size, num_heads * v_head_dim, bias=False)

    def forward(self, x, memory, attn_mask=None):
        output = self.forward_heads(x, memory, attn_mask=attn_mask)
        return self.o_proj(output * torch.sigmoid(self.gate_proj(x)))


def _apply_conv(x, conv):
    hidden = x.transpose(1, 2)
    hidden = F.pad(hidden, (conv.kernel_size[0] - 1, 0))
    return conv(hidden).transpose(1, 2)
