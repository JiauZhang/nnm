import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        hidden = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return hidden.type_as(x) * self.weight


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('inv_freq', 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, q, k):
        positions = torch.arange(q.shape[-2], device=q.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, key_padding_mask=None):
        batch, length, _ = x.shape
        query, key, value = (
            projection(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        query, key = self.rotary(query, key)

        if key_padding_mask is None:
            out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        else:
            keep = key_padding_mask.bool()[:, None, None, :]
            causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=keep & causal)

        return self.out_proj(out.transpose(1, 2).reshape(batch, length, -1))


class MultiHeadCrossAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, query, key, value, key_padding_mask=None):
        batch, q_len, _ = query.shape
        seq_len = key.shape[1]

        query = self.q_proj(query).view(batch, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(key).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(value).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        query, key = self.rotary(query, key)

        attn_mask = None if key_padding_mask is None else key_padding_mask.bool()[:, None, None, :]
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        return self.out_proj(out.transpose(1, 2).reshape(batch, q_len, -1))


class HierarchicalEmbedding(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model=256):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.d_model = d_model
        self.emb_s1 = nn.Embedding(2**s1_bits, d_model)
        self.emb_s2 = nn.Embedding(2**s2_bits, d_model)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

    def split_token(self, token_ids):
        mask = (1 << self.s2_bits) - 1
        tokens = token_ids.long()
        return tokens >> self.s2_bits, tokens & mask

    def forward(self, token_ids):
        s1_ids, s2_ids = token_ids if isinstance(token_ids, (tuple, list)) else self.split_token(token_ids)
        s1_emb = self.emb_s1(s1_ids) * math.sqrt(self.d_model)
        s2_emb = self.emb_s2(s2_ids) * math.sqrt(self.d_model)
        return self.fusion_proj(torch.cat([s1_emb, s2_emb], dim=-1))


class DependencyAwareLayer(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads)
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
        attn_out = self.cross_attn(
            query=sibling_embed, key=hidden_states, value=hidden_states, key_padding_mask=key_padding_mask
        )
        return self.norm(hidden_states + attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim)

    def forward(self, x, key_padding_mask=None):
        x = x + self.self_attn(self.norm1(x), key_padding_mask=key_padding_mask)
        return x + self.ffn(self.norm2(x))


class DualHead(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model):
        super().__init__()
        self.proj_s1 = nn.Linear(d_model, 2**s1_bits)
        self.proj_s2 = nn.Linear(d_model, 2**s2_bits)

    def forward(self, x):
        return self.proj_s1(x)

    def cond_forward(self, x):
        return self.proj_s2(x)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        weight = torch.zeros(c_in, d_model)
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    fields = ('minute', 'hour', 'weekday', 'day', 'month')
    sizes = (60, 24, 7, 32, 13)

    def __init__(self, d_model, learn_pe):
        super().__init__()
        embed = nn.Embedding if learn_pe else FixedEmbedding
        for field, size in zip(self.fields, self.sizes):
            setattr(self, f'{field}_embed', embed(size, d_model))

    def forward(self, x):
        stamps = x.long()
        return sum(getattr(self, f'{field}_embed')(stamps[:, :, index]) for index, field in enumerate(self.fields))


class BinarySphericalQuantizer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer('basis', 2 ** torch.arange(embed_dim - 1, -1, -1))

    def forward(self, z):
        zhat = torch.where(z > 0, torch.ones_like(z), -torch.ones_like(z))
        return (z + (zhat - z).detach()) / (self.embed_dim**0.5)

    def codes_to_indexes(self, zhat):
        return ((zhat + 1) / 2 * self.basis).sum(-1).to(torch.int64)

    def indexes_to_codes(self, indices):
        codes = torch.remainder(torch.floor_divide(indices.unsqueeze(-1), self.basis), 2)
        return codes * 2 - 1


class BSQuantizer(nn.Module):
    def __init__(self, s1_bits, s2_bits):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits
        self.bsq = BinarySphericalQuantizer(self.codebook_dim)

    def bits_to_indices(self, bits):
        weights = 2 ** torch.arange(bits.shape[-1], dtype=torch.long, device=bits.device)
        return ((bits >= 0).to(torch.long) * weights).sum(-1)

    def forward(self, z, half=False):
        codes = self.bsq(F.normalize(z, dim=-1))
        if half:
            return [
                self.bits_to_indices(codes[..., : self.s1_bits]),
                self.bits_to_indices(codes[..., self.s1_bits :]),
            ]
        return self.bits_to_indices(codes)
