import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from nnm.layers.attention import RotaryCrossAttention, RotarySelfAttention
from nnm.layers.mlp import GatedMLP as FeedForward
from nnm.layers.norm import RMSNorm
from nnm.layers.sinusoidal import sinusoidal_embedding
from nnm.layers.rope import RoPE, rotary_inv_freq
from nnm.layers.transformer import TransformerBlock as SharedTransformerBlock


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
        self.cross_attn = RotaryCrossAttention(d_model, n_heads, qkv_bias=True, o_bias=True)
        self.norm = RMSNorm(d_model)
        self.rotary = RoPE(rotary_inv_freq(d_model // n_heads, 10000))

    def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
        positions = torch.arange(sibling_embed.shape[1], device=sibling_embed.device)
        rotary_embeddings = self.rotary(positions)
        attn_mask = None if key_padding_mask is None else key_padding_mask.bool()[:, None, None, :]
        attn_out = self.cross_attn(sibling_embed, hidden_states, rotary_embeddings, attn_mask=attn_mask)
        return self.norm(hidden_states + attn_out)


class TransformerBlock(SharedTransformerBlock):
    def __init__(self, d_model, n_heads, ff_dim):
        super().__init__(
            d_model,
            RotarySelfAttention(d_model, n_heads, qkv_bias=True, o_bias=True, causal=True),
            FeedForward(d_model, ff_dim),
            norm_layer=partial(RMSNorm, eps=1e-5),
        )
        self.rotary = RoPE(rotary_inv_freq(d_model // n_heads, 10000))

    def forward(self, x, key_padding_mask=None):
        positions = torch.arange(x.shape[1], device=x.device)
        rotary_embeddings = self.rotary(positions)
        if key_padding_mask is None:
            return super().forward(x, rotary_embeddings=rotary_embeddings)
        causal = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).tril()
        keep = key_padding_mask.bool()[:, None, None, :] & causal
        attn_mask = torch.where(keep, 0.0, float("-inf"))
        return super().forward(x, rotary_embeddings=rotary_embeddings, attn_mask=attn_mask)


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
        weight = sinusoidal_embedding(torch.arange(c_in), d_model, interleaved=True)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    fields = ("minute", "hour", "weekday", "day", "month")
    sizes = (60, 24, 7, 32, 13)

    def __init__(self, d_model, learn_pe):
        super().__init__()
        embed = nn.Embedding if learn_pe else FixedEmbedding
        for field, size in zip(self.fields, self.sizes):
            setattr(self, f"{field}_embed", embed(size, d_model))

    def forward(self, x):
        stamps = x.long()
        return sum(getattr(self, f"{field}_embed")(stamps[:, :, index]) for index, field in enumerate(self.fields))


class BinarySphericalQuantizer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("basis", 2 ** torch.arange(embed_dim - 1, -1, -1))

    def forward(self, z):
        zhat = torch.where(z > 0, torch.ones_like(z), -torch.ones_like(z))
        return (z + (zhat - z).detach()) / (self.embed_dim**0.5)

    def codes_to_indexes(self, zhat):
        return ((zhat + 1) / 2 * self.basis).sum(-1).to(torch.int64)

    def indexes_to_codes(self, indices):
        codes = torch.remainder(torch.floor_divide(indices.unsqueeze(-1), self.basis), 2)
        return codes * 2 - 1


class BSQuantizer(nn.Module):
    def __init__(self, s1_bits, s2_bits, group_size):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits
        self.bsq = BinarySphericalQuantizer(self.codebook_dim)
        self.bsq.register_buffer("group_basis", torch.ones(group_size))

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
