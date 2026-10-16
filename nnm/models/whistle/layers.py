import numpy as np
import torch
from torch import nn

HADA_PERM_SEEDS = (11, 13)
HADA_COND_RANK = 8
SINKHORN_ITERS = 20
RMS_EPS = 1e-6


class ZCRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return (1 + self.scale) * x / rms


def rms_unit(x, eps=RMS_EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def sinkhorn(logits, iters=SINKHORN_ITERS):
    for _ in range(iters):
        logits = logits - torch.logsumexp(logits, -1, keepdim=True)
        logits = logits - torch.logsumexp(logits, -2, keepdim=True)
    return logits.exp()


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


class MHCStack(nn.Module):
    def __init__(self, d_model, num_layers, lanes):
        super().__init__()
        self.d_model, self.num_layers, self.lanes = d_model, num_layers, lanes
        n_lane = lanes * d_model
        self.a_pre = nn.Parameter(torch.zeros(num_layers))
        self.a_post = nn.Parameter(torch.zeros(num_layers))
        self.a_res = nn.Parameter(torch.zeros(num_layers))
        self.b_pre = nn.Parameter(torch.zeros(num_layers, lanes))
        self.b_post = nn.Parameter(torch.zeros(num_layers, lanes))
        self.b_res = nn.Parameter(torch.zeros(num_layers, lanes, lanes))
        self.phi_pre = nn.Parameter(torch.zeros(num_layers, lanes, n_lane))
        self.phi_post = nn.Parameter(torch.zeros(num_layers, lanes, n_lane))
        self.phi_res = nn.Parameter(torch.zeros(num_layers, lanes * lanes, n_lane))
        self.register_buffer("lane", torch.eye(lanes), persistent=False)
        self.register_buffer("lane_index", torch.arange(num_layers) % lanes, persistent=False)

    def layer_params(self, index):
        lane = self.lane[self.lane_index[index]]
        return dict(
            pre_off=8 * lane - 4,
            post_off=-4 * (1 - lane),
            phi_pre=self.phi_pre[index].T,
            phi_post=self.phi_post[index].T,
            phi_res=self.phi_res[index].T,
            a_pre=self.a_pre[index],
            a_post=self.a_post[index],
            a_res=self.a_res[index],
            b_pre=self.b_pre[index],
            b_post=self.b_post[index],
            b_res=self.b_res[index],
        )

    def advance(self, stream, block, params):
        batch, length, lanes, channels = stream.shape
        wide = rms_unit(stream.reshape(batch, length, lanes * channels))
        h_pre = torch.sigmoid(params["a_pre"] * (wide @ params["phi_pre"]) + params["b_pre"] + params["pre_off"])
        mixed = torch.einsum("btn,btnc->btc", h_pre, stream)
        delta = block(mixed) - mixed
        h_post = 2 * torch.sigmoid(params["a_post"] * (wide @ params["phi_post"]) + params["b_post"] + params["post_off"])
        h_res = sinkhorn(params["a_res"] * (wide @ params["phi_res"]).reshape(batch, length, lanes, lanes) + params["b_res"])
        return torch.einsum("btij,btjc->btic", h_res, stream) + h_post[..., None] * delta[:, :, None, :]


class RoPE(nn.Module):
    def __init__(self, head_dim, theta):
        super().__init__()
        self.head_dim, self.theta = head_dim, theta

    def forward(self, x):
        length = x.shape[-2]
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=x.device) / self.head_dim))
        angle = torch.arange(length, dtype=torch.float32, device=x.device)[:, None] * freqs[None]
        cos, sin = angle.cos()[None, None], angle.sin()[None, None]
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SelfAttention(nn.Module):
    def __init__(
        self, d_model, num_heads, num_kv_heads, qk_head_dim, v_head_dim, num_layers, theta, qkv_conv_taps=0, rope=True
    ):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.qk_head_dim, self.v_head_dim = qk_head_dim, v_head_dim
        self.repeats = num_heads // num_kv_heads
        self.scaling = qk_head_dim**-0.5
        self.q_proj = nn.Linear(d_model, num_heads * qk_head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * qk_head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * v_head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * v_head_dim, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, num_heads * v_head_dim, bias=False)
        self.q_norm = ZCRMSNorm(qk_head_dim)
        self.k_norm = ZCRMSNorm(qk_head_dim)
        self.taps = qkv_conv_taps
        if self.taps:
            self.q_taps = nn.Parameter(torch.zeros(self.taps, num_heads * qk_head_dim))
            self.k_taps = nn.Parameter(torch.zeros(self.taps, num_kv_heads * qk_head_dim))
            self.v_taps = nn.Parameter(torch.zeros(self.taps, num_kv_heads * v_head_dim))
        self.rope = RoPE(qk_head_dim, theta) if rope else None

    def forward(self, x, mask=None):
        batch, length, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.taps:
            q = _apply_taps(q, self.q_taps, self.taps, mask)
            k = _apply_taps(k, self.k_taps, self.taps, mask)
            v = _apply_taps(v, self.v_taps, self.taps, mask)
        q = self.q_norm(q.reshape(batch, length, self.num_heads, self.qk_head_dim).transpose(1, 2))
        k = self.k_norm(k.reshape(batch, length, self.num_kv_heads, self.qk_head_dim).transpose(1, 2))
        v = v.reshape(batch, length, self.num_kv_heads, self.v_head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        if self.repeats > 1:
            k = k.repeat_interleave(self.repeats, 1)
            v = v.repeat_interleave(self.repeats, 1)
        scores = q @ k.transpose(-1, -2) * self.scaling
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, -1)
        return self.out_proj(out * torch.sigmoid(self.gate_proj(x)))


def _apply_taps(z, taps, count, mask):
    length = z.shape[1]
    out = taps[0] * z
    for tap in range(1, count):
        shifted = torch.cat([torch.zeros_like(z[:, :tap]), z[:, :-tap]], dim=1)
        if mask is not None:
            shifted = shifted * (torch.arange(length, device=z.device) >= tap).to(z.dtype)[None, :, None]
        out = out + taps[tap] * shifted
    return out


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, qk_head_dim, v_head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.qk_head_dim, self.v_head_dim = qk_head_dim, v_head_dim
        self.scaling = qk_head_dim**-0.5
        self.q_proj = nn.Linear(d_model, num_heads * qk_head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * qk_head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * v_head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * v_head_dim, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, num_heads * v_head_dim, bias=False)
        self.q_norm = ZCRMSNorm(qk_head_dim)
        self.k_norm = ZCRMSNorm(qk_head_dim)

    def forward(self, x, memory):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.num_heads, self.qk_head_dim)
        k = self.k_proj(memory).reshape(batch, -1, self.num_heads, self.qk_head_dim)
        v = self.v_proj(memory).reshape(batch, -1, self.num_heads, self.v_head_dim)
        q = self.q_norm(q.transpose(1, 2))
        k = self.k_norm(k.transpose(1, 2))
        v = v.transpose(1, 2)
        scores = q @ k.transpose(-1, -2) * self.scaling
        weights = scores.softmax(-1)
        self.attention = weights.detach()
        out = (weights @ v).transpose(1, 2).reshape(batch, length, -1)
        return self.out_proj(out * torch.sigmoid(self.gate_proj(x)))


class Engram(nn.Module):
    def __init__(self, d_model, tables, slots, sub_dim, num_layers, dilation, orders, heads):
        super().__init__()
        self.d_model, self.tables, self.slots, self.sub_dim = d_model, tables, slots, sub_dim
        self.dilation, self.orders, self.heads = dilation, orders, heads
        self.embedding = nn.Parameter(torch.zeros(tables, slots, sub_dim))
        self.key_proj = nn.Linear(tables * sub_dim, d_model, bias=False)
        self.value_proj = nn.Linear(tables * sub_dim, d_model, bias=False)
        self.taps = nn.Parameter(torch.zeros(4, d_model))

    def forward(self, tokens):
        length = tokens.shape[-1]
        positions = torch.arange(length, device=tokens.device)
        tables = len(self.orders) * self.heads
        indices = engram_indices(tokens, self.orders, self.heads, self.slots)
        offsets = torch.arange(tables, device=tokens.device) * self.slots
        flat = (indices + offsets).reshape(-1)
        fetched = self.embedding.reshape(-1, self.sub_dim)[flat]
        fetched = fetched.reshape(*tokens.shape, tables, self.sub_dim)
        order_mask = torch.stack([positions >= order - 1 for order in self.orders for _ in range(self.heads)], -1)
        fetched = (fetched * order_mask.to(fetched.dtype)[None, :, :, None]).reshape(*tokens.shape, -1)
        key = self.key_proj(fetched)
        value = self.value_proj(fetched)
        tapped = self.taps[0] * value
        for tap in range(1, self.taps.shape[0]):
            offset = tap * self.dilation
            shifted = (
                torch.cat([torch.zeros_like(value[:, :offset]), value[:, :-offset]], dim=1)
                if offset < length
                else torch.zeros_like(value)
            )
            tapped = tapped + self.taps[tap] * shifted * (positions >= offset).to(value.dtype)[None, :, None]
        return key, tapped


def engram_indices(tokens, orders, heads, slots, seed_heads=0):
    u = tokens.long() & 0xFFFFFFFF
    stride = seed_heads or heads
    out = []
    for order_index, order in enumerate(orders):
        for head in range(heads):
            seed = (0x9E3779B9 * (order_index * stride + head + 1)) & 0xFFFFFFFF
            acc = torch.full_like(u, seed)
            for j in range(order):
                shifted = torch.cat([torch.zeros_like(u[:, :j]), u[:, :-j]], dim=1) if j else u
                acc = ((acc ^ shifted) * 0x01000193) & 0xFFFFFFFF
            acc = acc ^ (acc >> 15)
            out.append(acc % slots)
    return torch.stack(out, -1)
