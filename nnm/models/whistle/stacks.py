import math

import torch
from torch import nn

from .layers import CrossAttention, Engram, HadamardMLP, MHCStack, SelfAttention, ZCRMSNorm, rms_unit


class ConvolutionBlock(nn.Module):
    def __init__(self, d_model, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pw1 = nn.Linear(d_model, 2 * d_model, bias=False)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=0, groups=d_model, bias=False)
        self.pw2 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        hidden, gate = self.pw1(x).chunk(2, -1)
        hidden = hidden * torch.sigmoid(gate)
        pad = self.kernel_size // 2
        return self.dw(torch.nn.functional.pad(hidden.transpose(1, 2), (pad, pad))).transpose(1, 2)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = ZCRMSNorm(config.d_model)
        self.attn = SelfAttention(
            config.d_model,
            config.num_heads,
            config.num_kv_heads,
            config.qk_head_dim,
            config.v_head_dim,
            config.encoder_layers,
            config.rope_theta,
            qkv_conv_taps=0,
            rope=True,
        )
        self.post_attn_norm = ZCRMSNorm(config.d_model)
        self.attn_gate = nn.Parameter(torch.zeros(()))
        self.conv_norm = ZCRMSNorm(config.d_model)
        self.conv = ConvolutionBlock(config.d_model, config.conv_kernel)
        self.conv_out_norm = ZCRMSNorm(config.d_model)
        self.pre_hada_norm = ZCRMSNorm(config.d_model)
        self.mlp = HadamardMLP(config.d_model, config.hada_n)
        self.pre_hada_norm_0 = ZCRMSNorm(config.d_model)
        self.mlp0 = HadamardMLP(config.d_model, config.hada_n)

    def forward(self, x):
        x = x + 0.5 * self.mlp0(self.pre_hada_norm_0(x))
        attn = self.post_attn_norm(self.attn(self.norm(x)))
        x = x + torch.sigmoid(self.attn_gate) * attn
        conv = torch.nn.functional.silu(self.conv_out_norm(self.conv(self.conv_norm(x))))
        x = x + self.conv.pw2(conv)
        return x + 0.5 * self.mlp(self.pre_hada_norm(x))


class DecoderBlock(nn.Module):
    def __init__(self, config, cross_attention):
        super().__init__()
        self.norm = ZCRMSNorm(config.d_model)
        self.self_attn = SelfAttention(
            config.d_model,
            config.num_heads,
            config.num_kv_heads,
            config.qk_head_dim,
            config.v_head_dim,
            config.num_layers,
            config.rope_theta,
            qkv_conv_taps=config.qkv_conv_taps,
            rope=True,
        )
        self.post_attn_norm = ZCRMSNorm(config.d_model)
        self.attn_gate = nn.Parameter(torch.zeros(()))
        self.cross_attention = cross_attention
        if cross_attention:
            self.cross_norm = ZCRMSNorm(config.d_model)
            self.cross = CrossAttention(config.d_model, config.num_heads, config.qk_head_dim, config.v_head_dim)
            self.post_cross_norm = ZCRMSNorm(config.d_model)
            self.cross_gate = nn.Parameter(torch.zeros(()))
        self.pre_hada_norm = ZCRMSNorm(config.d_model)
        self.mlp = HadamardMLP(config.d_model, config.hada_n)

    def forward(self, x, mask, memory=None):
        skip = x
        attn = self.post_attn_norm(self.self_attn(self.norm(x), mask=mask))
        x = skip + torch.sigmoid(self.attn_gate) * attn
        if self.cross_attention and memory is not None:
            cross = self.post_cross_norm(self.cross(self.cross_norm(x), memory))
            x = x + torch.sigmoid(self.cross_gate) * cross
        skip = x
        return skip + self.mlp(self.pre_hada_norm(x))


class EncoderStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.encoder_layers))
        self.mhc = MHCStack(config.d_model, config.encoder_layers, config.encoder_lanes)
        self.final_norm = ZCRMSNorm(config.d_model)
        self.pe_gate = nn.Parameter(torch.zeros(()))
        self.position_embedding = PositionEmbedding(config.d_model, config.rope_theta)

    def forward(self, x):
        x = x + torch.sigmoid(self.pe_gate) * self.position_embedding(x.shape[0], x.device, x.dtype)
        stream = x[None, :, None, :].expand(1, x.shape[0], self.config.encoder_lanes, x.shape[1]).contiguous()
        for index, block in enumerate(self.blocks):
            stream = self.mhc.advance(stream, block, self.mhc.layer_params(index))
        return self.final_norm(stream.mean(2))[0]


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, theta):
        super().__init__()
        self.d_model, self.theta = d_model, theta

    def forward(self, length, device, dtype):
        half = self.d_model // 2
        inv = 1.0 / (self.theta ** (torch.arange(half, dtype=torch.float32, device=device) / half))
        angle = torch.arange(length, dtype=torch.float32, device=device)[:, None] * inv[None]
        return torch.cat([angle.sin(), angle.cos()], -1).to(dtype)


class DecoderStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(DecoderBlock(config, True) for _ in range(config.num_layers))
        self.mhc = MHCStack(config.d_model, config.num_layers, config.mhc_lanes)
        self.final_norm = ZCRMSNorm(config.d_model)
        heads = max(1, config.d_model // (len(config.engram_orders) * config.engram_sub_dim))
        self.engrams = nn.ModuleList(
            Engram(
                config.d_model,
                config.engram_tables,
                config.engram_slots,
                config.engram_sub_dim,
                config.num_layers,
                config.engram_dilation,
                config.engram_orders,
                heads,
            )
            for _ in config.engram_layers
        )
        self.register_buffer("site_flags", _site_flags(config.num_layers, config.engram_layers), persistent=False)

    def _engram_kv(self, tokens):
        pairs = [engram(tokens) for engram in self.engrams]
        return torch.stack([key for key, _ in pairs]), torch.stack([value for _, value in pairs])

    def forward(self, tokens, memory=None):
        length = tokens.shape[1]
        encoding = self.embedding(tokens) * math.sqrt(self.config.d_model)
        mask = torch.ones(length, length, dtype=torch.bool, device=tokens.device).tril()
        engram_kv = self._engram_kv(tokens) if self.engrams else None
        stream = encoding[:, :, None, :].expand(-1, length, self.config.mhc_lanes, self.config.d_model).contiguous()
        for index, block in enumerate(self.blocks):

            def run(value, block=block, index=index):
                return block(_inject_engram(value, engram_kv, self.site_flags[index]), mask, memory)

            stream = self.mhc.advance(stream, run, self.mhc.layer_params(index))
        hidden = self.final_norm(stream.mean(2))
        return hidden @ self.embedding.weight.T


def _site_flags(num_layers, sites):
    flags = torch.zeros(num_layers, len(sites))
    for slot, layer in enumerate(sites):
        flags[layer, slot] = 1.0
    return flags


def _inject_engram(x, engram_kv, flags):
    if engram_kv is None or not flags.any():
        return x
    keys, values = engram_kv
    alpha = torch.sigmoid(torch.einsum("btd,sbtd->sbt", rms_unit(x), rms_unit(keys)) / math.sqrt(x.shape[-1]))
    return x + torch.einsum("s,sbt,sbtd->btd", flags, alpha, values)
