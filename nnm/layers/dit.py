import math

import torch
from torch import nn

from nnm.layers.attention import SelfAttention
from nnm.layers.mlp import MLP


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half)
    arguments = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, timesteps):
        return self.mlp(timestep_embedding(timesteps, self.frequency_embedding_size).to(self.mlp[0].weight.dtype))


class LearnableTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, num_freqs):
        super().__init__()
        self.freqs = nn.Parameter(torch.zeros(num_freqs))
        self.proj = nn.Linear(2 * num_freqs, hidden_size, bias=False)

    def forward(self, timesteps):
        angles = timesteps.float()[:, None] * self.freqs
        embedding = torch.stack([angles.cos(), angles.sin()], dim=-1).flatten(-2)
        return self.proj(embedding.to(self.proj.weight.dtype))


class AdaLNModulation(nn.Module):
    def __init__(self, hidden_size, num_chunks):
        super().__init__()
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, num_chunks * hidden_size, bias=True))
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, condition):
        return self.proj(condition)


def modulate(hidden, shift, scale):
    return hidden * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, norm_eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.attn = SelfAttention(
            hidden_size, num_heads, head_dim=hidden_size // num_heads, qkv_bias=True, o_bias=True, causal=False
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size, "gelu", bias=True)
        self.adaLN_modulation = AdaLNModulation(hidden_size, 6)

    def forward(self, hidden, condition):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=1)
        hidden = hidden + gate_attn.unsqueeze(1) * self.attn(modulate(self.norm1(hidden), shift_attn, scale_attn))
        return hidden + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(hidden), shift_mlp, scale_mlp))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, norm_eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.adaLN_modulation = AdaLNModulation(hidden_size, 2)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, hidden, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(modulate(self.norm(hidden), shift, scale))
