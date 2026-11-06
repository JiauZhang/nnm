import math

import torch
from torch import nn


def sinusoidal_embedding(positions, dim, base=10000.0, *, denominator=None, interleaved=False):
    half = dim // 2
    denominator = half if denominator is None else denominator
    inv_freq = torch.exp(-math.log(base) / denominator * torch.arange(half, dtype=torch.float32, device=positions.device))
    angles = positions.to(torch.float32)[:, None] * inv_freq[None]
    if interleaved:
        return torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(-2)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, length, dim, base=10000.0, *, denominator=None, interleaved=False):
        super().__init__()
        embedding = sinusoidal_embedding(torch.arange(length), dim, base, denominator=denominator, interleaved=interleaved)
        self.register_buffer("positional_embedding", embedding, persistent=False)
