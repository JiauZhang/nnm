import math

import torch
from torch import nn


def rotary_inv_freq(head_dim, base, rotary_dim=None):
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    assert 0 < rotary_dim <= head_dim and rotary_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    if rotary_dim < head_dim:
        inv_freq = torch.cat([inv_freq, torch.zeros(head_dim // 2 - rotary_dim // 2)])
    return inv_freq


def yarn_inv_freq(head_dim, base, factor, original_max_position, beta_fast=32, beta_slow=1):
    pos_freqs = base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)

    def correction_dim(rotations):
        return (head_dim * math.log(original_max_position / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), head_dim - 1)
    span = high - low if high != low else 0.001
    ramp = torch.clamp((torch.arange(head_dim // 2, dtype=torch.float32) - low) / span, 0, 1)
    return interpolation * ramp + extrapolation * (1 - ramp)


class RoPE(nn.Module):
    def __init__(self, inv_freq, scaling=1.0):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.scaling = scaling

    def forward(self, positions):
        angles = positions[..., None].float() * self.inv_freq
        embedding = torch.cat([angles, angles], dim=-1)
        return embedding.cos() * self.scaling, embedding.sin() * self.scaling


class RoPE2D(nn.Module):
    def __init__(self, inv_freqs):
        super().__init__()
        self.num_axes = len(inv_freqs)
        for index, inv_freq in enumerate(inv_freqs):
            self.register_buffer(f"inv_freq_{index}", inv_freq, persistent=False)

    def forward(self, positions):
        angles = []
        for index in range(self.num_axes):
            inv_freq = getattr(self, f"inv_freq_{index}")
            angles.append(positions[..., index].unsqueeze(-1).float() * inv_freq)
        angle = torch.cat(angles, dim=-1)
        embedding = torch.cat([angle, angle], dim=-1)
        return embedding.cos(), embedding.sin()


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(x, cos, sin, dim=None):
    if dim is not None:
        cos, sin = cos.unsqueeze(dim), sin.unsqueeze(dim)
    cos, sin = cos.float(), sin.float()
    return (x.float() * cos + rotate_half(x.float()) * sin).to(x.dtype)
