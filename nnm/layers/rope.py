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


class RopePositionEmbedding(nn.Module):
    def __init__(self, dim, num_heads, base=100.0, min_period=None, max_period=None, normalize_coords="separate"):
        super().__init__()
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.normalize_coords = normalize_coords
        if base is not None:
            periods = base ** (2 * torch.arange(head_dim // 4, dtype=torch.float32) / (head_dim // 2))
        else:
            ratio = max_period / min_period
            exponents = torch.linspace(0, 1, head_dim // 4)
            periods = ratio**exponents / ratio * max_period
        self.register_buffer("periods", periods, persistent=True)

    def forward(self, height, width):
        if self.normalize_coords == "max":
            max_hw = max(height, width)
            coords_h = torch.arange(0.5, height, device=self.periods.device) / max_hw
            coords_w = torch.arange(0.5, width, device=self.periods.device) / max_hw
        elif self.normalize_coords == "min":
            min_hw = min(height, width)
            coords_h = torch.arange(0.5, height, device=self.periods.device) / min_hw
            coords_w = torch.arange(0.5, width, device=self.periods.device) / min_hw
        else:
            coords_h = torch.arange(0.5, height, device=self.periods.device) / height
            coords_w = torch.arange(0.5, width, device=self.periods.device) / width
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return angles.cos(), angles.sin()


class LearnableRoPE2D(nn.Module):
    def __init__(self, num_layers, num_heads, head_dim, num_dims=2):
        super().__init__()
        self.num_dims = num_dims
        self.freqs = nn.Parameter(torch.zeros(num_layers, num_dims, num_heads, head_dim // 2))

    def forward(self, height, width):
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=self.freqs.device),
                torch.arange(width, dtype=torch.float32, device=self.freqs.device),
                indexing="ij",
            ),
            dim=0,
        )
        phase = torch.einsum("d..., ldhf -> l...hf", coords, self.freqs)
        phase = phase.reshape(phase.shape[0], height * width, *phase.shape[3:])
        return phase.cos().repeat_interleave(2, dim=-1), phase.sin().repeat_interleave(2, dim=-1)


def axial_rope_cos_sin(head_dim, base, height, width, scale, device=None):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 4, dtype=torch.float32, device=device)[: head_dim // 4] / head_dim))
    indices = torch.arange(height * width, dtype=torch.float32, device=device)
    x_positions = (indices % height) * scale
    y_positions = torch.div(indices, height, rounding_mode="floor") * scale
    angles = torch.cat([torch.outer(x_positions, freqs), torch.outer(y_positions, freqs)], dim=-1)
    angles = angles.repeat_interleave(2, dim=-1)
    return angles.cos(), angles.sin()


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


def rotate_pairwise(x):
    paired = x.view(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def apply_rope(x, cos, sin, dim=None):
    if dim is not None:
        cos, sin = cos.unsqueeze(dim), sin.unsqueeze(dim)
    cos, sin = cos.float(), sin.float()
    return (x.float() * cos + rotate_half(x.float()) * sin).to(x.dtype)


def apply_rope_pairwise(x, cos, sin, dim=None):
    if dim is not None:
        cos, sin = cos.unsqueeze(dim), sin.unsqueeze(dim)
    cos, sin = cos.float(), sin.float()
    return (x.float() * cos + rotate_pairwise(x.float()) * sin).to(x.dtype)
