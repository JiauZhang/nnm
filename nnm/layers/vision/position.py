import torch
from torch.nn import functional as F


def resize_abs_pos(abs_pos, size, mode="bicubic", antialias=True):
    dtype = abs_pos.dtype
    size = (size, size) if isinstance(size, int) else tuple(size)
    hidden = abs_pos.permute(0, 3, 1, 2).to(torch.float32)
    hidden = F.interpolate(hidden, size=size, mode=mode, antialias=antialias, align_corners=False).to(dtype)
    return hidden.permute(0, 2, 3, 1)


def merge_position_ids(grid_thw, merge_size):
    positions = []
    for frames, height, width in grid_thw.tolist():
        rows = torch.arange(height, device=grid_thw.device).view(height, 1).expand(height, width)
        cols = torch.arange(width, device=grid_thw.device).view(1, width).expand(height, width)
        block = (height // merge_size, merge_size, width // merge_size, merge_size)
        rows = rows.reshape(block).transpose(1, 2).flatten()
        cols = cols.reshape(block).transpose(1, 2).flatten()
        positions.append(torch.stack([rows, cols], dim=-1).repeat(frames, 1))
    return torch.cat(positions, dim=0)
