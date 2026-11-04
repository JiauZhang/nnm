import math

import torch
from torch import nn


def smart_resize(height, width, factor, min_pixels, max_pixels):
    if height < factor or width < factor:
        if height < width:
            width = round(factor / height * width)
            height = factor
        else:
            height = round(factor / width * height)
            width = factor
    elif max(height, width) / min(height, width) > 200:
        if height > width:
            height = 200 * width
        else:
            width = 200 * height

    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = math.floor(height / beta / factor) * factor
        resized_width = math.floor(width / beta / factor) * factor
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


class NaViTPreprocessor(nn.Module):
    def __init__(self, *, patch_size, hidden_stride, temporal_patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_stride = hidden_stride
        self.temporal_patch_size = temporal_patch_size

    @property
    def resize_factor(self):
        return self.patch_size * self.hidden_stride

    def resize(self, height, width, min_pixels, max_pixels):
        return smart_resize(height, width, self.resize_factor, min_pixels, max_pixels)

    def _patchify(self, frames):
        num_frames, num_channels, height, width = frames.shape
        if num_frames % self.temporal_patch_size != 0:
            repeats = frames[-1:].repeat(self.temporal_patch_size - 1, 1, 1, 1)
            frames = torch.cat([frames, repeats], dim=0)
            num_frames = frames.shape[0]

        if height % self.resize_factor != 0 or width % self.resize_factor != 0:
            raise ValueError(f"image size ({height}, {width}) must be divisible by patch_size * hidden_stride")

        grid_t = num_frames // self.temporal_patch_size
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        patches = frames.reshape(
            grid_t,
            self.temporal_patch_size,
            num_channels,
            grid_h // self.hidden_stride,
            self.hidden_stride,
            self.patch_size,
            grid_w // self.hidden_stride,
            self.hidden_stride,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        patch_dim = num_channels * self.temporal_patch_size * self.patch_size**2
        return patches.reshape(grid_t * grid_h * grid_w, patch_dim), (grid_t, grid_h, grid_w)

    def forward(self, images):
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images = list(images.unbind(0))
        patches, grids = [], []
        for image in images:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image_patches, grid = self._patchify(image)
            patches.append(image_patches)
            grids.append(grid)
        grid_thws = torch.tensor(grids, dtype=torch.long)
        return torch.cat(patches, dim=0), grid_thws
