import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "none":
            self.conv = nn.Identity()
        elif layer_type == "timepreserve":
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, (3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif layer_type == "half":
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, (3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise ValueError(f"unexpected downsample type {layer_type}")

    def forward(self, x):
        return self.conv(x)


class LearnedUpSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "none":
            self.conv = nn.Identity()
        elif layer_type == "timepreserve":
            self.conv = nn.ConvTranspose2d(
                dim_in, dim_in, (3, 1), stride=(2, 1), groups=dim_in, output_padding=(1, 0), padding=(1, 0)
            )
        elif layer_type == "half":
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, (3, 3), stride=(2, 2), groups=dim_in, output_padding=1, padding=1)
        else:
            raise ValueError(f"unexpected upsample type {layer_type}")

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        if self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        if self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        raise ValueError(f"unexpected downsample type {self.layer_type}")


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        if self.layer_type == "timepreserve":
            return F.interpolate(x, scale_factor=(2, 1), mode="nearest")
        if self.layer_type == "half":
            return F.interpolate(x, scale_factor=2, mode="nearest")
        raise ValueError(f"unexpected upsample type {self.layer_type}")


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, normalize=False, downsample="none"):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        return self.downsample(x)

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        return self.conv2(x)

    def forward(self, x):
        return (self._shortcut(x) + self._residual(x)) / math.sqrt(2)


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]
        for _ in range(4):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample="half"))
            dim_in = dim_out
        blocks += [
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0)),
            nn.AdaptiveAvgPool2d(1),
            nn.LeakyReLU(0.2),
        ]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        return self.unshared(h)
