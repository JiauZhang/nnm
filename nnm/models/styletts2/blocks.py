import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear_layer(x)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
        return x.transpose(1, -1)


class AdaIN1d(nn.Module):
    def __init__(self, channels, style_dim, affine=False):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=affine)
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, style):
        h = self.fc(style).view(style.size(0), -1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, style):
        h = self.fc(style).view(style.size(0), -1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        return (1 + gamma) * x + beta


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        return F.interpolate(x, scale_factor=2, mode="nearest")


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, upsample="none", dropout_p=0.0, affine=False):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(dim_in, style_dim, affine=affine)
        self.norm2 = AdaIN1d(dim_out, style_dim, affine=affine)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))
        self.dropout = nn.Dropout(dropout_p)
        if upsample == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1)
            )

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, style):
        x = self.actv(self.norm1(x, style))
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.actv(self.norm2(x, style))
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, style):
        out = self._residual(x, style)
        return (out + self._shortcut(x)) / math.sqrt(2)
