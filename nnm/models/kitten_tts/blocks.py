import math

import torch
import torch.nn.functional as F
from torch import nn


class ConvLayerNormAct1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, leaky_relu_slope=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(out_channels, eps=1e-5)
        self.act = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x):
        hidden = self.norm(self.conv(x).transpose(1, 2))
        return self.act(hidden).transpose(1, 2)


class AdaIN1d(nn.Module):
    def __init__(self, channels, style_dim, affine=True):
        super().__init__()
        self.channels = channels
        self.norm = nn.InstanceNorm1d(channels, affine=affine)
        self.fc = nn.Linear(style_dim, 2 * channels)

    def normalize(self, x):
        return self.norm(x)

    def apply_style(self, x_norm, style):
        if style.dim() == 1:
            style = style.unsqueeze(0)
        params = self.fc(style)
        scale = params[:, : self.channels].unsqueeze(2)
        shift = params[:, self.channels :].unsqueeze(2)
        return (scale + 1.0) * x_norm + shift

    def forward(self, x, style, normalized=None):
        x_norm = self.normalize(x) if normalized is None else normalized
        return self.apply_style(x_norm, style)


class StyleNormBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        *,
        style_dim=128,
        kernel_size=3,
        padding=None,
        affine_norm=True,
        leaky_relu_slope=0.2,
    ):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.adain = AdaIN1d(in_channels, style_dim, affine=affine_norm)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding) if out_channels is not None else None
        self.act = nn.LeakyReLU(leaky_relu_slope)

    def normalize(self, x):
        return self.adain.normalize(x)

    def forward(self, x, style, normalized=None):
        hidden = self.act(self.adain(x, style, normalized))
        return self.conv(hidden) if self.conv is not None else hidden


class StyleResidualBlock(nn.Module):
    def __init__(
        self, channels, *, style_dim=128, affine1=True, affine2=False, leaky_relu_slope=0.2, residual_scale=math.sqrt(2)
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.norm1 = StyleNormBlock(
            channels,
            channels,
            style_dim=style_dim,
            affine_norm=affine1,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.norm2 = StyleNormBlock(
            channels,
            channels,
            style_dim=style_dim,
            affine_norm=affine2,
            leaky_relu_slope=leaky_relu_slope,
        )

    def normalize(self, x):
        return self.norm1.normalize(x)

    def forward(self, x, style):
        hidden = self.norm1(x, style)
        hidden = self.norm2(hidden, style)
        return (x + hidden) / self.residual_scale


class SharedStyleResidualBlock(StyleResidualBlock):
    def forward(self, x, style, normalized):
        hidden = self.norm1(x, style, normalized)
        hidden = self.norm2(hidden, style)
        return (x + hidden) / self.residual_scale


class StyleLayerNorm(nn.Module):
    def __init__(self, feat_dim, style_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.fc = nn.Linear(style_dim, 2 * feat_dim)

    def forward(self, x, style):
        x_norm = F.layer_norm(x, [self.feat_dim])
        params = self.fc(style)
        scale = params[:, : self.feat_dim] + 1.0
        bias = params[:, self.feat_dim :]
        return x_norm * scale.unsqueeze(1) + bias.unsqueeze(1)


class StyleLSTMStage(nn.Module):
    def __init__(self, input_size, hidden_size, style_dim, leaky_relu_slope=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.style_norm = StyleLayerNorm(2 * hidden_size, style_dim)

    def forward(self, x, style):
        style_broadcast = style.unsqueeze(1).expand(-1, x.shape[1], -1)
        hidden, _ = self.lstm(torch.cat([x, style_broadcast], dim=-1))
        return self.style_norm(hidden, style)
