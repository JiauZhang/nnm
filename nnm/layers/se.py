import torch
from torch import nn

from nnm.layers.activation import get_activation


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4, activation="relu"):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0)
        self.act_fn = get_activation(activation) if activation is not None else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.avg_pool(x)
        x = self.conv2(self.act_fn(self.conv1(x)))
        x = torch.clip(0.2 * x + 0.5, min=0.0, max=1.0)
        return residual * x


class SqueezeExcitationLayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.convs = nn.ModuleList()
        for in_channels, out_channels, activation in [
            [channel, channel // reduction, nn.ReLU()],
            [channel // reduction, channel, nn.Hardsigmoid()],
        ]:
            self.convs.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )
            self.convs.append(activation)

    def forward(self, x):
        residual = x
        x = self.avg_pool(x)
        for layer in self.convs:
            x = layer(x)
        return residual * x
