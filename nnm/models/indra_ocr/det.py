import torch
import torch.nn.functional as F
from torch import nn

from nnm.models.pretrained import PretrainedModel


class ResidualBlock(nn.Module):
    def __init__(self, channels, expand_channels, kernel_size):
        super().__init__()
        self.expand = nn.Conv2d(channels, expand_channels, 1)
        self.dw = nn.Conv2d(expand_channels, expand_channels, kernel_size, padding=kernel_size // 2, groups=expand_channels)
        self.project = nn.Conv2d(expand_channels, channels, 1)

    def forward(self, x):
        hidden = F.hardswish(self.expand(x))
        hidden = F.hardswish(self.dw(hidden))
        return x + self.project(hidden)


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, blocks):
        super().__init__()
        self.down = nn.Conv2d(in_channels, in_channels, kernel_size, stride=2, padding=kernel_size // 2, groups=in_channels)
        self.expand = nn.Conv2d(in_channels, out_channels, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels, out_channels * 2, kernel_size) for _ in range(blocks)])

    def forward(self, x):
        hidden = F.hardswish(self.down(x))
        hidden = F.hardswish(self.expand(hidden))
        return self.blocks(hidden)


class Lateral(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return F.hardswish(self.conv(x))


class IndraOCRDetection(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        channels = [config.stem_channels, *config.stage_channels]
        kernel_size = config.kernel_size

        self.stem = nn.Conv2d(1, config.stem_channels, kernel_size, stride=2, padding=kernel_size // 2)
        self.stages = nn.ModuleList(
            [
                Stage(channels[index], channels[index + 1], config.stage_kernel_size[index], config.blocks[index])
                for index in range(len(config.stage_channels))
            ]
        )
        self.lateral = nn.ModuleList([Lateral(channel, config.lateral_channels) for channel in channels[1:]])
        self.smooth = nn.Conv2d(
            config.lateral_channels,
            config.lateral_channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=config.lateral_channels,
        )
        self.stem_lateral = nn.Conv2d(config.stem_channels, config.stem_lateral_channels, 1)
        self.head = nn.ModuleList(
            [
                nn.Conv2d(config.lateral_channels + config.stem_lateral_channels * 4, config.head_channels, 1),
                nn.Conv2d(
                    config.head_channels,
                    config.head_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    groups=config.head_channels,
                ),
                nn.Conv2d(config.head_channels, config.output_channels * 4, 1),
            ]
        )

    def forward(self, x):
        stem = F.hardswish(self.stem(x))

        features = []
        hidden = stem
        for stage in self.stages:
            hidden = stage(hidden)
            features.append(hidden)

        fused = self.lateral[-1](features[-1])
        for index in range(len(features) - 2, -1, -1):
            fused = F.interpolate(fused, scale_factor=2, mode='nearest') + self.lateral[index](features[index])

        fused = F.hardswish(self.smooth(fused))
        stem_lateral = F.pixel_unshuffle(F.hardswish(self.stem_lateral(stem)), 2)

        hidden = F.hardswish(self.head[0](torch.cat([fused, stem_lateral], dim=1)))
        hidden = F.hardswish(self.head[1](hidden))
        hidden = F.pixel_shuffle(self.head[2](hidden), 2)

        return torch.cat([hidden[:, :1], 8.0 * F.softplus(hidden[:, 1:])], dim=1)
