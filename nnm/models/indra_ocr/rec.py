import torch
import torch.nn.functional as F
from torch import nn

from nnm.models.pretrained import PretrainedModel


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduced_channels):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, reduced_channels, 1)
        self.fc2 = nn.Conv1d(reduced_channels, channels, 1)

    def forward(self, x, mask, count):
        pooled = (x * mask).sum(-1, keepdim=True) / count
        return torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))


class ConvBlock(nn.Module):
    def __init__(self, channels, expand_channels, reduced_channels, kernel_size):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.expand = nn.Conv1d(channels, expand_channels, 1)
        self.se = SqueezeExcitation(expand_channels, reduced_channels)
        self.project = nn.Conv1d(expand_channels, channels, 1)

    def forward(self, x, mask, count):
        hidden = F.hardswish(self.expand(self.dw(x)))
        hidden = hidden * self.se(hidden, mask, count)
        return (x + self.project(hidden)) * mask


class RowContext(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.groups = config.context_groups
        self.patch = nn.Conv2d(1, config.patch_channels, config.patch_size, stride=config.patch_size)
        self.dw = nn.Conv2d(
            config.patch_channels, config.patch_channels, 3, padding=1, groups=config.patch_channels, bias=False
        )
        self.pw = nn.Conv2d(config.patch_channels, config.context_channels, 1)
        self.score = nn.Conv2d(config.context_channels, config.context_groups, 1)
        self.row_prior = nn.Parameter(torch.zeros(1, config.context_groups, config.image_height // config.patch_size, 1))
        self.proj = nn.Conv1d(config.context_channels, config.channels, 1)

    def forward(self, x):
        hidden = F.hardswish(self.patch(x))
        hidden = F.hardswish(self.pw(self.dw(hidden)))
        weight = torch.softmax(self.score(hidden) + self.row_prior, dim=2)
        batch, channels, rows, steps = hidden.shape
        pooled = hidden.reshape(batch, self.groups, channels // self.groups, rows, steps)
        pooled = (pooled * weight[:, :, None]).sum(3)
        return self.proj(pooled.reshape(batch, channels, steps))


class IndraOCRRecognition(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.band = (config.image_height - config.band_height) // 2

        timestep = config.timestep
        self.stem = nn.Conv2d(1, config.channels, (config.band_height, timestep), stride=(config.band_height, timestep))
        self.stem_bn = nn.BatchNorm1d(config.channels)
        self.context = RowContext(config)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(config.channels, config.expand_channels, config.reduced_channels, config.kernel_size)
                for _ in range(config.num_blocks)
            ]
        )
        self.head = nn.Conv1d(config.channels, config.vocab_size, 1)

    def forward(self, x, widths):
        hidden = self.stem(x[:, :, self.band : self.band + self.config.band_height]).squeeze(2)
        hidden = F.hardswish(self.stem_bn(hidden + self.context(x)))

        lengths = self.lengths(widths, hidden.shape[-1])
        mask = (torch.arange(hidden.shape[-1], device=x.device) < lengths[:, None]).to(hidden.dtype).unsqueeze(1)
        count = mask.sum(-1, keepdim=True).clamp(min=1)

        hidden = hidden * mask
        for block in self.blocks:
            hidden = block(hidden, mask, count)
        return self.head(hidden).transpose(1, 2)

    def lengths(self, widths, steps):
        timestep = self.config.timestep
        return ((widths + timestep - 1) // timestep).clamp(max=steps)

    def recognize(self, x, widths):
        logits = self.forward(x, widths)
        log_probs = torch.log_softmax(logits, dim=-1)
        values, indices = log_probs.topk(self.config.topk, dim=-1)
        return {
            'ids': logits.argmax(dim=-1),
            'conf': torch.softmax(logits, dim=-1).max(dim=-1).values,
            'lengths': self.lengths(widths, logits.shape[1]),
            'blank': log_probs[..., 0],
            'topv': values,
            'topi': indices,
        }
