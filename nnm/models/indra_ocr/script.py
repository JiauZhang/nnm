import torch
import torch.nn.functional as F
from torch import nn

from nnm.models.pretrained import PretrainedModel


class IndraOCRScript(PretrainedModel):
    module_name = "script"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.downsample = config.downsample
        self.pool = nn.AvgPool2d(2, 2, count_include_pad=True)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 3, padding=1)
                for in_channels, out_channels in zip([1, *config.channels[:-1]], config.channels)
            ]
        )
        self.pools = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in config.channels])
        self.head = nn.Linear(config.channels[-1], config.classes)

    def forward(self, x, widths):
        hidden = self.pool(x)
        for conv, pool in zip(self.convs, self.pools):
            hidden = pool(F.relu(conv(hidden)))
        hidden = hidden.squeeze(2)
        steps = (widths // self.downsample).clamp(min=1)
        mask = (torch.arange(hidden.shape[-1], device=x.device) < steps[:, None]).to(hidden.dtype).unsqueeze(1)
        hidden = (hidden * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return F.softmax(self.head(hidden), dim=-1)
