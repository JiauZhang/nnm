import torch.nn.functional as F
from torch import nn

from nnm.models.indra_ocr.det import Stage
from nnm.models.pretrained import PretrainedModel


class IndraOCROrientation(PretrainedModel):
    module_name = "orient"

    def __init__(self, config):
        super().__init__()
        self.config = config
        channels = [config.stem_channels, *config.stage_channels]

        self.stem = nn.Conv2d(1, config.stem_channels, config.kernel_size, stride=2, padding=config.kernel_size // 2)
        self.stages = nn.ModuleList(
            [
                Stage(channels[index], channels[index + 1], config.stage_kernel_size[index], config.blocks[index])
                for index in range(len(config.stage_channels))
            ]
        )
        self.head = nn.Linear(config.stage_channels[-1], config.num_classes)

    def forward(self, x):
        hidden = F.hardswish(self.stem(x))
        for stage in self.stages:
            hidden = stage(hidden)
        hidden = hidden.mean((2, 3))
        return F.softmax(self.head(hidden), dim=-1)
