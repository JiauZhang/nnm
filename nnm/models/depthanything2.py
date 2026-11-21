import torch.nn.functional as F

from nnm.models.dino import DINOv2
from nnm.models.dpt import DPTHead
from nnm.models.pretrained import PretrainedModel

TAP_INDICES = {
    "vits": (2, 5, 8, 11),
    "vitb": (2, 5, 8, 11),
    "vitl": (4, 11, 17, 23),
    "vitg": (9, 19, 29, 39),
}


class DepthAnythingV2(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.backbone = DINOv2(config)
        self.head = DPTHead(
            config.hidden_size,
            features=config.get("features", 256),
            out_channels=config.get("out_channels", [256, 512, 1024, 1024]),
            use_clstoken=config.get("use_clstoken", False),
            final_activation="sigmoid" if config.get("metric_max_depth", 0) else "relu",
        )
        self.tapped_blocks = tuple(config.get("tapped_blocks", TAP_INDICES[config.get("variant", "vits")]))
        self.patch_size = config.get("patch_size", 14)
        self.metric_max_depth = config.get("metric_max_depth", 0.0)

    def forward(self, pixel_values):
        height, width = pixel_values.shape[-2:]
        grid = (height // self.patch_size, width // self.patch_size)
        features = self.backbone.get_intermediate_layers(
            pixel_values, self.tapped_blocks, return_class_token=self.head.use_clstoken
        )
        depth = self.head(features, grid, (height, width))
        if self.metric_max_depth:
            return (depth * self.metric_max_depth).squeeze(1)
        return F.relu(depth).squeeze(1)
