import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.residual = residual

    def forward(self, x):
        return x + self.residual(x)


class FeatureFusionBlock2d(nn.Module):
    def __init__(self, dim, with_skip):
        super().__init__()
        if with_skip:
            self.resnet1 = self._residual_block(dim)
        self.resnet2 = self._residual_block(dim)
        self.deconv = self.make_upsample(dim)
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def make_upsample(self, dim):
        return nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, bias=False)

    def upsample(self, x, size=None):
        return self.deconv(x)

    def forward(self, x, skip=None, size=None):
        if skip is not None:
            x = x + self.resnet1(skip)
        x = self.resnet2(x)
        x = self.upsample(x, size)
        return self.out_conv(x)

    @staticmethod
    def _residual_block(dim):
        return ResidualBlock(
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            )
        )


class BilinearFusionBlock(FeatureFusionBlock2d):
    def make_upsample(self, dim):
        return nn.Identity()

    def upsample(self, x, size=None):
        if size is None:
            return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class DPTDecoder(nn.Module):
    def __init__(self, dims_reassemble, dim_decoder, fusion_block=FeatureFusionBlock2d):
        super().__init__()
        self.dim_out = dim_decoder
        num_levels = len(dims_reassemble)
        self.convs = nn.ModuleList(
            [nn.Conv2d(dim, dim_decoder, kernel_size=3, padding=1, bias=False) for dim in dims_reassemble]
        )
        self.fusions = nn.ModuleList([fusion_block(dim_decoder, with_skip=i < num_levels - 1) for i in range(num_levels)])

    def forward(self, encodings):
        features = self.fusions[-1](self.convs[-1](encodings[-1]))
        for i in range(len(encodings) - 2, -1, -1):
            features = self.fusions[i](features, self.convs[i](encodings[i]))
        return features


class DPTHead(nn.Module):
    def __init__(
        self, in_channels, features=256, out_channels=(256, 512, 1024, 1024), use_clstoken=False, final_activation="relu"
    ):
        super().__init__()
        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList([nn.Conv2d(in_channels, dim, kernel_size=1) for dim in out_channels])
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        if use_clstoken:
            self.readout_projects = nn.ModuleList(
                [nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()) for _ in range(len(out_channels))]
            )
        self.layer_rn = nn.ModuleList([nn.Conv2d(dim, features, kernel_size=3, padding=1, bias=False) for dim in out_channels])
        self.refinenets = nn.ModuleList([BilinearFusionBlock(features, with_skip=True) for _ in out_channels])
        self.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() if final_activation == "sigmoid" else nn.ReLU(True),
        )

    def forward(self, features, grid, output_size):
        patch_h, patch_w = grid
        encodings = []
        for i, feature in enumerate(features):
            if self.use_clstoken:
                patches, cls_token = feature
                readout = cls_token.unsqueeze(1).expand_as(patches)
                patches = self.readout_projects[i](torch.cat((patches, readout), dim=-1))
            else:
                patches = feature
            x = patches.permute(0, 2, 1).reshape(patches.shape[0], patches.shape[-1], patch_h, patch_w)
            encodings.append(self.resize_layers[i](self.projects[i](x)))
        rn = [self.layer_rn[i](encoding) for i, encoding in enumerate(encodings)]
        path = self.refinenets[3](rn[3], size=rn[2].shape[2:])
        path = self.refinenets[2](path, rn[2], size=rn[1].shape[2:])
        path = self.refinenets[1](path, rn[1], size=rn[0].shape[2:])
        path = self.refinenets[0](path, rn[0])
        out = self.output_conv1(path)
        out = F.interpolate(out, output_size, mode="bilinear", align_corners=True)
        return self.output_conv2(out)
