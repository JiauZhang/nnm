import re, torch
import torch.nn as nn
import torch.nn.functional as F
from nnm.layers.activation import get_activation
from nnm.backbone.pp_lcnet_v3 import LCNetV3Encoder
from conippets.config import Config


class SqueezeExcitationModule(nn.Module):
    def __init__(self, in_channels, reduction, activation="relu"):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels // reduction, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )
        self.act_fn = get_activation(activation) if activation is not None else nn.Identity()

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.avg_pool(hidden_states)
        hidden_states = self.conv2(self.act_fn(self.conv1(hidden_states)))
        hidden_states = torch.clip(0.2 * hidden_states + 0.5, min=0.0, max=1.0)
        return residual * hidden_states


class ResidualSqueezeExcitationLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, reduction):
        super().__init__()
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            bias=False,
        )
        self.squeeze_excitation_block = SqueezeExcitationModule(out_channels, reduction)

    def forward(self, hidden_states):
        hidden_states = self.in_conv(hidden_states)
        hidden_states = hidden_states + self.squeeze_excitation_block(hidden_states)

        return hidden_states


class Neck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.interpolate_mode = config.interpolate_mode
        self.insert_conv = nn.ModuleList()
        self.input_conv = nn.ModuleList()
        for i in range(len(config.layer_list_out_channels)):
            self.insert_conv.append(
                ResidualSqueezeExcitationLayer(
                    config.layer_list_out_channels[i], config.neck_out_channels, 1, config.backbone_config.reduction
                )
            )
            self.input_conv.append(
                ResidualSqueezeExcitationLayer(
                    config.neck_out_channels, config.neck_out_channels // 4, 3, config.backbone_config.reduction
                )
            )

    def forward(self, feature_maps):
        fused = []
        for conv, feature in zip(self.insert_conv, feature_maps):
            hidden_states = conv(feature)
            fused.append(hidden_states)

        for i in range(2, -1, -1):
            fused[i] = fused[i] + F.interpolate(fused[i + 1], scale_factor=2, mode=self.interpolate_mode)

        features = []
        for conv, feat in zip(self.input_conv, [fused[0], fused[1], fused[2], fused[3]]):
            features.append(conv(feat))

        processed = []
        upsample_scales = [1, 2, 4, 8]
        for feat, scale in zip(features, upsample_scales):
            if scale != 1:
                hidden_states = F.interpolate(feat, scale_factor=scale, mode=self.interpolate_mode)
            else:
                hidden_states = feat
            processed.append(hidden_states)

        fused_feature_map = torch.cat(processed[::-1], dim=1)
        return fused_feature_map


class ConvBatchnormLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        groups=1,
        activation="relu",
        bias=False,
        conv_transpose=False,
    ):
        super().__init__()
        if conv_transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            )

        self.norm = nn.BatchNorm2d(out_channels)
        self.act_fn = get_activation(activation) if activation is not None else nn.Identity()

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config.neck_out_channels
        kernel_list = config.kernel_list
        self.conv_down = ConvBatchnormLayer(
            in_channels=in_channels, out_channels=in_channels // 4, kernel_size=kernel_list[0], padding=int(kernel_list[0] // 2)
        )
        self.conv_up = ConvBatchnormLayer(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            conv_transpose=True,
        )

        self.conv_final = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=1, kernel_size=kernel_list[2], stride=2)

    def forward(self, hidden_states):
        hidden_states = self.conv_down(hidden_states)
        hidden_states = self.conv_up(hidden_states)
        hidden_states = self.conv_final(hidden_states)
        hidden_states = torch.sigmoid(hidden_states)
        return hidden_states


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = LCNetV3Encoder(config.backbone_config)
        out_channels = [self.backbone.num_features[i] for i in self.backbone.out_indices]
        self.layer = nn.ModuleList()
        for idx, out_channel in enumerate(out_channels):
            self.layer.append(nn.Conv2d(out_channel, config.layer_list_out_channels[idx], 1, 1, 0))

        self.neck = Neck(config)

    def forward(self, hidden_states):
        feature_maps = self.backbone(hidden_states)
        processed_features = []
        for i in range(len(feature_maps)):
            processed_features.append(self.layer[i](feature_maps[i]))
        hidden_states = self.neck(processed_features)

        return hidden_states


class PPOCRv5MobileDetectionModel(nn.Module):
    hf_config = Config(
        backbone_config=dict(
            stem_channels=16,
            stem_stride=2,
            scale=0.75,
            block_configs=[
                [[3, 16, 32, 1, False]],
                [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
                [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
                [
                    [3, 128, 256, 2, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                ],
                [[5, 256, 512, 2, True], [5, 512, 512, 1, True], [5, 512, 512, 1, False], [5, 512, 512, 1, False]],
            ],
            reduction=4,
            divisor=16,
            conv_symmetric_num=4,
            hidden_act="hardswish",
            out_indices=[2, 3, 4, 5],
        ),
        layer_list_out_channels=[12, 18, 42, 360],
        neck_out_channels=96,
        interpolate_mode="nearest",
        kernel_list=[3, 2, 2],
    )

    def __init__(self, config=hf_config):
        super().__init__()
        self.model = Model(config)
        self.head = Head(config)

    def forward(self, pixel_values):
        hidden_states = self.model(pixel_values)
        logits = self.head(hidden_states)

        return logits

    def load_hf_state_dict(self, hf_state_dict):
        REPLACEMENTS = [
            (r"^model\.backbone\.encoder\.", "model.backbone."),
            (r"\.convolution\.convolution(\.|$)", r".conv.conv\1"),
            (r"\.depthwise_convolution\.", ".dw_conv."),
            (r"\.pointwise_convolution\.", ".pw_conv."),
            (r"\.convolution(\.|$)", r".conv\1"),
            (r"\.conv_small_symmetric(\.|$)", ".convs.4."),
            (r"\.conv_symmetric\.(\d+)(\.|$)", r".convs.\1."),
            (r"\.act\.lab\.", ".act_"),
            (r"\.lab\.", "."),
            (r"\.squeeze_excitation_module\.", ".se."),
            (r"\.se\.convolutions\.", ".se.convs."),
            (r"\.identity\.", ".bn."),
        ]
        converted = {}
        for key, value in hf_state_dict.items():
            new_key = key
            for pattern, repl in REPLACEMENTS:
                new_key = re.sub(pattern, repl, new_key)
            converted[new_key] = value
        self.load_state_dict(converted)
