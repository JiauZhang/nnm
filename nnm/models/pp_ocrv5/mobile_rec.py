from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnm.layers.attention import SelfAttention
from nnm.layers.conv import ConvNormAct
from nnm.layers.mlp import MLP
from nnm.layers.transformer import TransformerBlock

from .pp_lcnet_v3 import LCNetV3Encoder, make_divisible
from conippets.config import Config


class Block(TransformerBlock):
    def __init__(self, config):
        hidden_size = config.hidden_size
        super().__init__(
            hidden_size,
            SelfAttention(
                hidden_size,
                config.num_attention_heads,
                head_dim=hidden_size // config.num_attention_heads,
                qkv_bias=(config.qkv_bias, False, config.qkv_bias),
                o_bias=True,
                causal=False,
            ),
            MLP(hidden_size, int(hidden_size * config.mlp_ratio), activation=config.hidden_act),
            norm_layer=partial(nn.LayerNorm, eps=config.layer_norm_eps),
        )


class EncoderWithSVTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = make_divisible(
            config.backbone_config.block_configs[-1][-1][2] * config.backbone_config.scale, config.backbone_config.divisor
        )
        hidden_size = config.hidden_size
        conv_configs = [
            (in_channels, in_channels // 8, config.conv_kernel_size),
            (in_channels // 8, hidden_size, (1, 1)),
            (hidden_size, in_channels, (1, 1)),
            (2 * in_channels, in_channels // 8, config.conv_kernel_size),
            (in_channels // 8, hidden_size, (1, 1)),
        ]
        self.convs = nn.ModuleList(
            [
                ConvNormAct(in_ch, out_ch, kernel, padding=(kernel[0] // 2, kernel[1] // 2), activation="silu")
                for in_ch, out_ch, kernel in conv_configs
            ]
        )
        self.svtr_block = nn.ModuleList()
        for _ in range(config.depth):
            self.svtr_block.append(Block(config=config))
        self.norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states

        hidden_states = self.convs[0](hidden_states)
        hidden_states = self.convs[1](hidden_states)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        for block in self.svtr_block:
            hidden_states = block(hidden_states)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        hidden_states = self.convs[2](hidden_states)
        hidden_states = self.convs[3](torch.cat((residual, hidden_states), dim=1))
        hidden_states = self.convs[4](hidden_states)
        hidden_states = hidden_states.squeeze(2).transpose(1, 2)

        return hidden_states


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = LCNetV3Encoder(config.backbone_config)

    def forward(self, pixel_values):
        feature_maps = self.backbone(pixel_values)
        hidden_state = feature_maps[-1]
        hidden_state = F.avg_pool2d(hidden_state, (3, 2))
        return hidden_state


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = EncoderWithSVTR(config)
        self.head = nn.Linear(config.hidden_size, config.head_out_channels)

    def forward(self, hidden_states):
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.head(hidden_states)
        hidden_states = F.softmax(hidden_states, dim=2, dtype=torch.float32).to(hidden_states.dtype)
        return hidden_states


class PPOCRv5MobileRecognitionModel(nn.Module):
    hf_config = Config(
        backbone_config=dict(
            stem_channels=16,
            stem_stride=2,
            scale=0.95,
            out_features=["stage2", "stage3", "stage4", "stage5"],
            out_indices=[2, 3, 4, 5],
            divisor=16,
            block_configs=[
                [[3, 16, 32, 1, False]],
                [[3, 32, 64, 1, False], [3, 64, 64, 1, False]],
                [[3, 64, 128, [2, 1], False], [3, 128, 128, 1, False]],
                [
                    [3, 128, 256, [1, 2], False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                ],
                [[5, 256, 512, [2, 1], True], [5, 512, 512, 1, True], [5, 512, 512, [2, 1], False], [5, 512, 512, 1, False]],
            ],
            conv_symmetric_num=4,
            reduction=4,
            hidden_act="hardswish",
        ),
        hidden_act="silu",
        hidden_size=120,
        mlp_ratio=2.0,
        depth=2,
        head_out_channels=18385,
        conv_kernel_size=[1, 3],
        qkv_bias=True,
        num_attention_heads=8,
        attention_dropout=0.0,
        layer_norm_eps=1e-6,
    )

    def __init__(self, config=hf_config):
        super().__init__()
        self.model = Model(config)
        self.head = Head(config)

    def forward(self, pixel_values):
        hidden_states = self.model(pixel_values)
        logits = self.head(hidden_states)
        return logits
