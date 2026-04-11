import torch
from torch import nn
from nnm.layers.activation import get_activation

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation="hardswish", groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False, groups=groups)
        self.normalization = nn.BatchNorm2d(out_channels)
        self.act_fn = get_activation(activation) if activation is not None else nn.Identity()

    def forward(self, input):
        hidden_state = self.conv(input)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.act_fn(hidden_state)
        return hidden_state


class LearnableRepLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, stride, num_branches, groups=1):
        super().__init__()
        self.stride = stride
        self.bn = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.convs = nn.ModuleList(
            [ConvLayer(in_channels, out_channels, kernel_size, stride, groups=groups, activation=None) for _ in range(num_branches)]
            + ([ConvLayer(in_channels, out_channels, 1, stride, groups=groups, activation=None)] if kernel_size > 1 else [])
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.act_scale = nn.Parameter(torch.tensor(1.0))
        self.act_bias = nn.Parameter(torch.tensor(0.0))
        self.act = get_activation(activation) if activation is not None else nn.Identity()

    def forward(self, hidden_state):
        output = None
        if self.bn is not None:
            output = self.bn(hidden_state)
        for conv in self.convs:
            residual = conv(hidden_state)
            output = residual if output is None else output + residual
        hidden_state = self.scale * output + self.bias
        if self.stride != 2:
            hidden_state = self.act_scale * self.act(hidden_state) + self.act_bias
        return hidden_state


class SqueezeExcitationLayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.convs = nn.ModuleList()
        for in_channels, out_channels, activation in [
            [channel, channel // reduction, nn.ReLU()],
            [channel // reduction, channel, nn.Hardsigmoid()],
        ]:
            self.convs.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True))
            self.convs.append(activation)

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.avg_pool(hidden_state)
        for layer in self.convs:
            hidden_state = layer(hidden_state)
        hidden_state = residual * hidden_state

        return hidden_state


class DepthwiseSeparableConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, use_squeeze_excitation, config):
        super().__init__()
        self.dw_conv = LearnableRepLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels, num_branches=config.conv_symmetric_num, activation=config.hidden_act)
        self.se = (
            SqueezeExcitationLayer(in_channels, config.reduction) if use_squeeze_excitation else nn.Identity()
        )
        self.pw_conv = LearnableRepLayer(in_channels=in_channels, kernel_size=1, out_channels=out_channels, stride=1, num_branches=config.conv_symmetric_num, activation=config.hidden_act)

    def forward(self, hidden_state):
        hidden_state = self.dw_conv(hidden_state)
        hidden_state = self.se(hidden_state)
        hidden_state = self.pw_conv(hidden_state)

        return hidden_state


def make_divisible(value, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class Block(nn.Module):
    def __init__(self, config, stage_index):
        super().__init__()
        self.config = config
        blocks = config.block_configs[stage_index]
        self.layers = nn.ModuleList()
        for kernel_size, in_channels, out_channels, stride, use_squeeze_excitation in blocks:
            scaled_in_channels = make_divisible(in_channels * config.scale, config.divisor)
            scaled_out_channels = make_divisible(out_channels * config.scale, config.divisor)

            dw_block = DepthwiseSeparableConvLayer(in_channels=scaled_in_channels, out_channels=scaled_out_channels, kernel_size=kernel_size, stride=stride, use_squeeze_excitation=use_squeeze_excitation, config=config)
            self.layers.append(dw_block)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class LCNetV3Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_features = [make_divisible(config.stem_channels * config.scale, config.divisor)]
        for block_config in config.block_configs:
            out_channels = block_config[-1][2]
            num_features.append(make_divisible(out_channels * config.scale, config.divisor))
        self.num_features = num_features
        self.out_indices = config.out_indices if hasattr(config, 'out_indices') and config.out_indices else list(range(len(config.block_configs) + 1))
        self.conv = ConvLayer(in_channels=3, kernel_size=3, out_channels=make_divisible(config.stem_channels * config.scale, config.divisor), stride=config.stem_stride, activation=None)
        self.blocks = nn.ModuleList([])
        for stage_index in range(len(config.block_configs)):
            block = Block(config, stage_index)
            self.blocks.append(block)

    def forward(self, pixel_values):
        hidden_state = self.conv(pixel_values)
        hidden_states = (hidden_state,)
        for block in self.blocks:
            hidden_state = block(hidden_state)
            hidden_states += (hidden_state,)

        return tuple(hidden_states[i] for i in self.out_indices)
