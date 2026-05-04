import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import BidirectionalLSTM


class SharedINAdaINBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        padding=1,
        has_affine_norm=True,
        style_dim=128,
        leaky_relu_slope=0.2,
    ):
        super().__init__()
        self.has_affine_norm = has_affine_norm
        self.in_ch = in_ch

        if has_affine_norm:
            self.norm_weight = nn.Parameter(torch.ones(in_ch))
            self.norm_bias = nn.Parameter(torch.zeros(in_ch))

        self.norm_fc = nn.Linear(style_dim, in_ch * 2)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)

    def instance_norm(self, x):
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        if self.has_affine_norm:
            x_norm = x_norm * self.norm_weight.view(1, -1, 1) + self.norm_bias.view(1, -1, 1)
        return x_norm

    def forward(self, x, style, shared_in=None):
        if shared_in is not None:
            x_norm = shared_in
        else:
            x_norm = self.instance_norm(x)
        proj = self.norm_fc(style)
        gamma = proj[:, : self.in_ch].unsqueeze(2) + 1.0
        beta = proj[:, self.in_ch :].unsqueeze(2)
        x = x_norm * gamma + beta
        x = self.leaky_relu(x)
        x = self.conv(x)
        return x, x_norm


class PredictorBlock0(nn.Module):
    def __init__(
        self,
        channels=128,
        style_dim=128,
        has_affine_norm=True,
        leaky_relu_slope=0.2,
        residual_scale=math.sqrt(2),
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.norm1 = SharedINAdaINBlock(
            channels, channels, has_affine_norm=has_affine_norm, style_dim=style_dim, leaky_relu_slope=leaky_relu_slope
        )
        self.norm2 = SharedINAdaINBlock(
            channels, channels, has_affine_norm=False, style_dim=style_dim, leaky_relu_slope=leaky_relu_slope
        )

    def forward(self, x, style, shared_in=None):
        h, in_out = self.norm1(x, style, shared_in=shared_in)
        h, _ = self.norm2(h, style)
        return (x + h) / self.residual_scale, in_out


class PredictorBlock1(nn.Module):
    def __init__(
        self,
        in_channels=128,
        out_channels=64,
        style_dim=128,
        has_affine_norm1=True,
        has_affine_norm2=False,
        pool_kernel=3,
        pool_padding=1,
        pool_stride=2,
        pool_output_padding=1,
        block_kernel=3,
        block_padding=1,
        leaky_relu_slope=0.2,
        residual_scale=math.sqrt(2),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.pool_output_padding = pool_output_padding
        self.residual_scale = residual_scale

        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool_weight = nn.Parameter(torch.empty(in_channels, 1, pool_kernel))
        self.pool_bias = nn.Parameter(torch.empty(in_channels))
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=block_kernel, padding=block_padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=block_kernel, padding=block_padding)

        self.norm1 = SharedINAdaINBlock(
            in_channels, in_channels, has_affine_norm=has_affine_norm1, style_dim=style_dim, leaky_relu_slope=leaky_relu_slope
        )
        self.norm1.conv = nn.Identity()

        self.norm2 = SharedINAdaINBlock(
            out_channels, out_channels, has_affine_norm=has_affine_norm2, style_dim=style_dim, leaky_relu_slope=leaky_relu_slope
        )
        self.norm2.conv = nn.Identity()

    def forward(self, x, style, shared_in=None):
        batch, ch, t = x.shape
        t2 = t * 2

        x_up = F.interpolate(x, size=t2, mode='nearest')
        skip = self.conv1x1(x_up)

        h, _ = self.norm1(x, style, shared_in=None)
        h = F.conv_transpose1d(
            h, self.pool_weight, self.pool_bias,
            stride=self.pool_stride, padding=self.pool_padding,
            output_padding=self.pool_output_padding, groups=h.shape[1]
        )
        h = self.conv1(h)
        h, _ = self.norm2(h, style)
        h = self.conv2(h)

        return (skip + h) / self.residual_scale


class PredictorBlock2(nn.Module):
    def __init__(
        self,
        channels=64,
        style_dim=128,
        leaky_relu_slope=0.2,
        residual_scale=math.sqrt(2),
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.norm1 = SharedINAdaINBlock(
            channels, channels, has_affine_norm=False, style_dim=style_dim, leaky_relu_slope=leaky_relu_slope
        )
        self.norm2 = SharedINAdaINBlock(
            channels, channels, has_affine_norm=False, style_dim=style_dim, leaky_relu_slope=leaky_relu_slope
        )

    def forward(self, x, style):
        h, _ = self.norm1(x, style)
        h, _ = self.norm2(h, style)
        return (x + h) / self.residual_scale


class PredictorBranch(nn.Module):
    def __init__(
        self,
        in_channels=128,
        out_channels=64,
        style_dim=128,
        block0_has_affine=True,
        block1_norm2_has_affine=True,
        proj_kernel=1,
        residual_scale=math.sqrt(2),
        leaky_relu_slope=0.2,
    ):
        super().__init__()
        self.block0 = PredictorBlock0(channels=in_channels, style_dim=style_dim, has_affine_norm=block0_has_affine, leaky_relu_slope=leaky_relu_slope, residual_scale=residual_scale)
        self.block1 = PredictorBlock1(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            has_affine_norm1=block0_has_affine,
            has_affine_norm2=block1_norm2_has_affine,
            residual_scale=residual_scale,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.block2 = PredictorBlock2(channels=out_channels, style_dim=style_dim, leaky_relu_slope=leaky_relu_slope, residual_scale=residual_scale)
        self.proj = nn.Conv1d(out_channels, 1, kernel_size=proj_kernel)

    def forward(self, x, style, shared_in=None):
        x, in_out = self.block0(x, style, shared_in=shared_in)
        x = self.block1(x, style, shared_in=in_out)
        x = self.block2(x, style)
        return self.proj(x)


class LengthRegulator(nn.Module):
    def __init__(self, dur_min=1, dur_max=50):
        super().__init__()
        self.dur_min = dur_min
        self.dur_max = dur_max

    def forward(self, text_features, durations):
        if text_features.dim() == 3 and text_features.shape[0] != durations.shape[0]:
            text_features = text_features.transpose(0, 1)

        seq_len, batch_size, hidden_dim = text_features.shape

        if durations.dim() == 1:
            durations = durations.unsqueeze(1)

        durations = durations.clamp(min=self.dur_min, max=self.dur_max)

        max_len = int(durations.sum(dim=0).max().item())

        expanded_list = []
        for b in range(batch_size):
            repeated = torch.repeat_interleave(text_features[:, b], durations[:, b], dim=0)
            if repeated.shape[0] < max_len:
                pad = torch.zeros(max_len - repeated.shape[0], hidden_dim, device=repeated.device, dtype=repeated.dtype)
                repeated = torch.cat([repeated, pad], dim=0)
            expanded_list.append(repeated.unsqueeze(0))

        expanded = torch.cat(expanded_list, dim=0)
        return expanded.transpose(1, 2)


class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.predictor
        self.dur_lstm = BidirectionalLSTM(input_size=c.dur_lstm_input, hidden_size=c.dur_lstm_hidden)
        self.duration_proj = nn.Linear(c.dur_lstm_hidden * 2, c.dur_proj_out)

        self.length_regulator = LengthRegulator(dur_min=c.dur_min, dur_max=c.dur_max)

        self.shared_lstm = BidirectionalLSTM(input_size=c.shared_lstm_input, hidden_size=c.shared_lstm_hidden)

        self.f0_branch = PredictorBranch(
            in_channels=c.f0_branch_in,
            out_channels=c.f0_branch_out,
            style_dim=config.text_encoder.style_dim,
            block0_has_affine=True,
            block1_norm2_has_affine=True,
            proj_kernel=c.proj_kernel,
            residual_scale=c.residual_scale,
            leaky_relu_slope=c.leaky_relu_slope,
        )

        self.n_branch = PredictorBranch(
            in_channels=c.f0_branch_in,
            out_channels=c.f0_branch_out,
            style_dim=config.text_encoder.style_dim,
            block0_has_affine=True,
            block1_norm2_has_affine=False,
            proj_kernel=c.proj_kernel,
            residual_scale=c.residual_scale,
            leaky_relu_slope=c.leaky_relu_slope,
        )

    def forward(self, text_features, style, speed=1.0):
        if text_features.dim() == 3 and text_features.shape[0] == 1:
            batch_size = text_features.shape[1]
            seq_len = text_features.shape[0]
        elif text_features.dim() == 3 and text_features.shape[1] == 1:
            batch_size = text_features.shape[1]
            seq_len = text_features.shape[0]
        else:
            batch_size, seq_len, _ = text_features.shape
            text_features = text_features.transpose(0, 1)

        style_half = style[:, style.shape[-1] // 2 :]

        dur_hidden = self.dur_lstm(text_features)
        logits = self.duration_proj(dur_hidden)
        probs = torch.sigmoid(logits)
        dur_raw = probs.sum(dim=-1)
        dur_scaled = dur_raw / speed
        durations = torch.round(dur_scaled).clamp(min=self.length_regulator.dur_min, max=self.length_regulator.dur_max).long()

        expanded = self.length_regulator(text_features, durations)
        expanded_lstm = expanded.permute(2, 0, 1)
        shared_hidden = self.shared_lstm(expanded_lstm)
        shared_t = shared_hidden.permute(1, 2, 0)

        f0_b0_norm1 = self.f0_branch.block0.norm1
        shared_in = f0_b0_norm1.instance_norm(shared_t)

        f0 = self.f0_branch(shared_t, style_half, shared_in=shared_in)
        n_amp = self.n_branch(shared_t, style_half, shared_in=shared_in)

        if batch_size == 1:
            durations = durations.squeeze(1) if durations.dim() > 1 else durations

        return durations, expanded, shared_t, f0, n_amp
