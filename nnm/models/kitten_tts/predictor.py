import torch
import torch.nn.functional as F
from torch import nn

from .blocks import SharedStyleResidualBlock, StyleNormBlock, StyleResidualBlock


class PredictorUpsampleBlock(nn.Module):
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
        residual_scale=1.4142135623730951,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.ConvTranspose1d(
            in_channels,
            in_channels,
            kernel_size=pool_kernel,
            stride=pool_stride,
            padding=pool_padding,
            output_padding=pool_output_padding,
            groups=in_channels,
        )
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=block_kernel, padding=block_padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=block_kernel, padding=block_padding)
        self.norm1 = StyleNormBlock(
            in_channels,
            style_dim=style_dim,
            kernel_size=block_kernel,
            padding=block_padding,
            affine_norm=has_affine_norm1,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.norm2 = StyleNormBlock(
            out_channels,
            style_dim=style_dim,
            kernel_size=block_kernel,
            padding=block_padding,
            affine_norm=has_affine_norm2,
            leaky_relu_slope=leaky_relu_slope,
        )

    def forward(self, x, style):
        skip = self.conv1x1(F.interpolate(x, size=x.shape[-1] * 2, mode="nearest"))

        hidden = self.norm1(x, style)
        hidden = self.pool(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden, style)
        hidden = self.conv2(hidden)

        return (skip + hidden) / self.residual_scale


class PredictorBranch(nn.Module):
    def __init__(
        self,
        in_channels=128,
        out_channels=64,
        style_dim=128,
        block0_has_affine=True,
        block1_norm2_has_affine=True,
        proj_kernel=1,
        residual_scale=1.4142135623730951,
        leaky_relu_slope=0.2,
    ):
        super().__init__()
        self.block0 = SharedStyleResidualBlock(
            in_channels,
            style_dim=style_dim,
            affine1=block0_has_affine,
            leaky_relu_slope=leaky_relu_slope,
            residual_scale=residual_scale,
        )
        self.block1 = PredictorUpsampleBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            has_affine_norm1=block0_has_affine,
            has_affine_norm2=block1_norm2_has_affine,
            residual_scale=residual_scale,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.block2 = StyleResidualBlock(
            out_channels,
            style_dim=style_dim,
            affine1=False,
            affine2=False,
            leaky_relu_slope=leaky_relu_slope,
            residual_scale=residual_scale,
        )
        self.proj = nn.Conv1d(out_channels, 1, kernel_size=proj_kernel)

    def forward(self, x, style, normalized):
        x = self.block0(x, style, normalized)
        x = self.block1(x, style)
        x = self.block2(x, style)
        return self.proj(x)


class LengthRegulator(nn.Module):
    def __init__(self, dur_min=1, dur_max=50):
        super().__init__()
        self.dur_min = dur_min
        self.dur_max = dur_max

    def forward(self, text_features, durations):
        seq_len, batch_size, hidden_dim = text_features.shape
        durations = durations.clamp(min=self.dur_min, max=self.dur_max)
        max_len = int(durations.sum(dim=0).max().item())

        expanded_list = []
        for index in range(batch_size):
            repeated = torch.repeat_interleave(text_features[:, index], durations[:, index], dim=0)
            if repeated.shape[0] < max_len:
                pad = torch.zeros(max_len - repeated.shape[0], hidden_dim, device=repeated.device, dtype=repeated.dtype)
                repeated = torch.cat([repeated, pad], dim=0)
            expanded_list.append(repeated.unsqueeze(0))

        return torch.cat(expanded_list, dim=0).transpose(1, 2)


class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.predictor
        self.dur_lstm = nn.LSTM(c.dur_lstm_input, c.dur_lstm_hidden, bidirectional=True, batch_first=False)
        self.duration_proj = nn.Linear(c.dur_lstm_hidden * 2, c.dur_proj_out)
        self.length_regulator = LengthRegulator(dur_min=c.dur_min, dur_max=c.dur_max)
        self.shared_lstm = nn.LSTM(c.shared_lstm_input, c.shared_lstm_hidden, bidirectional=True, batch_first=False)
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
        seq_len, batch_size, _ = text_features.shape
        style_half = style[:, style.shape[-1] // 2 :]

        dur_hidden, _ = self.dur_lstm(text_features)
        logits = self.duration_proj(dur_hidden)
        probs = torch.sigmoid(logits)
        durations = torch.round(probs.sum(dim=-1) / speed).clamp(
            min=self.length_regulator.dur_min, max=self.length_regulator.dur_max
        )
        durations = durations.long()

        expanded = self.length_regulator(text_features, durations)
        shared_hidden, _ = self.shared_lstm(expanded.permute(2, 0, 1))
        shared_t = shared_hidden.permute(1, 2, 0)

        normalized = self._shared_normalization(shared_t)
        f0 = self.f0_branch(shared_t, style_half, normalized)
        n_amp = self.n_branch(shared_t, style_half, normalized)

        return durations, f0, n_amp

    def _shared_normalization(self, x):
        return self.f0_branch.block0.normalize(x)
