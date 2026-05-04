import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaINBlock(nn.Module):
    def __init__(self, channels, style_dim=128):
        super().__init__()
        self.channels = channels
        self.fc = nn.Linear(style_dim, channels * 2)
        self.norm = nn.InstanceNorm1d(channels, affine=True)

    def forward(self, x, style):
        if style.dim() == 1:
            style = style.unsqueeze(0)
        params = self.fc(style)
        scale = params[:, : self.channels].unsqueeze(2)
        shift = params[:, self.channels :].unsqueeze(2)
        x_norm = self.norm(x)
        return (scale + 1.0) * x_norm + shift


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        style_dim=128,
        kernel_size=3,
        leaky_relu_slope=0.2,
    ):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.adain1 = AdaINBlock(in_channels, style_dim=style_dim)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.adain2 = AdaINBlock(out_channels, style_dim=style_dim)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.residual_scale = 1 / math.sqrt(2)
        self.leaky_relu_slope = leaky_relu_slope

    def forward(self, x, style):
        if style.dim() == 1:
            style = style.unsqueeze(0)
        x_res = self.conv1x1(x)
        x = F.leaky_relu(self.adain1(x, style), self.leaky_relu_slope)
        x = self.conv1(x)
        x = F.leaky_relu(self.adain2(x, style), self.leaky_relu_slope)
        x = self.conv2(x)
        x = (x + x_res) * self.residual_scale
        return x


class DecoderUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        style_dim=128,
        kernel_size=3,
        leaky_relu_slope=0.2,
    ):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.adain1 = AdaINBlock(in_channels, style_dim=style_dim)
        self.pool = nn.ConvTranspose1d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=in_channels,
        )
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.adain2 = AdaINBlock(out_channels, style_dim=style_dim)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.residual_scale = 1 / math.sqrt(2)
        self.leaky_relu_slope = leaky_relu_slope

    def forward(self, x, style):
        if style.dim() == 1:
            style = style.unsqueeze(0)

        x_up = F.interpolate(x, scale_factor=2, mode='nearest')
        x_res = self.conv1x1(x_up)

        x = F.leaky_relu(self.adain1(x, style), self.leaky_relu_slope)
        x = self.pool(x)
        x = self.conv1(x)
        x = F.leaky_relu(self.adain2(x, style), self.leaky_relu_slope)
        x = self.conv2(x)
        x = (x + x_res) * self.residual_scale

        return x


class KittenDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.decoder
        self._style_dim = c.style_dim
        self.encode = DecoderBlock(c.encode_in_channels, c.encode_out_channels, style_dim=c.style_dim, kernel_size=c.block_kernel, leaky_relu_slope=c.leaky_relu_slope)

        self.decode0 = DecoderBlock(c.decode_in_channels, c.decode_out_channels, style_dim=c.style_dim, kernel_size=c.block_kernel, leaky_relu_slope=c.leaky_relu_slope)
        self.decode1 = DecoderBlock(c.decode_in_channels, c.decode_out_channels, style_dim=c.style_dim, kernel_size=c.block_kernel, leaky_relu_slope=c.leaky_relu_slope)
        self.decode2 = DecoderBlock(c.decode_in_channels, c.decode_out_channels, style_dim=c.style_dim, kernel_size=c.block_kernel, leaky_relu_slope=c.leaky_relu_slope)
        self.decode3 = DecoderUpsampleBlock(c.decode_in_channels, c.decode_out_channels, style_dim=c.style_dim, kernel_size=c.block_kernel, leaky_relu_slope=c.leaky_relu_slope)

        self.asr_res = nn.Conv1d(c.asr_res_in, c.asr_res_out, 1)

        self.F0_conv = nn.Conv1d(c.f0_downsample_channels, c.f0_downsample_channels, c.downsample_kernel, stride=c.downsample_stride, padding=c.downsample_padding)
        self.N_conv = nn.Conv1d(c.n_downsample_channels, c.n_downsample_channels, c.downsample_kernel, stride=c.downsample_stride, padding=c.downsample_padding)

    def _build_decoder_input(self, x, asr_out, f0_ds, n_ds):
        return torch.cat([x, asr_out, f0_ds, n_ds], dim=1)

    def forward(self, text_expanded, f0, n_amp, style):
        if style.dim() == 1:
            style = style.unsqueeze(0)
        decoder_style = style[:, :self._style_dim]

        f0_ds = self.F0_conv(f0)
        n_ds = self.N_conv(n_amp)

        encode_input = torch.cat([text_expanded, f0_ds, n_ds], dim=1)
        encode_out = self.encode(encode_input, decoder_style)

        asr_out = self.asr_res(text_expanded)

        x0 = self.decode0(self._build_decoder_input(encode_out, asr_out, f0_ds, n_ds), decoder_style)
        x1 = self.decode1(self._build_decoder_input(x0, asr_out, f0_ds, n_ds), decoder_style)
        x2 = self.decode2(self._build_decoder_input(x1, asr_out, f0_ds, n_ds), decoder_style)
        x3 = self.decode3(self._build_decoder_input(x2, asr_out, f0_ds, n_ds), decoder_style)

        return x3
