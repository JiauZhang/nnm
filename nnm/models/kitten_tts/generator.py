import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class AdaIN(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.fc = nn.Linear(style_dim, 2 * channels)
        self.channels = channels

    def forward(self, x, style):
        x_norm = self.norm(x)
        style_params = self.fc(style)
        gamma = style_params[:, :self.channels].unsqueeze(2)
        beta = style_params[:, self.channels:].unsqueeze(2)
        return x_norm * (gamma + 1.0) + beta


def _interpolate_1d_half_pixel(x, target_size, mode='linear'):
    B, C, L = x.shape
    positions = (torch.arange(target_size, device=x.device, dtype=x.dtype) + 0.5) * L / target_size - 0.5
    pos_floor = torch.clamp(torch.floor(positions).long(), 0, L - 1)
    pos_ceil = torch.clamp(pos_floor + 1, 0, L - 1)
    alpha = (positions - pos_floor.float()).clamp(0, 1)

    x_left = x[:, :, pos_floor]
    x_right = x[:, :, pos_ceil]
    return x_left * (1 - alpha.view(1, 1, -1)) + x_right * alpha.view(1, 1, -1)


class NoiseResBlock(nn.Module):
    def __init__(
        self,
        channels,
        style_dim,
        kernel_size=3,
        dilations1=(1, 3, 5),
        dilations2=(1, 1, 1),
        padding_mode='auto',
        base_padding=None,
    ):
        super().__init__()
        self.padding_mode = padding_mode

        if padding_mode == 'auto':
            paddings1 = [get_padding(kernel_size, d) for d in dilations1]
            paddings2 = [get_padding(kernel_size, 1) for _ in dilations2]
        else:
            bp = base_padding or 3
            paddings1 = [bp * d for d in dilations1]
            paddings2 = [bp * d for d in dilations2]

        self.convs1 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, padding=p, dilation=d)
            for p, d in zip(paddings1, dilations1)
        ])
        self.convs2 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, padding=p, dilation=d)
            for p, d in zip(paddings2, dilations2)
        ])

        n_stages = len(dilations1)
        self.adain1 = nn.ModuleList([AdaIN(channels, style_dim) for _ in range(n_stages)])
        self.adain2 = nn.ModuleList([AdaIN(channels, style_dim) for _ in range(n_stages)])

        self.alpha1 = nn.ParameterList([
            nn.Parameter(torch.randn(1, channels, 1) * 0.02) for _ in range(n_stages)
        ])
        self.alpha2 = nn.ParameterList([
            nn.Parameter(torch.randn(1, channels, 1) * 0.02) for _ in range(n_stages)
        ])

        self.reciprocal1 = nn.ParameterList([
            nn.Parameter(torch.ones(1, channels, 1)) for _ in range(n_stages)
        ])
        self.reciprocal2 = nn.ParameterList([
            nn.Parameter(torch.ones(1, channels, 1)) for _ in range(n_stages)
        ])

    def _generate_noise(self, adain_output, alpha, reciprocal_const):
        scaled = adain_output * alpha
        sin_val = torch.sin(scaled)
        pow_val = sin_val ** 2
        noise = reciprocal_const * pow_val
        return noise

    def forward(self, x, style):
        prev_stage_output = x

        for c1, c2, ad1, ad2, a1, a2, r1, r2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2,
            self.alpha1, self.alpha2, self.reciprocal1, self.reciprocal2
        ):
            xt_adain = ad1(x, style)
            noise1 = self._generate_noise(xt_adain, a1, r1)
            xt_add = xt_adain + noise1
            xt_c1 = c1(xt_add)

            xt_adain2 = ad2(xt_c1, style)
            noise2 = self._generate_noise(xt_adain2, a2, r2)
            xt_add2 = xt_adain2 + noise2
            xt_c2 = c2(xt_add2)

            x = xt_c2 + prev_stage_output
            prev_stage_output = x

        return x


class MSinGenerator(nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        hop_length=300,
        n_harmonics=9,
        voiced_threshold=10.0,
        phase_jitter_scale=0.01,
    ):
        super().__init__()
        self.l_linear = nn.Linear(n_harmonics, 1, bias=True)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_harmonics = n_harmonics
        self.voiced_threshold = voiced_threshold
        self.phase_jitter_scale = phase_jitter_scale

    def forward(self, f0):
        B, _, T = f0.shape
        audio_len = T * self.hop_length

        f0_t = f0.transpose(1, 2)
        harmonics = torch.arange(1, self.n_harmonics + 1, device=f0.device).float().view(1, 1, -1)
        phase_inc = f0_t * harmonics
        phase_inc = phase_inc / self.sample_rate
        phase_inc = phase_inc - torch.floor(phase_inc)

        phase_jitter = torch.rand_like(phase_inc) * self.phase_jitter_scale
        phase_inc = phase_inc + phase_jitter
        phase_inc = phase_inc - torch.floor(phase_inc)

        phase_inc_expanded = phase_inc.transpose(1, 2)
        phase_inc_expanded = phase_inc_expanded.unsqueeze(-1).expand(B, self.n_harmonics, T, self.hop_length)
        phase_inc_expanded = phase_inc_expanded.reshape(B, self.n_harmonics, audio_len)

        phase_frame = _interpolate_1d_half_pixel(
            phase_inc_expanded, T, mode='linear'
        )

        phase_frame = torch.cumsum(phase_frame, dim=2)
        phase_frame = phase_frame * 2.0 * math.pi * self.hop_length

        phase_sample = _interpolate_1d_half_pixel(
            phase_frame, audio_len, mode='linear'
        )

        phase_sample = phase_sample.transpose(1, 2)
        sine_waves = torch.sin(phase_sample) * 0.1

        voiced_mask = (f0 > self.voiced_threshold).float()
        voiced_mask_sample = F.interpolate(voiced_mask, size=audio_len, mode='nearest')
        voiced_mask_expanded = voiced_mask_sample.transpose(1, 2).expand(B, audio_len, self.n_harmonics)

        sine_waves = sine_waves * voiced_mask_expanded

        unvoiced_mask = 1.0 - voiced_mask_expanded
        noise_scale = voiced_mask_expanded * 0.003 + unvoiced_mask * (0.1 / 3.0)
        sine_waves = sine_waves + torch.randn_like(sine_waves) * noise_scale

        signal = torch.tanh(self.l_linear(sine_waves))
        return signal


class MSource(nn.Module):
    def __init__(self, stft_real, stft_imag):
        super().__init__()
        self.sin_gen = MSinGenerator()
        self.register_buffer('stft_real', stft_real)
        self.register_buffer('stft_imag', stft_imag)

    def forward(self, f0):
        signal = self.sin_gen(f0)
        signal = signal.transpose(1, 2)
        signal = F.pad(signal, (10, 10))

        real = F.conv1d(signal, self.stft_real, stride=5, padding=0)
        imag = F.conv1d(signal, self.stft_imag, stride=5, padding=0)

        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase = torch.atan2(imag, real + 1e-8)

        return torch.cat([magnitude, phase], dim=1)


class KittenGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.generator
        style_dim = c.style_dim
        self.leaky_relu_slope = c.leaky_relu_slope

        self._init_from_up_blocks(c, style_dim)

        self.m_source = MSource(self.stft_weight_forward_real, self.stft_weight_forward_imag)

    def _init_from_up_blocks(self, c, style_dim):
        self.ups = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for blk in c.up_blocks:
            self.ups.append(ConvTranspose1d(
                blk["in_channels"], blk["out_channels"],
                kernel_size=blk["upsample_kernel_size"],
                stride=blk["upsample_stride"],
                padding=blk["upsample_padding"],
            ))
            self.noise_res.append(NoiseResBlock(
                blk["out_channels"], style_dim=style_dim,
                kernel_size=blk["noise_res_kernel_size"],
                padding_mode=blk["noise_res_padding_mode"],
                base_padding=blk["noise_res_base_padding"],
            ))
            self.noise_convs.append(Conv1d(
                c.m_source_out, blk["out_channels"],
                kernel_size=blk["noise_conv_kernel_size"],
                stride=blk["noise_conv_stride"],
                padding=blk["noise_conv_padding"],
            ))
            for _ in range(blk["n_resblocks"]):
                self.resblocks.append(NoiseResBlock(
                    blk["out_channels"], style_dim=style_dim,
                    kernel_size=blk["resblock_kernel_size"],
                    dilations1=tuple(blk["resblock_dilations1"]),
                    dilations2=tuple(blk["resblock_dilations2"]),
                ))

        last_blk = c.up_blocks[-1]
        self.conv_post = Conv1d(
            last_blk["out_channels"], c.m_source_out,
            kernel_size=c.conv_post_kernel_size,
            padding=c.conv_post_padding,
        )

        self.stft_weight_forward_real = nn.Parameter(torch.randn(c.stft_n_bins, 1, c.stft_weight_size))
        self.stft_weight_forward_imag = nn.Parameter(torch.randn(c.stft_n_bins, 1, c.stft_weight_size))
        self.stft_weight_backward_real = nn.Parameter(torch.randn(c.stft_n_bins, 1, c.stft_weight_size))
        self.stft_weight_backward_imag = nn.Parameter(torch.randn(c.stft_n_bins, 1, c.stft_weight_size))

    def _stft_postprocess(self, x):
        half = x.shape[1] // 2
        x_imag = x[:, half:, :]
        x_real = x[:, :half, :]

        sin_val = torch.sin(x_imag)
        sin_sin = torch.sin(sin_val)
        exp_val = torch.exp(x_real)

        mul_imag = exp_val * sin_sin
        cos_val = torch.cos(sin_val)
        mul_real = exp_val * cos_val

        imag_out = F.conv_transpose1d(
            mul_imag,
            self.stft_weight_backward_imag,
            stride=5,
            padding=0,
            groups=1
        )

        real_out = F.conv_transpose1d(
            mul_real,
            self.stft_weight_backward_real,
            stride=5,
            padding=0,
            groups=1
        )

        waveform = real_out - imag_out
        waveform = waveform[:, 0:1, 10:-10]
        if waveform.shape[0] == 1:
            waveform = waveform.squeeze(0).squeeze(0)
        else:
            waveform = waveform.squeeze(1)

        return waveform

    def forward(self, x, style, harmonic_features):
        if style.dim() == 1:
            style = style.unsqueeze(0)
        style_slice = style[:, :128]

        x = F.leaky_relu(x, self.leaky_relu_slope)

        resblock_idx = 0
        for i, blk in enumerate(self.config.generator.up_blocks):
            x_up = self.ups[i](x)
            if blk["pad_left"] > 0:
                x_up = F.pad(x_up, (blk["pad_left"], 0), mode='reflect')

            noise_conv = self.noise_convs[i](harmonic_features)
            if noise_conv.shape[2] < x_up.shape[2]:
                noise_conv = F.pad(noise_conv, (x_up.shape[2] - noise_conv.shape[2], 0))

            x_noise = self.noise_res[i](noise_conv, style_slice)
            x = x_up + x_noise

            n_res = blk["n_resblocks"]
            res_outs = []
            for _ in range(n_res):
                res_outs.append(self.resblocks[resblock_idx](x, style_slice))
                resblock_idx += 1
            x = sum(res_outs) / len(res_outs)

            x = F.leaky_relu(x, self.leaky_relu_slope)

        x = F.leaky_relu(x, 0.01)
        x = self.conv_post(x)
        waveform = self._stft_postprocess(x)
        return waveform
