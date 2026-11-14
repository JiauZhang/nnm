import math
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from .blocks import AdainResBlk1d
from .vocoder import AdaINResBlock1, SourceModuleHnNSF


class Generator(nn.Module):
    def __init__(
        self,
        style_dim,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        upsample_scale = math.prod(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000, upsample_scale=upsample_scale, harmonic_num=8, voiced_threshod=10
        )
        self.f0_upsamp = nn.Upsample(scale_factor=upsample_scale)
        self.noise_convs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i), c_cur, k, u, padding=(u // 2 + u % 2), output_padding=u % 2
                    )
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0 + 1) // 2)
                )
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1, 3, 5], style_dim))
        self.resblocks = nn.ModuleList()
        self.alphas = nn.ParameterList([nn.Parameter(torch.ones(1, upsample_initial_channel, 1))])
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x, style, f0):
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2)
        for i in range(self.num_upsamples):
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x_source = self.noise_res[i](self.noise_convs[i](har_source), style)
            x = self.ups[i](x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, style)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, style)
            x = xs / self.num_kernels
        x = x + (1 / self.alphas[i + 1]) * (torch.sin(self.alphas[i + 1] * x) ** 2)
        x = self.conv_post(x)
        return torch.tanh(x)


class DecoderBase(nn.Module):
    def forward(self, asr, F0_curve, N, style):
        if self.training and self.training_downsample:
            F0_down = random.choice([0, 3, 7])
            N_down = random.choice([0, 3, 7, 15])
            if F0_down:
                kernel = torch.ones(1, 1, F0_down, device=F0_curve.device, dtype=F0_curve.dtype)
                F0_curve = F.conv1d(F0_curve.unsqueeze(1), kernel, padding=F0_down // 2).squeeze(1) / F0_down
            if N_down:
                kernel = torch.ones(1, 1, N_down, device=N.device, dtype=N.dtype)
                N = F.conv1d(N.unsqueeze(1), kernel, padding=N_down // 2).squeeze(1) / N_down
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, style)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, style)
            if block.upsample_type != "none":
                res = False
        return self.generator(x, style, F0_curve)


class Decoder(DecoderBase):
    def __init__(
        self,
        dim_in,
        style_dim,
        dim_out,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        encode_out_channels=None,
        decode_out_channels=None,
        decode_upsample_out_channels=None,
        asr_res_in_channels=None,
        training_downsample=True,
    ):
        super().__init__()
        encode_out_channels = 2 * upsample_initial_channel if encode_out_channels is None else encode_out_channels
        decode_out_channels = 2 * upsample_initial_channel if decode_out_channels is None else decode_out_channels
        decode_upsample_out_channels = (
            upsample_initial_channel if decode_upsample_out_channels is None else decode_upsample_out_channels
        )
        asr_res_in_channels = dim_in if asr_res_in_channels is None else asr_res_in_channels
        self.training_downsample = training_downsample
        self.encode = AdainResBlk1d(dim_in + 2, encode_out_channels, style_dim)
        self.decode = nn.ModuleList(
            [AdainResBlk1d(decode_out_channels + 2 + 64, decode_out_channels, style_dim) for _ in range(3)]
        )
        self.decode.append(AdainResBlk1d(decode_out_channels + 2 + 64, decode_upsample_out_channels, style_dim, upsample=True))
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(asr_res_in_channels, 64, kernel_size=1)))
        self.generator = Generator(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
        )
