import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from ..styletts2.blocks import AdainResBlk1d
from ..styletts2.decoder import DecoderBase
from ..styletts2.vocoder import AdaINResBlock1, SourceModuleHnNSF


class TorchSTFT(nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window_float64=False):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        if window_float64:
            n = torch.arange(win_length, dtype=torch.float64)
            window = (0.5 - 0.5 * torch.cos(2 * math.pi * n / win_length)).float()
        else:
            window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        self.register_buffer("window", window, persistent=False)

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(input_data.device),
            return_complex=True,
        )
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(magnitude.device),
        )
        return inverse_transform.unsqueeze(-2)


class Generator(nn.Module):
    def __init__(
        self,
        style_dim,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
        adain_affine=False,
        magnitude_activation="exp",
        window_float64=False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.magnitude_activation = magnitude_activation
        upsample_scale = math.prod(upsample_rates) * gen_istft_hop_size
        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000, upsample_scale=upsample_scale, harmonic_num=8, voiced_threshod=10
        )
        self.f0_upsamp = nn.Upsample(scale_factor=upsample_scale)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim, affine=adain_affine))
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0 + 1) // 2
                    )
                )
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1, 3, 5], style_dim, affine=adain_affine))
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1, 3, 5], style_dim, affine=adain_affine))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(nn.Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
            window_float64=window_float64,
        )

    def forward(self, x, style, f0):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, _, _ = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_res[i](self.noise_convs[i](har), style)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, style)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, style)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = x[:, : self.post_n_fft // 2 + 1, :]
        if self.magnitude_activation == "exp":
            spec = torch.exp(magnitude)
        elif self.magnitude_activation == "softplus":
            spec = F.softplus(magnitude)
        else:
            spec = F.softplus(magnitude) + 1e-5
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spec, phase)


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
        gen_istft_n_fft,
        gen_istft_hop_size,
        encode_out_channels,
        decode_out_channels,
        decode_upsample_out_channels,
        asr_res_in_channels,
        adain_affine=False,
        magnitude_activation="exp",
        training_downsample=False,
        window_float64=False,
    ):
        super().__init__()
        self.training_downsample = training_downsample
        self.encode = AdainResBlk1d(dim_in + 2, encode_out_channels, style_dim, affine=adain_affine)
        self.decode = nn.ModuleList()
        for _ in range(3):
            self.decode.append(AdainResBlk1d(decode_out_channels + 2 + 64, decode_out_channels, style_dim, affine=adain_affine))
        self.decode.append(
            AdainResBlk1d(
                decode_out_channels + 2 + 64, decode_upsample_out_channels, style_dim, upsample=True, affine=adain_affine
            )
        )
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
            gen_istft_n_fft,
            gen_istft_hop_size,
            adain_affine=adain_affine,
            magnitude_activation=magnitude_activation,
            window_float64=window_float64,
        )
