import math

import torch
from torch import nn

GATE_BAND_HZ = (250.0, 3000.0)


class LogMel(nn.Module):
    """The learned 257-by-80 front-end filter, filled from the fbank asset when the model is loaded."""

    def __init__(self, config, floor=1e-10):
        super().__init__()
        self.sample_rate, self.n_fft = config.sample_rate, config.n_fft
        self.win_length, self.hop_length, self.floor = config.win_length, config.hop_length, floor
        self.register_buffer("fbank", torch.zeros(config.n_fft // 2 + 1, config.mel_bins), persistent=False)
        self.register_buffer("window", torch.hann_window(config.win_length, periodic=False), persistent=False)

    def forward(self, audio):
        power = self._spectrum(audio).abs().transpose(-2, -1).pow(2) @ self.fbank
        log_mel = power.clamp(min=self.floor).log()
        mean = log_mel.mean(0, keepdim=True)
        var = ((log_mel - mean) ** 2).mean(0, keepdim=True)
        return (log_mel - mean) / torch.sqrt(var + 1e-5)

    def levels(self, audio):
        """Per-frame level in dB over the 250-3000 Hz band the engine gates on."""
        low, high = (round(hz * self.n_fft / self.sample_rate) for hz in GATE_BAND_HZ)
        band = self._spectrum(audio).abs().pow(2)[low : high + 1]
        return 10 * torch.log10(band.sum(0) + 1e-30)

    def _spectrum(self, audio):
        frames = max(1, audio.shape[-1] // self.hop_length)
        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="constant",
            normalized=True,
            return_complex=True,
        )
        return spectrum[:, :frames]


class ConvStem(nn.Module):
    def __init__(self, channels, d_model, freq_bins=80):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, 3, stride=2, padding=1, bias=False)
        for stage in (1, 2):
            setattr(self, f"dw_{stage}", nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels, bias=False))
            setattr(self, f"pw_{stage}", nn.Linear(channels, channels, bias=False))
        self.freq_bins = freq_bins
        self.out = nn.Linear(channels * ((freq_bins + 7) // 8), d_model, bias=False)

    def forward(self, mel):
        x = torch.nn.functional.silu(self.conv(mel[None, None]))
        for stage in (1, 2):
            x = getattr(self, f"dw_{stage}")(x)
            x = torch.nn.functional.silu(getattr(self, f"pw_{stage}")(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = x.permute(0, 2, 1, 3).flatten(2)
        return self.out(x)[0]
