import torch
from torch import nn
from torch.nn import functional as F

from .sampler import ADPM2Sampler, DiffusionSampler
from .schedule import KarrasSchedule


def _to_batch(batch_size, device, x=None, xs=None):
    if x is not None:
        return torch.full((batch_size,), fill_value=x, device=device)
    return xs


class KDiffusion(nn.Module):
    def __init__(self, net, *, sigma_distribution, sigma_data, dynamic_threshold=0.0):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas):
        sigma_data = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        sigmas = sigmas[:, None, None]
        c_skip = (sigma_data**2) / (sigmas**2 + sigma_data**2)
        c_out = sigmas * sigma_data * (sigma_data**2 + sigmas**2) ** -0.5
        c_in = (sigmas**2 + sigma_data**2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(self, x_noisy, sigmas=None, sigma=None, **kwargs):
        sigmas = _to_batch(x_noisy.shape[0], x_noisy.device, x=sigma, xs=sigmas)
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        return c_skip * x_noisy + c_out * x_pred

    def loss_weight(self, sigmas):
        return (sigmas**2 + self.sigma_data**2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x, noise=None, **kwargs):
        batch_size, device = x.shape[0], x.device
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        noise = torch.randn_like(x) if noise is None else noise
        x_noisy = x + sigmas[:, None, None] * noise
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)
        losses = F.mse_loss(x_denoised, x, reduction="none").flatten(1).mean(1)
        return (losses * self.loss_weight(sigmas)).mean()

    def sample(self, noise, num_steps=5, **kwargs):
        sampler = DiffusionSampler(
            self, sampler=ADPM2Sampler(), sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), clamp=False
        )
        return sampler(noise, num_steps=num_steps, **kwargs)
