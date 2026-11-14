import math

import torch
from torch import nn


class VSampler(nn.Module):
    def get_alpha_beta(self, sigma):
        angle = sigma * math.pi / 2
        return math.cos(angle), math.sin(angle)

    def forward(self, noise, fn, sigmas, num_steps):
        x = sigmas[0] * noise
        alpha, beta = self.get_alpha_beta(sigmas[0].item())
        for i in range(num_steps - 1):
            is_last = i == num_steps - 1
            x_denoised = fn(x, sigma=sigmas[i])
            x_pred = x * alpha - x_denoised * beta
            x_eps = x * beta + x_denoised * alpha
            if not is_last:
                alpha, beta = self.get_alpha_beta(sigmas[i + 1].item())
                x = x_pred * alpha + x_eps * beta
        return x_pred


class KarrasSampler(nn.Module):
    def __init__(self, s_tmin=0, s_tmax=float("inf"), s_churn=0.0, s_noise=1.0):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.s_churn = s_churn

    def step(self, x, fn, sigma, sigma_next, gamma):
        sigma_hat = sigma + gamma * sigma
        epsilon = self.s_noise * torch.randn_like(x)
        x_hat = x + math.sqrt(sigma_hat**2 - sigma**2) * epsilon
        d = (x_hat - fn(x_hat, sigma=sigma_hat)) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat) * d
        if sigma_next != 0:
            model_out_next = fn(x_next, sigma=sigma_next)
            d_prime = (x_next - model_out_next) / sigma_next
            x_next = x_hat + 0.5 * (sigma - sigma_hat) * (d + d_prime)
        return x_next

    def forward(self, noise, fn, sigmas, num_steps):
        x = sigmas[0] * noise
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax), min(self.s_churn / num_steps, math.sqrt(2) - 1), 0.0
        )
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1], gamma=gammas[i])
        return x


class ADPM2Sampler(nn.Module):
    def __init__(self, rho=1.0):
        super().__init__()
        self.rho = rho

    def get_sigmas(self, sigma, sigma_next):
        r = self.rho
        sigma_up = math.sqrt(sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2)
        sigma_down = math.sqrt(sigma_next**2 - sigma_up**2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def step(self, x, fn, sigma, sigma_next):
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)
        d = (x - fn(x, sigma=sigma)) / sigma
        x_mid = x + d * (sigma_mid - sigma)
        d_mid = (x_mid - fn(x_mid, sigma=sigma_mid)) / sigma_mid
        x = x + d_mid * (sigma_down - sigma)
        return x + torch.randn_like(x) * sigma_up

    def forward(self, noise, fn, sigmas, num_steps):
        x = sigmas[0] * noise
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])
        return x


class DiffusionSampler(nn.Module):
    def __init__(self, diffusion, *, sampler, sigma_schedule, num_steps=None, clamp=True):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.sampler = sampler
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps
        self.clamp = clamp

    def forward(self, noise, num_steps=None, **kwargs):
        num_steps = self.num_steps if num_steps is None else num_steps
        sigmas = self.sigma_schedule(num_steps, noise.device)

        def fn(*a, **ka):
            return self.denoise_fn(*a, **{**ka, **kwargs})

        x = self.sampler(noise, fn=fn, sigmas=sigmas, num_steps=num_steps)
        return x.clamp(-1.0, 1.0) if self.clamp else x
