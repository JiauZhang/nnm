import torch
from torch import nn
from torch.nn import functional as F


class LogNormalDistribution:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, num_samples, device=torch.device("cpu")):
        return (self.mean + self.std * torch.randn((num_samples,), device=device)).exp()


class UniformDistribution:
    def __call__(self, num_samples, device=torch.device("cpu")):
        return torch.rand(num_samples, device=device)


class LinearSchedule(nn.Module):
    def forward(self, num_steps, device):
        return torch.linspace(1, 0, num_steps + 1)[:-1]


class KarrasSchedule(nn.Module):
    def __init__(self, sigma_min, sigma_max, rho=7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps, device):
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**rho_inv + (steps / (num_steps - 1)) * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
        ) ** self.rho
        return F.pad(sigmas, pad=(0, 1), value=0.0)
