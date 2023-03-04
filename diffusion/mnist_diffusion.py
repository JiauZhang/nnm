import torch, torchvision
import torch.nn as nn
from datasets.mnist import get_dataloader
from torchvision.utils import save_image
import numpy as np

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
    def forward(self, inputs):
        return self.model(inputs)

# https://github.com/hojonathanho/diffusion/blob/master/scripts/run_cifar.py
class Diffusion(nn.Module):
    def __init__(self, beta_start=0.0001, beta_end=0.02, lr=2e-4, num_diffusion_timesteps=1000,
                 model_mean_type='eps', model_var_type='fixedlarge', loss_type='mse',
                 beta_schedule='linear', batch_size=16):
        super(Diffusion, self).__init__()
        self.batch_size = batch_size
        self.betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        self.num_timesteps = int(len(self.betas))
        alphas = 1. - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1. - self.alphas_cumprod)

    def training_losses(self, denoise_fn, x_start, t, noise=None):
        # Add noise to data
        assert t.shape == [x_start.shape[0]]
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape and noise.dtype == x_start.dtype
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        target = noise
        model_output = denoise_fn(x_t, t)
        assert model_output.shape == target.shape == x_start.shape
        losses = torch.mean(torch.abs(target, model_output))

        assert losses.shape == t.shape
        return losses

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = a[t]
        assert out.shape == [bs]
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def train(self):
        t = np.random.randint(self.num_timesteps, size=(self.num_timesteps,))
        