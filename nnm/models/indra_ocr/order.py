import math

import torch
from torch import nn

from nnm.models.pretrained import PretrainedModel


class OrderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.scale = math.sqrt(self.head_dim)
        self.eps = config.layer_norm_eps

        self.n1 = nn.LayerNorm(config.dim, eps=self.eps)
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.proj = nn.Linear(config.dim, config.dim)
        self.n2 = nn.LayerNorm(config.dim, eps=self.eps)
        self.ff = nn.Sequential(
            nn.Linear(config.dim, config.ffn_channels),
            nn.GELU(),
            nn.Linear(config.ffn_channels, config.dim),
        )

    def forward(self, x, mask):
        residual = x
        hidden = self.n1(x)
        qkv = self.qkv(hidden).reshape(*hidden.shape[:2], 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = query @ key.transpose(-1, -2) / self.scale
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        hidden = torch.softmax(scores, dim=-1) @ value
        hidden = hidden.transpose(1, 2).reshape(*x.shape[:2], self.heads * self.head_dim)
        x = residual + self.proj(hidden)

        residual = x
        return residual + self.ff(self.n2(x))


class IndraOCROrder(PretrainedModel):
    module_name = "order"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.eps = config.layer_norm_eps
        self.register_buffer(
            "frequencies",
            torch.pi * 0.5 * 2.0 ** torch.arange(config.fourier_features, dtype=torch.float32),
            persistent=False,
        )

        self.inp = nn.Sequential(
            nn.Linear(config.input_features + 8 * config.fourier_features, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim),
        )
        self.layers = nn.ModuleList([OrderLayer(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.dim, eps=self.eps)
        self.q = nn.Linear(config.dim, config.dim)
        self.k = nn.Linear(config.dim, config.dim)
        self.bu = nn.Linear(config.dim, config.dim)
        self.bw = nn.Linear(config.dim, config.dim)
        self.cls = nn.Linear(config.dim, config.classes)
        self.tu = nn.Linear(config.dim, config.tables * config.dim)
        self.tw = nn.Linear(config.dim, config.tables * config.dim)
        self.scale = math.sqrt(config.dim)

    def fourier(self, feats):
        angles = feats[..., :4].unsqueeze(-1) * self.frequencies
        encoding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return encoding.flatten(-2)

    def forward(self, feats, mask):
        hidden = self.inp(torch.cat([feats, self.fourier(feats)], dim=-1))
        for layer in self.layers:
            hidden = layer(hidden, mask)

        hidden = self.norm(hidden)
        direct = self.q(hidden) @ self.k(hidden).transpose(-1, -2) / self.scale
        a = direct - direct.transpose(-1, -2)

        block = self.bu(hidden) @ self.bw(hidden).transpose(-1, -2) / self.scale
        same = block + block.transpose(-1, -2)

        tu = self.tu(hidden).reshape(*hidden.shape[:2], self.config.tables, self.dim).permute(0, 2, 1, 3)
        tw = self.tw(hidden).reshape(*hidden.shape[:2], self.config.tables, self.dim).permute(0, 2, 3, 1)
        table = tu @ tw / self.scale
        tab = table + table.transpose(-1, -2)

        return a, self.cls(hidden), same, tab
