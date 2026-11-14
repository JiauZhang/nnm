import torch
from torch import nn


class LayerScale(nn.Module):
    def __init__(self, dim, init_scale=1e-6, spatial=False):
        super().__init__()
        shape = (dim, 1, 1) if spatial else (dim,)
        self.gamma = nn.Parameter(torch.full(shape, init_scale))

    def forward(self, hidden):
        return self.gamma * hidden
