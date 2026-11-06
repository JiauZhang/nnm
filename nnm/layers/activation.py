import torch
import torch.nn.functional as F
from torch import nn


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


ACTIVATIONS = {
    "gelu": F.gelu,
    "silu": F.silu,
    "relu": F.relu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "quick_gelu": quick_gelu,
}


def get_activation(name):
    if name in ACTIVATIONS:
        return ACTIVATIONS[name]
    return getattr(F, name)


class Activation(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.fn = get_activation(name)

    def forward(self, hidden):
        return self.fn(hidden)
