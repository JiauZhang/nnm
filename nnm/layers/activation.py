import torch
import torch.nn.functional as F


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
