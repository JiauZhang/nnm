import torch.nn.functional as F

def get_activation(name):
    return getattr(F, name)
