import torch
from torch import nn

class Qwen2RMSNorm(nn.RMSNorm):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__(embed_dim, eps=eps, elementwise_affine=True)
