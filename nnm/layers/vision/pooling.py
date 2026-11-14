import torch
from torch import nn

from nnm.layers.attention import CrossAttention
from nnm.layers.mlp import MLP


class AttentionPool(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, activation, norm_eps):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.attention = CrossAttention(
            hidden_size,
            num_heads,
            head_dim=hidden_size // num_heads,
            qkv_bias=True,
            o_bias=True,
        )
        self.layernorm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size, activation, bias=True)

    def forward(self, hidden):
        probe = self.probe.expand(hidden.shape[0], -1, -1)
        pooled = self.attention(probe, hidden)
        return (pooled + self.mlp(self.layernorm(pooled)))[:, 0]
