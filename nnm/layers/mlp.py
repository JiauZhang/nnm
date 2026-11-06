from torch import nn

from nnm.layers.activation import get_activation


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation="gelu", bias=True, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.act(self.dropout(self.fc1(x))))


class GatedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False, activation="silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = get_activation(activation)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
