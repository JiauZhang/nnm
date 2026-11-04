from torch import nn


class MlpProjector(nn.Module):
    def __init__(self, *, input_dim, n_embed):
        super().__init__()
        self.layers = nn.Linear(input_dim, n_embed)

    def forward(self, x):
        return self.layers(x)
