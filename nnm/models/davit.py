from torch import nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layers(x)
