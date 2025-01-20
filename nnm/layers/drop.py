import torch
from torch import nn

class DropPath(nn.Module):
    def __init__(self, prob=0.5, normalization=True):
        super().__init__()
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f'DropPath probability has to be between 0 and 1, but got {prob}')
        self.prob = prob
        self.normalization = normalization

    def forward(self, input):
        if self.prob == 0.0 or self.training == False:
            return input
        elif self.prob == 1.0:
            return torch.zeros_like(input)
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        keep_prob = 1 - self.prob
        mask = torch.empty(shape).bernoulli_(keep_prob)
        if self.normalization:
            mask.div_(keep_prob)
        return input * mask
