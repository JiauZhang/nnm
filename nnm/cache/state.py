import torch
import torch.nn.functional as F
from .base import Cache


class StateCache(Cache):
    def __init__(self, state_size: int):
        self.state_size = state_size
        self.state = None

    def update(self, x):
        if self.state is None:
            if x.shape[-1] >= self.state_size:
                self.state = x[..., -self.state_size :]
            else:
                self.state = F.pad(x, (self.state_size - x.shape[-1], 0))
        else:
            self.state = torch.cat([self.state, x], dim=-1)
            self.state = self.state[..., -self.state_size :]
        return self.state

    def clear(self):
        self.state = None

    def is_empty(self):
        return self.state is None
