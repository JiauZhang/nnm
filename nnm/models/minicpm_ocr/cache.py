import torch
import torch.nn.functional as F

from nnm.cache import Cache, CacheLayer, KVCache


class GatedDeltaNetCache(CacheLayer):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.conv_state = None
        self.recurrent_state = None

    def update_conv(self, x):
        if self.conv_state is None:
            combined = F.pad(x, (self.kernel_size - 1, 0))
        else:
            combined = torch.cat([self.conv_state, x], dim=-1)
        self.conv_state = combined[..., -self.kernel_size :].clone()
        return combined

    def update_recurrent(self, state):
        self.recurrent_state = state
        return state

    def update(self, x):
        return self.update_conv(x)

    def clear(self):
        self.conv_state = None
        self.recurrent_state = None

    def is_empty(self):
        return self.conv_state is None


def build_layers_cache(layer_types, text_config):
    layers = []
    for layer_type in layer_types:
        if layer_type == "full_attention":
            layers.append(KVCache())
        else:
            layers.append(GatedDeltaNetCache(text_config.linear_conv_kernel_dim))
    return Cache(layers)
