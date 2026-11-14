import torch

from .attention import KVCache, SlidingWindowKVCache
from .base import CacheLayer
from .state import StateCache


class Cache:
    def __init__(self, layers: list[CacheLayer]):
        self.layers = list(layers)

    @classmethod
    def from_layer_types(cls, layer_types, *, capacity: int, window_size=None, state_size=3):
        layers = []
        for layer_type in layer_types:
            if layer_type == "conv":
                layers.append(StateCache(state_size))
            elif layer_type == "sliding_attention":
                if window_size is None:
                    raise ValueError("sliding_attention layers require window_size")
                layers.append(SlidingWindowKVCache(window_size))
            else:
                layers.append(KVCache(capacity))
        return cls(layers)

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index):
        return self.layers[index]

    def _positional_layer(self):
        for layer in self.layers:
            if layer.is_positional:
                return layer
        return None

    def get_seq_length(self) -> int:
        layer = self._positional_layer()
        return 0 if layer is None else layer.get_seq_length()

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        layer = self._positional_layer()
        if layer is None:
            return query_length, 0
        return layer.get_mask_sizes(query_length)

    def get_positions(self, query_length: int, device=None) -> torch.Tensor:
        start = self.get_seq_length()
        return torch.arange(start, start + query_length, device=device)

    def reserve(self, capacity: int):
        for layer in self.layers:
            layer.reserve(capacity)

    def clear(self):
        for layer in self.layers:
            layer.clear()

    def is_empty(self) -> bool:
        return all(layer.is_empty() for layer in self.layers)
