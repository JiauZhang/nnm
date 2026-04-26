import torch
from .base import Cache


class KVCache(Cache):
    def __init__(self):
        self._k_cache = []
        self._v_cache = []

    def update(self, k, v):
        self._k_cache.append(k)
        self._v_cache.append(v)
        return torch.cat(self._k_cache, dim=-2), torch.cat(self._v_cache, dim=-2)

    def clear(self):
        self._k_cache.clear()
        self._v_cache.clear()

    def is_empty(self):
        return len(self._k_cache) == 0

    @property
    def kv_len(self):
        if not self._k_cache:
            return 0
        return sum(k.shape[-2] for k in self._k_cache)


class SlidingWindowKVCache(Cache):
    def __init__(self, window_size: int):
        self.window_size = window_size
        self._k_cache = []
        self._v_cache = []
        self.cumulative_length = 0

    def update(self, k, v):
        self.cumulative_length += k.shape[-2]
        self._k_cache.append(k)
        self._v_cache.append(v)

        full_k = torch.cat(self._k_cache, dim=-2)
        full_v = torch.cat(self._v_cache, dim=-2)

        if full_k.shape[-2] > self.window_size - 1:
            full_k = full_k[:, :, -self.window_size + 1 :, :]
            full_v = full_v[:, :, -self.window_size + 1 :, :]

        self._k_cache = [full_k]
        self._v_cache = [full_v]

        return full_k, full_v

    def clear(self):
        self._k_cache.clear()
        self._v_cache.clear()
        self.cumulative_length = 0

    def is_empty(self):
        return len(self._k_cache) == 0

    @property
    def kv_len(self):
        if not self._k_cache:
            return 0
        return self._k_cache[0].shape[-2]
