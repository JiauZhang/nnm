import torch
from .base import Cache


class KVCache(Cache):
    def __init__(self):
        self._k_cache = None
        self._v_cache = None

    def update(self, k, v):
        if self._k_cache is None:
            self._k_cache = k
            self._v_cache = v
        else:
            self._k_cache = torch.cat([self._k_cache, k], dim=-2)
            self._v_cache = torch.cat([self._v_cache, v], dim=-2)
        return self._k_cache, self._v_cache

    def clear(self):
        self._k_cache = None
        self._v_cache = None

    def is_empty(self):
        return self._k_cache is None

    @property
    def kv_len(self):
        return 0 if self._k_cache is None else self._k_cache.shape[-2]


class SlidingWindowKVCache(Cache):
    def __init__(self, window_size: int):
        self.window_size = window_size
        self._k_cache = None
        self._v_cache = None

    def update(self, k, v):
        if self._k_cache is None:
            if k.shape[-2] > self.window_size:
                self._k_cache = k[..., -self.window_size:, :].clone()
                self._v_cache = v[..., -self.window_size:, :].clone()
            else:
                self._k_cache = k.clone()
                self._v_cache = v.clone()
            return self._k_cache, self._v_cache

        new_seq_len = k.shape[-2]
        current_len = self._k_cache.shape[-2]

        if current_len < self.window_size:
            total_len = current_len + new_seq_len
            if total_len <= self.window_size:
                self._k_cache = torch.cat([self._k_cache, k], dim=-2)
                self._v_cache = torch.cat([self._v_cache, v], dim=-2)
            else:
                self._k_cache = torch.cat([self._k_cache, k], dim=-2)
                self._v_cache = torch.cat([self._v_cache, v], dim=-2)
                self._k_cache = self._k_cache[..., -self.window_size:, :].clone()
                self._v_cache = self._v_cache[..., -self.window_size:, :].clone()
        else:
            if new_seq_len >= self.window_size:
                self._k_cache = k[..., -self.window_size:, :].clone()
                self._v_cache = v[..., -self.window_size:, :].clone()
            else:
                keep_len = self.window_size - new_seq_len
                self._k_cache[..., :-new_seq_len, :] = self._k_cache[..., -keep_len:, :]
                self._k_cache[..., -new_seq_len:, :] = k
                self._v_cache[..., :-new_seq_len, :] = self._v_cache[..., -keep_len:, :]
                self._v_cache[..., -new_seq_len:, :] = v

        return self._k_cache, self._v_cache

    def clear(self):
        self._k_cache = None
        self._v_cache = None

    def is_empty(self):
        return self._k_cache is None

    @property
    def kv_len(self):
        return 0 if self._k_cache is None else self._k_cache.shape[-2]
