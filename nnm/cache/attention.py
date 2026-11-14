import torch

from .base import CacheLayer


class KVCache(CacheLayer):
    is_positional = True

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

    def get_seq_length(self) -> int:
        return 0 if self._k_cache is None else self._k_cache.shape[-2]

    @property
    def stored_length(self):
        return self.get_seq_length()


class SlidingWindowKVCache(CacheLayer):
    is_positional = True

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.cumulative_length = 0
        self._k_cache = None
        self._v_cache = None

    def update(self, k, v):
        new_len = k.shape[-2]
        self.cumulative_length += new_len
        if self._k_cache is None:
            full_k, full_v = k, v
        else:
            full_k = torch.cat([self._k_cache, k], dim=-2)
            full_v = torch.cat([self._v_cache, v], dim=-2)
        keep = max(self.window_size - 1, 0)
        self._k_cache = full_k[..., -keep:, :].clone() if keep else None
        self._v_cache = full_v[..., -keep:, :].clone() if keep else None
        return full_k, full_v

    def clear(self):
        self.cumulative_length = 0
        self._k_cache = None
        self._v_cache = None

    def is_empty(self):
        return self.cumulative_length == 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        is_full = self.cumulative_length >= self.window_size
        kv_offset = max(self.cumulative_length - self.window_size + 1, 0)
        kv_length = self.window_size - 1 + query_length if is_full else self.cumulative_length + query_length
        return kv_length, kv_offset

    def get_max_cache_shape(self) -> int:
        return self.window_size

    @property
    def stored_length(self):
        return 0 if self._k_cache is None else self._k_cache.shape[-2]
