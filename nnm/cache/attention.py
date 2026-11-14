import torch

from .base import CacheLayer


class KVCache(CacheLayer):
    is_positional = True

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.position = 0
        self._keys = None
        self._values = None

    def _allocate(self, key, value):
        shape = (*key.shape[:-2], self.capacity, key.shape[-1])
        self._keys = torch.zeros(shape, dtype=key.dtype, device=key.device)
        shape = (*value.shape[:-2], self.capacity, value.shape[-1])
        self._values = torch.zeros(shape, dtype=value.dtype, device=value.device)

    def reserve(self, capacity: int):
        if capacity <= self.capacity:
            return
        if self._keys is not None:
            keys = torch.zeros(
                (*self._keys.shape[:-2], capacity, self._keys.shape[-1]),
                dtype=self._keys.dtype,
                device=self._keys.device,
            )
            values = torch.zeros(
                (*self._values.shape[:-2], capacity, self._values.shape[-1]),
                dtype=self._values.dtype,
                device=self._values.device,
            )
            keys[..., : self.position, :] = self._keys[..., : self.position, :]
            values[..., : self.position, :] = self._values[..., : self.position, :]
            self._keys = keys
            self._values = values
        self.capacity = capacity

    def update(self, key, value):
        length = key.shape[-2]
        if self._keys is None:
            self._allocate(key, value)
        self.reserve(self.position + length)
        end = self.position + length
        self._keys[..., self.position : end, :] = key
        self._values[..., self.position : end, :] = value
        self.position = end
        return self._keys[..., :end, :], self._values[..., :end, :]

    def clear(self):
        self.position = 0

    def is_empty(self):
        return self.position == 0

    def get_seq_length(self) -> int:
        return self.position

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.position + query_length, 0

    def get_max_cache_shape(self) -> int:
        return self.capacity

    @property
    def stored_length(self):
        return self.position


class SlidingWindowKVCache(CacheLayer):
    is_positional = True

    def __init__(self, window_size: int):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.cumulative_length = 0
        self._length = 0
        self._keys = None
        self._values = None

    def _allocate(self, key, value):
        shape = (*key.shape[:-2], self.window_size, key.shape[-1])
        self._keys = torch.zeros(shape, dtype=key.dtype, device=key.device)
        shape = (*value.shape[:-2], self.window_size, value.shape[-1])
        self._values = torch.zeros(shape, dtype=value.dtype, device=value.device)

    def update(self, key, value):
        length = key.shape[-2]
        if self._keys is None:
            self._allocate(key, value)

        if length == 1:
            if self._length + 1 > self.window_size:
                shift = self._length + 1 - self.window_size
                self._keys[..., : self._length - shift, :] = self._keys[..., shift : self._length, :].clone()
                self._values[..., : self._length - shift, :] = self._values[..., shift : self._length, :].clone()
                self._length -= shift
            self._keys[..., self._length : self._length + 1, :] = key
            self._values[..., self._length : self._length + 1, :] = value
            self._length += 1
            self.cumulative_length += 1
            return self._keys[..., : self._length, :], self._values[..., : self._length, :]

        if self._length:
            full_keys = torch.cat([self._keys[..., : self._length, :], key], dim=-2)
            full_values = torch.cat([self._values[..., : self._length, :], value], dim=-2)
        else:
            full_keys = key
            full_values = value

        store = min(self.cumulative_length + length, self.window_size - 1)
        self._keys[..., :store, :] = full_keys[..., -store:, :]
        self._values[..., :store, :] = full_values[..., -store:, :]
        self._length = store
        self.cumulative_length += length
        return full_keys, full_values

    def clear(self):
        self.cumulative_length = 0
        self._length = 0

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
        return self._length
