from abc import ABC, abstractmethod


class CacheLayer(ABC):
    is_positional = False

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    def get_seq_length(self) -> int:
        return 0

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_max_cache_shape(self) -> int:
        return -1
