from abc import ABC, abstractmethod


class Cache(ABC):
    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass
