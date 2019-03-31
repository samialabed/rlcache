from abc import ABC
from typing import Union

from backend.base import Storage
from observers.observer import Observer


class EvictionStrategy(ABC):
    def trim_cache(self, cache: Storage):
        """ Called when cache is full, finds an item to evict from the cache and evict it."""
        raise NotImplementedError

    def observer(self) -> Union[Observer, None]:
        """Returns an observer implementation that can be called on various methods."""
        return None
