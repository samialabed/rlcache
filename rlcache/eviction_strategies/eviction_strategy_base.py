from abc import ABC

from backend import Storage
from observers.observer import Observer


class EvictionStrategy(ABC):
    def __init__(self, cache: Storage):
        self.cache = cache

    def trim_cache(self, cache: Storage):
        """ Called when cache is full, finds an item to evict from the cache and evict it."""
        raise NotImplementedError

    def observer(self) -> Observer:
        """Returns an observer implementation that can be called on various methods."""
        pass
