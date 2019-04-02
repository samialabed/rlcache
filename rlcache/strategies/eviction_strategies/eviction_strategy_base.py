from abc import ABC
from typing import Dict

from rlcache.backend.base import Storage
from rlcache.observers.observer import Observer


class EvictionStrategy(ABC):
    def __init__(self, config: Dict[str, any]):
        self.config = config

    def trim_cache(self, cache: Storage):
        """ Called when cache is full, finds an item to evict from the cache and evict it."""
        raise NotImplementedError

    def observer(self) -> Observer:
        """Returns an observer implementation that can be called on various methods."""
        pass
