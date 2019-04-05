from abc import ABC

from rlcache.backend.base import Storage
from rlcache.observer import Observer
from rlcache.strategies.BaseStrategy import BaseStrategy


class EvictionStrategy(BaseStrategy, ABC):
    # TODO should this take a cache as well?

    def trim_cache(self, cache: Storage):
        """ Called when cache is full, finds an item to evict from the cache and evict it."""
        raise NotImplementedError

    def observer(self, shared_stats) -> Observer:
        """Returns an observer implementation that can be called on various methods."""
        pass
