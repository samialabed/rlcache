from abc import ABC

from rlcache.backend.base import Storage
from rlcache.strategies.base_strategy import BaseStrategy


class EvictionStrategy(BaseStrategy, ABC):
    # TODO should this take a cache as well?

    def trim_cache(self, cache: Storage):
        """ Called when cache is full, finds an item to evict from the cache and evict it."""
        raise NotImplementedError
