from abc import ABC
from typing import Dict

from rlcache.backend.base import Storage
from rlcache.cache_constants import CacheInformation
from rlcache.observer import ObservationType
from rlcache.strategies.base_strategy import BaseStrategy


class EvictionStrategy(BaseStrategy, ABC):
    def __init__(self, config: Dict[str, any], result_dir: str, cache_stats: CacheInformation):
        super().__init__(config, result_dir, cache_stats)
        self.supported_observations = {ObservationType.Hit,
                                       ObservationType.Miss,
                                       ObservationType.Invalidate,
                                       ObservationType.Expiration,
                                       ObservationType.Write}

    def trim_cache(self, cache: Storage):
        """ Called when cache is full, finds an item to evict from the cache and evict it."""
        raise NotImplementedError
