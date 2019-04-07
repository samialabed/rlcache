from collections import OrderedDict
from typing import Dict

from rlcache.backend import TTLCache
from rlcache.cache_constants import CacheInformation
from rlcache.observer import ObservationType
from rlcache.strategies.eviction_strategies.eviction_strategy_base import EvictionStrategy


class LRUEvictionStrategy(EvictionStrategy):
    def __init__(self, config: Dict[str, any], shared_stats: CacheInformation):
        super().__init__(config, shared_stats)
        self.lru = OrderedDict()

    def observe(self, key: str, observation_type: ObservationType, **kwargs):
        try:
            self.lru.pop(key)
        except KeyError:
            pass  # item not observed in cache
        # refresh lru if hit
        if observation_type in [ObservationType.Read, ObservationType.Write]:
            self.lru[key] = observation_type

    def trim_cache(self, cache: TTLCache):
        eviction_key = self.lru.popitem(last=False)[0]
        cache.delete(eviction_key)
        return eviction_key
