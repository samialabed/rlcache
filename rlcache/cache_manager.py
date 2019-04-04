from typing import Dict

from rlcache.backend import TTLCache
from rlcache.backend.base import Storage
from rlcache.cache_constants import CacheStatus
from rlcache.observers.observer import ObserverContainer, ObservationType
from rlcache.strategies.caching_strategies import caching_strategy_from_config
from rlcache.strategies.eviction_strategies import eviction_strategy_from_config
from rlcache.strategies.ttl_selection_strategies import ttl_strategy_from_config


class CacheManager(object):
    def __init__(self, config: Dict[str, any], cache: TTLCache, backend: Storage):

        self.cache_stats = {'cache_hit': 0, 'cache_miss': 0, 'cache_invalidation': 0}

        self.caching_strategy = caching_strategy_from_config(config['caching_strategy_settings'])
        self.eviction_strategy = eviction_strategy_from_config(config['eviction_strategy_settings'])
        self.ttl_strategy = ttl_strategy_from_config(config['ttl_strategy_settings'])
        self.cache = cache
        self.backend = backend
        # get the observers from our strategies
        self.observers = ObserverContainer([strategy for strategy in
                                            [self.caching_strategy, self.eviction_strategy, self.ttl_strategy]
                                            if strategy.observer(self.cache_stats) is not None])

    def get(self, key: str) -> Dict[str, any]:
        if self.cache.contains(key):
            self.cache_stats['cache_hit'] += 1
            values = self.cache.get(key)
        else:
            self.cache_stats['cache_miss'] += 1
            values = self.backend.get(key)
            self._set(key, values, CacheStatus.Miss)

        self.observers.observe(key, ObservationType.Read)
        return values

    def set(self, key: str, values: Dict[str, str]) -> None:
        if self.cache.contains(key):
            self.cache_stats['cache_invalidation'] += 1
            self.observers.observe(key, ObservationType.Update)
            self.cache.delete(key)
            status = CacheStatus.Invalidation
        else:
            self.observers.observe(key, ObservationType.Write)
            status = CacheStatus.New
        self._set(key, values, status)

    def delete(self, key: str) -> None:
        if self.cache.contains(key):
            self.cache_stats['cache_invalidation'] += 1
            self.observers.observe(key, ObservationType.Eviction)
            self.cache.delete(key)

    def stats(self) -> Dict[str, int]:
        return self.cache_stats

    def _set(self, key: str, values: Dict[str, any], cache_status: CacheStatus) -> None:
        if self.caching_strategy.should_cache(key, values, cache_status):
            ttl = self.ttl_strategy.estimate_ttl(key)
            if self.cache.is_full():
                self.eviction_strategy.trim_cache(self.cache)
            self.cache.set(key, values, ttl)
