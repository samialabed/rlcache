from collections import Counter
from typing import Dict

from backend.base import Storage
from cache_constants import CacheStatus
from caching_strategies import caching_strategy_from_config
from eviction_strategies import eviction_strategy_from_config
from observers.observer import ObserverContainer, ObservationType
from ttl_selection_strategies import ttl_strategy_from_config


class CacheManager(object):
    def __init__(self, config: Dict[str, any], cache: Storage, backend: Storage):

        self.caching_strategy = caching_strategy_from_config(config['caching_strategy_settings'])
        self.eviction_strategy = eviction_strategy_from_config(config['eviction_strategy_settings'])
        self.ttl_strategy = ttl_strategy_from_config(config['ttl_strategy_settings'])
        self.cache = cache
        self.backend = backend
        # TODO move metrics to own class?
        self.cache_metrics = Counter({'cache_hits': 0,
                                      'cache_miss': 0,
                                      'cache_invalidation': 0,
                                      'cache_size': 0})
        # get the observers from our strategies
        self.observers = ObserverContainer([strategy.observer() for strategy in
                                            [self.caching_strategy, self.eviction_strategy, self.ttl_strategy]
                                            if strategy.observer() is not None])

    def get(self, key: str) -> Dict[str, any]:
        self.observers.observe(key, ObservationType.Read)
        if self.cache.contains(key):
            self.cache_metrics['cache_hits'] += 1
            return self.cache.get(key)
        else:
            self.cache_metrics['cache_miss'] += 1
            values = self.backend.get(key)
            self._set(key, values, CacheStatus.Miss)
            return values

    def set(self, key: str, values: Dict[str, str]):
        if self.cache.contains(key):
            self.observers.observe(key, ObservationType.Update)
            self.cache_metrics['cache_invalidation'] += 1
            self.cache.delete(key)
            status = CacheStatus.Invalidation
        else:
            self.observers.observe(key, ObservationType.Write)
            status = CacheStatus.New
        self._set(key, values, status)

    def delete(self, key):
        if self.cache.contains(key):
            self.observers.observe(key, ObservationType.Eviction)
            self.cache.delete(key)

    def stats(self) -> Dict[str, str]:
        return self.cache_metrics

    def _set(self, key, values, cache_status):
        if self.caching_strategy.should_cache(key, values, cache_status):
            ttl = self.ttl_strategy.estimate_ttl(key)
            if self.cache.is_full():
                self.eviction_strategy.trim_cache(self.cache)
            self.cache.set(key, values, ttl)
