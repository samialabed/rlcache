from typing import Dict

from rlcache.backend import TTLCache
from rlcache.backend.base import Storage, OutOfMemoryError
from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObserversOrchestrator, ObservationType
from rlcache.strategies.caching_strategies import caching_strategy_from_config
from rlcache.strategies.eviction_strategies import eviction_strategy_from_config
from rlcache.strategies.ttl_selection_strategies import ttl_strategy_from_config


# TODO remove self.cache_stats
class CacheManager(object):
    def __init__(self, config: Dict[str, any], cache: TTLCache, backend: Storage):
        self.cache = cache
        self.backend = backend
        self.cache_stats = CacheInformation(cache.capacity(), size_check_func=cache.size)
        self.caching_strategy = caching_strategy_from_config(config['caching_strategy_settings'], self.cache_stats)
        self.eviction_strategy = eviction_strategy_from_config(config['eviction_strategy_settings'], self.cache_stats)
        self.ttl_strategy = ttl_strategy_from_config(config['ttl_strategy_settings'], self.cache_stats)
        # memory to maintain decisions
        self.observers = ObserversOrchestrator(observers=[self.caching_strategy,
                                                          self.eviction_strategy,
                                                          self.ttl_strategy])
        self.cache.register_hook_func(self.observers.observe)

    def get(self, key: str) -> Dict[str, any]:
        if self.cache.contains(key):
            self.cache_stats.hit += 1
            self.observers.observe(key, ObservationType.Hit)
            values = self.cache.get(key)
        else:
            self.cache_stats.miss += 1
            values = self.backend.get(key)
            self.observers.observe(key, ObservationType.Miss)
            self._set(key, values, OperationType.Miss)

        return values

    def set(self, key: str, values: Dict[str, str]) -> None:
        if self.cache.contains(key):
            self.cache_stats.invalidate += 1
            self.cache.delete(key)  # ensure key isn't cached anymore
            status = OperationType.Update
            self.observers.observe(key, ObservationType.Invalidate)
        else:
            status = OperationType.New
            self.observers.observe(key, ObservationType.InvalidateNotInCache)

        self._set(key, values, status)

    def delete(self, key: str) -> None:
        if self.cache.contains(key):
            self.cache_stats.invalidate += 1
            self.cache.delete(key)
            self.observers.observe(key, ObservationType.Invalidate)
        else:
            self.observers.observe(key, ObservationType.InvalidateNotInCache)

    def stats(self) -> str:
        return str(self.cache_stats)

    def _set(self, key: str, values: Dict[str, any], operation_type: OperationType) -> None:
        ttl = self.ttl_strategy.estimate_ttl(key)
        should_cache = self.caching_strategy.should_cache(key, values, ttl, operation_type)
        if should_cache:
            self.cache_stats.should_cache_true += 1
            try:
                self.cache.set(key, values, ttl)
            except OutOfMemoryError:
                evicted_key = self.eviction_strategy.trim_cache(self.cache)
                self.observers.observe(evicted_key, ObservationType.EvictionPolicy)
                self.cache_stats.manual_evicts += 1
                self.cache.set(key, values, ttl)
            self.eviction_strategy.monitor_key(key)
        else:
            self.cache_stats.should_cache_false += 1