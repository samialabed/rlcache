from typing import Dict

from rlcache.backend import TTLCache
from rlcache.backend.base import Storage
from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observers.observer import ObserversOrchestrator, ObservationType
from rlcache.strategies.caching_strategies import caching_strategy_from_config
from rlcache.strategies.eviction_strategies import eviction_strategy_from_config
from rlcache.strategies.ttl_selection_strategies import ttl_strategy_from_config


class CacheManager(object):
    def __init__(self, config: Dict[str, any], cache_config: Dict[str, any], backend: Storage):
        self.caching_strategy = caching_strategy_from_config(config['caching_strategy_settings'])
        self.eviction_strategy = eviction_strategy_from_config(config['eviction_strategy_settings'])
        self.ttl_strategy = ttl_strategy_from_config(config['ttl_strategy_settings'])
        self.cache = cache
        self.backend = backend
        self.cache_stats = CacheInformation(size_check_func=cache.size)

        # memory to maintain decisions
        # TODO FIXME get the observers from our strategies
        self.observers = ObserversOrchestrator([strategy for strategy in
                                                [self.caching_strategy, self.eviction_strategy, self.ttl_strategy]
                                                if strategy.observer(self.cache_stats) is not None])

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
        else:
            status = OperationType.New

        self.observers.observe(key, ObservationType.Invalidate)
        self._set(key, values, status)

    def delete(self, key: str) -> None:
        if self.cache.contains(key):
            self.cache_stats.invalidate += 1
            self.observers.observe(key, ObservationType.Eviction)
            self.cache.delete(key)

    def stats(self) -> CacheInformation:
        return self.cache_stats

    def _set(self, key: str, values: Dict[str, any], operation_type: OperationType) -> None:
        if self.caching_strategy.should_cache(key, values, operation_type):
            ttl = self.ttl_strategy.estimate_ttl(key)
            if self.cache.is_full():
                self.eviction_strategy.trim_cache(self.cache)
            self.cache.set(key, values, ttl)
        # else TODO let ttl_strategy attempt to learn anyway
