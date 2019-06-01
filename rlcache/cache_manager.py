from typing import Dict

from rlcache.backend.base import Storage
from rlcache.backend.ttl_cache import TTLCache
from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObservationType, ObserversOrchestrator
from rlcache.strategies.strategies_from_config import strategies_from_config


class CacheManager(object):

    def __init__(self, config: Dict[str, any], cache: TTLCache, backend: Storage, result_dir: str):
        self.cache = cache
        self.backend = backend
        self.cache_stats = CacheInformation(cache.capacity(), size_check_func=cache.size)
        self.caching_strategy, self.eviction_strategy, self.ttl_strategy = strategies_from_config(config,
                                                                                                  result_dir,
                                                                                                  self.cache_stats)
        if 'multi_strategy_settings' in config:
            # any of the strategies work for multi-strategy
            self.observer_orchestrator = ObserversOrchestrator([self.caching_strategy], result_dir, self.cache_stats)
            self.multi_strategy = True
        else:
            self.observer_orchestrator = ObserversOrchestrator([self.caching_strategy,
                                                                self.eviction_strategy,
                                                                self.ttl_strategy], result_dir, self.cache_stats)
            self.multi_strategy = False

        self.cache.expired_entry_callback(self.observer_orchestrator.observe)

    def get(self, key: str) -> Dict[str, any]:
        if self.cache.contains(key):
            self.cache_stats.hit += 1
            self.observer_orchestrator.observe(key, ObservationType.Hit, {})
            values = self.cache.get(key)
        else:
            self.cache_stats.miss += 1
            values = self.backend.get(key)
            self.observer_orchestrator.observe(key, ObservationType.Miss, {})
            self._set(key, values, OperationType.Miss)

        return values

    def set(self, key: str, values: Dict[str, str]) -> None:
        if self.cache.contains(key):
            self.observer_orchestrator.observe(key, ObservationType.Invalidate, {})
            self.cache_stats.invalidate += 1
            self.cache.delete(key)  # ensure key isn't cached anymore
            status = OperationType.Update
        else:
            self.observer_orchestrator.observe(key, ObservationType.SetNotInCache, {})
            status = OperationType.New

        self._set(key, values, status)

    def delete(self, key: str) -> None:
        if self.cache.contains(key):
            self.observer_orchestrator.observe(key, ObservationType.Invalidate, {})
            self.cache_stats.invalidate += 1
            self.cache.delete(key)
        else:
            self.observer_orchestrator.observe(key, ObservationType.Invalidate, {})

    def stats(self) -> str:
        return str(self.cache_stats)

    def close(self):
        if self.multi_strategy:
            self.ttl_strategy.close()
        else:
            self.ttl_strategy.close()
            self.caching_strategy.close()
            self.eviction_strategy.close()

        self.observer_orchestrator.close()
        self.cache_stats.close()

    def _set(self, key: str, values: Dict[str, any], operation_type: OperationType) -> None:
        ttl = self.ttl_strategy.estimate_ttl(key, values, operation_type)
        should_cache = self.caching_strategy.should_cache(key, values, ttl, operation_type)
        if should_cache:
            self.cache_stats.should_cache_true += 1
            while self.cache.is_full():
                evicted_keys = self.eviction_strategy.trim_cache(self.cache)
                for evicted_key in evicted_keys:
                    self.observer_orchestrator.observe(evicted_key, ObservationType.EvictionPolicy, {})
                    self.cache_stats.manual_evicts += 1

            self.cache.set(key, values, ttl)
            self.observer_orchestrator.observe(key, ObservationType.Write, {'ttl': ttl})
        else:
            self.cache_stats.should_cache_false += 1
