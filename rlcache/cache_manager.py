from typing import Dict

from rlcache.backend.base import Storage, OutOfMemoryError
from rlcache.backend.ttl_cache_v2 import TTLCacheV2
from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObservationType
from rlcache.strategies.strategies_from_config import strategies_from_config

"""
    TODOs:
        - Add metric logging.
        - [LP] Refactor the observer architecture. There are a lot of duplication there.
"""


class CacheManager(object):

    def __init__(self, config: Dict[str, any], cache: TTLCacheV2, backend: Storage):
        self.cache = cache
        self.backend = backend
        self.cache_stats = CacheInformation(cache.capacity(), size_check_func=cache.size)
        self.caching_strategy, self.eviction_strategy, self.ttl_strategy = strategies_from_config(config)
        self.cache.register_hook_func(self.caching_strategy.observe)
        self.cache.register_hook_func(self.eviction_strategy.observe)

    def get(self, key: str) -> Dict[str, any]:
        if self.cache.contains(key):
            self.cache_stats.hit += 1
            self.caching_strategy.observe(key, ObservationType.Hit, {})
            self.eviction_strategy.observe(key, ObservationType.Read)
            values = self.cache.get(key)
        else:
            self.cache_stats.miss += 1
            values = self.backend.get(key)
            self.caching_strategy.observe(key, ObservationType.Miss, {})
            self._set(key, values, OperationType.Miss)

        return values

    def set(self, key: str, values: Dict[str, str]) -> None:
        if self.cache.contains(key):
            self.cache_stats.invalidate += 1
            self.cache.delete(key)  # ensure key isn't cached anymore
            status = OperationType.Update
            self.caching_strategy.observe(key, ObservationType.Invalidate, {})
            self.eviction_strategy.observe(key, ObservationType.Invalidate)
        else:
            status = OperationType.New
            self.caching_strategy.observe(key, ObservationType.InvalidateNotInCache, {})

        self._set(key, values, status)

    def delete(self, key: str) -> None:
        if self.cache.contains(key):
            self.cache_stats.invalidate += 1
            self.cache.delete(key)
            self.caching_strategy.observe(key, ObservationType.Invalidate, {})
            self.eviction_strategy.observe(key, ObservationType.Delete)
        else:
            self.caching_strategy.observe(key, ObservationType.InvalidateNotInCache, {})

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
                self.caching_strategy.observe(evicted_key, ObservationType.EvictionPolicy, {})
                self.cache_stats.manual_evicts += 1
                self.cache.set(key, values, ttl)
            # TODO monitor leftover TTL to judge incomplete experience for the TTL
            self.eviction_strategy.observe(key, ObservationType.Write)
        else:
            self.cache_stats.should_cache_false += 1

    def end_episode(self):
        self.cache.clear()
        self.caching_strategy.end_episode()
        self.ttl_strategy.end_episode()
        self.eviction_strategy.end_episode()

    def save_results(self):
        self.caching_strategy.save_results()
        self.ttl_strategy.save_results()
        self.eviction_strategy.save_results()
