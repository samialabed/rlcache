from typing import Dict

from rlcache.backend.base import Storage, OutOfMemoryError
from rlcache.backend.ttl_cache import TTLCache
from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObservationType
from rlcache.strategies.strategies_from_config import strategies_from_config
from rlcache.utils.loggers import create_file_logger


class CacheManager(object):

    def __init__(self, config: Dict[str, any], cache: TTLCache, backend: Storage, result_dir: str):
        self.cache = cache
        self.backend = backend
        self.cache_stats = CacheInformation(cache.capacity(), size_check_func=cache.size)
        self.caching_strategy, self.eviction_strategy, self.ttl_strategy = strategies_from_config(config,
                                                                                                  result_dir,
                                                                                                  self.cache_stats)
        self.cache.expired_entry_callback(self.caching_strategy.observe)
        self.cache.expired_entry_callback(self.eviction_strategy.observe)
        self.cache.expired_entry_callback(self.ttl_strategy.observe)
        self.cache_hit_logger = create_file_logger(result_dir=result_dir, name=f'cache_hit_logger')

    def get(self, key: str) -> Dict[str, any]:
        if self.cache.contains(key):
            self.cache_stats.hit += 1
            self.cache_hit_logger.info(f'{key},{True},{False}')
            self.caching_strategy.observe(key, ObservationType.Hit, {})
            self.eviction_strategy.observe(key, ObservationType.Hit, {})
            self.ttl_strategy.observe(key, ObservationType.Hit, {})
            values = self.cache.get(key)
        else:
            self.cache_hit_logger.info(f'{key},{False},{True}')
            self.cache_stats.miss += 1
            values = self.backend.get(key)
            self.caching_strategy.observe(key, ObservationType.Miss, {})
            self.eviction_strategy.observe(key, ObservationType.Miss, {})
            self.ttl_strategy.observe(key, ObservationType.Miss, {})
            self._set(key, values, OperationType.Miss)

        return values

    def set(self, key: str, values: Dict[str, str]) -> None:
        self.ttl_strategy.observe(key, ObservationType.Invalidate, {})
        self.caching_strategy.observe(key, ObservationType.Invalidate, {})
        self.eviction_strategy.observe(key, ObservationType.Invalidate, {})

        if self.cache.contains(key):
            self.cache_stats.invalidate += 1
            self.cache.delete(key)  # ensure key isn't cached anymore
            status = OperationType.Update
        else:
            status = OperationType.New

        self._set(key, values, status)

    def delete(self, key: str) -> None:
        self.ttl_strategy.observe(key, ObservationType.Invalidate, {})
        self.caching_strategy.observe(key, ObservationType.Invalidate, {})
        self.eviction_strategy.observe(key, ObservationType.Invalidate, {})

        if self.cache.contains(key):
            self.cache_stats.invalidate += 1
            self.cache.delete(key)

    def stats(self) -> str:
        return str(self.cache_stats)

    def close(self):
        self.ttl_strategy.close()
        self.caching_strategy.close()
        self.eviction_strategy.close()

    def _set(self, key: str, values: Dict[str, any], operation_type: OperationType) -> None:
        ttl = self.ttl_strategy.estimate_ttl(key, values, operation_type)
        should_cache = self.caching_strategy.should_cache(key, values, ttl, operation_type)
        if should_cache:
            self.cache_stats.should_cache_true += 1
            try:
                self.cache.set(key, values, ttl)
            except OutOfMemoryError:
                evicted_keys = self.eviction_strategy.trim_cache(self.cache)
                for evicted_key in evicted_keys:
                    self.caching_strategy.observe(evicted_key, ObservationType.EvictionPolicy, {})
                    self.ttl_strategy.observe(evicted_key, ObservationType.EvictionPolicy, {})
                    self.cache_stats.manual_evicts += 1

                # TODO this should be in a loop
                self.cache.set(key, values, ttl)

            self.eviction_strategy.observe(key, ObservationType.Write, {'ttl': ttl, })
        else:
            self.cache_stats.should_cache_false += 1
