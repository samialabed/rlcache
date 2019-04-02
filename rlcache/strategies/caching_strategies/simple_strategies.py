from typing import Dict

from rlcache.cache_constants import CacheStatus
from rlcache.strategies.caching_strategies.caching_strategy_base import CachingStrategy


class OnReadWriteCacheStrategy(CachingStrategy):
    def should_cache(self, key: str, values: Dict[str, str], cache_status: CacheStatus) -> bool:
        return True


class OnReadOnlyCacheStrategy(CachingStrategy):
    def should_cache(self, key: str, values: Dict[str, str], cache_status: CacheStatus) -> bool:
        return cache_status == CacheStatus.Miss
