from typing import Dict

from cache_constants import CacheStatus
from strategies.caching_strategies.caching_strategy_base import CachingStrategy


class OnReadWriteCacheStrategy(CachingStrategy):
    def should_cache(self, key: str, values: Dict[str, str], cache_request_type: CacheStatus) -> bool:
        return True


class OnReadOnlyCacheStrategy(CachingStrategy):
    def should_cache(self, key: str, values: Dict[str, str], cache_request_type: CacheStatus) -> bool:
        return cache_request_type == CacheStatus.Miss
