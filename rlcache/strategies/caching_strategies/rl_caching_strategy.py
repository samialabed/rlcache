from typing import Dict

from cache_constants import CacheStatus
from strategies.caching_strategies.caching_strategy_base import CachingStrategy


class RLCachingStrategy(CachingStrategy):
    def should_cache(self, key: str, values: Dict[str, str], invalidation: CacheStatus) -> bool:
        pass
