from typing import Dict

from rlcache.cache_constants import OperationType
from rlcache.strategies.caching_strategies.caching_strategy_base import CachingStrategy


class OnReadWriteCacheStrategy(CachingStrategy):
    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        return True


class OnReadOnlyCacheStrategy(CachingStrategy):
    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        return operation_type == OperationType.Miss
