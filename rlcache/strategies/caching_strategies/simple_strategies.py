from typing import Dict

from rlcache.cache_constants import OperationType
from rlcache.observer import ObservationType
from rlcache.strategies.caching_strategies.caching_strategy_base import CachingStrategy


class OnReadWriteCacheStrategy(CachingStrategy):
    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        pass

    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        return True


class OnReadOnlyCacheStrategy(CachingStrategy):
    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        pass

    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        return operation_type == OperationType.Miss
