from abc import ABC
from typing import Dict

from rlcache.cache_constants import OperationType
from rlcache.strategies.base_strategy import BaseStrategy


class CachingStrategy(BaseStrategy, ABC):
    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        raise NotImplementedError
