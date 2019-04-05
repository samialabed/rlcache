from typing import Dict

from rlcache.cache_constants import OperationType
from rlcache.strategies.BaseStrategy import BaseStrategy


class CachingStrategy(BaseStrategy):
    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        raise NotImplementedError
