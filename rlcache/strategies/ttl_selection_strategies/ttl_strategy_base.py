from abc import ABC
from typing import Dict

from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.strategies.base_strategy import BaseStrategy


class TtlStrategy(BaseStrategy, ABC):
    def estimate_ttl(self, key: str,
                     values: Dict[str, any],
                     operation_type: OperationType,
                     cache_information: CacheInformation) -> int:
        """Estimates a time to live based on the key."""
        raise NotImplementedError
