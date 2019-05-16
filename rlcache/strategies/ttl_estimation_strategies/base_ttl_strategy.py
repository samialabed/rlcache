from abc import ABC
from typing import Dict

from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObservationType
from rlcache.strategies.base_strategy import BaseStrategy


class TtlStrategy(BaseStrategy, ABC):
    def __init__(self, config: Dict[str, any], result_dir: str, cache_stats: CacheInformation):
        super().__init__(config, result_dir, cache_stats)
        self.supported_observations = {ObservationType.Hit,
                                       ObservationType.Miss,
                                       ObservationType.Invalidate,
                                       ObservationType.Expiration,
                                       ObservationType.EvictionPolicy}

    def estimate_ttl(self, key: str,
                     values: Dict[str, any],
                     operation_type: OperationType) -> int:
        """Estimates a time to live based on the key."""
        raise NotImplementedError
