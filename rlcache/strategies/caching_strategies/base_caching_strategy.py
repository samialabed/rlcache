from abc import ABC
from typing import Dict

from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObservationType
from rlcache.strategies.base_strategy import BaseStrategy


class CachingStrategy(BaseStrategy, ABC):
    def __init__(self, config: Dict[str, any], result_dir: str, cache_stats: CacheInformation):
        super().__init__(config, result_dir, cache_stats)
        self.supported_observations = {ObservationType.Hit,
                                       ObservationType.Miss,
                                       ObservationType.Invalidate,
                                       ObservationType.Expiration,
                                       ObservationType.EvictionPolicy,
                                       ObservationType.SetNotInCache,
                                       ObservationType.DeleteNotInCache}

    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        raise NotImplementedError
