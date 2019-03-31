from abc import ABC
from typing import Dict

from cache_constants import CacheStatus
from observers.observer import Observer


class CachingStrategy(ABC):
    def should_cache(self, key: str, values: Dict[str, str], invalidation: CacheStatus) -> bool:
        raise NotImplementedError

    def observer(self) -> Observer:
        """Returns an observer implementation that can be called on various methods."""
        pass
