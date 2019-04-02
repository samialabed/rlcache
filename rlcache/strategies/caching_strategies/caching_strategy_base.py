from abc import ABC
from typing import Dict

from rlcache.cache_constants import CacheStatus
from rlcache.observers.observer import Observer


class CachingStrategy(ABC):
    def __init__(self, config: Dict[str, any]):
        self.config = config

    def should_cache(self, key: str, values: Dict[str, str], cache_status: CacheStatus) -> bool:
        raise NotImplementedError

    def observer(self) -> Observer:
        """Returns an observer implementation that can be called on various methods."""
        pass
