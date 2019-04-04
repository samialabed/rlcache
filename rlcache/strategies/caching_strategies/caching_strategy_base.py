from abc import ABC
from typing import Dict

from rlcache.cache_constants import OperationType
from rlcache.observers.observer import Observer


class CachingStrategy(ABC):
    def __init__(self, config: Dict[str, any]):
        self.config = config

    def should_cache(self, key: str, values: Dict[str, str], cache_status: OperationType) -> bool:
        raise NotImplementedError

    def observer(self, shared_stats) -> Observer:
        """Returns an observer implementation that can be called on various methods.
        :param shared_stats:
        """
        pass
