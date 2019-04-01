from abc import ABC
from typing import Dict, Union

from cache_constants import CacheStatus
from observers.observer import Observer


class CachingStrategy(ABC):
    def __init__(self, config: Dict[str, any]):
        self.config = config
        
    def should_cache(self, key: str, values: Dict[str, str], invalidation: CacheStatus) -> bool:
        raise NotImplementedError

    def observer(self) -> Union[Observer, None]:
        """Returns an observer implementation that can be called on various methods."""
        return None
