import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from time import time


class OperationType(Enum):
    New = 0
    Miss = 1
    Update = 2


@dataclass
class MonitoringEntry(object):
    cache_hit: bool
    cache_miss: bool
    should_cache: bool
    key: str
    timestamp: time = time()

    def to_csv(self) -> str:
        """Convert information to csv compatible format."""
        return f'{self.timestamp},{self.key},{self.cache_hit},{self.cache_miss},{self.should_cache}'


class CacheInformation(object):
    """Class for keeping track of the environment information across all strategies."""

    def __init__(self, max_capacity: int, size_check_func: Callable[[], int]):
        self.invalidate = 0
        self.hit = 0
        self.miss = 0
        self.manual_evicts = 0
        self.should_cache_true = 0
        self.should_cache_false = 0
        self.max_capacity = max_capacity
        self._size_check_func = size_check_func

    @property
    def size(self):
        return self._size_check_func()

    def __str__(self):
        return json.dumps({"Invalidation": self.invalidate,
                           "Hits": self.hit,
                           "Misses": self.miss,
                           "Hit rate (%)": (self.hit / (self.miss + self.hit)) * 100,
                           "Should cache": self.should_cache_true,
                           "Shouldn't cache": self.should_cache_false,
                           "Should cache ratio (%)": (self.should_cache_true / (
                                   self.should_cache_false + self.should_cache_true)) * 100,
                           "Manual Evicts": self.manual_evicts,
                           "Size": self.size,
                           "capacity": self.max_capacity
                           })
