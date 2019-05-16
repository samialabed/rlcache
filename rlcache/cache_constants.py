import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from time import time


class OperationType(Enum):
    New = 0
    Miss = 1
    Update = 2
    EndEpisode = 3


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

    @property
    def cache_utility(self):
        if self.max_capacity is None:
            return 1.0  # unbounded, cache utility is maxed

        return self.size / self.max_capacity

    @property
    def should_cache_ratio(self) -> float:
        return self.should_cache_true / max((self.should_cache_false + self.should_cache_true), 1)

    @property
    def hit_ratio(self) -> float:
        return self.hit / max(self.miss + self.hit, 1)

    def to_log(self) -> str:
        return "{},{},{},{},{},{},{},{},{}".format(self.invalidate,
                                                   self.hit,
                                                   self.miss,
                                                   self.hit_ratio * 100,
                                                   self.should_cache_true,
                                                   self.should_cache_false,
                                                   self.should_cache_ratio * 100,
                                                   self.manual_evicts,
                                                   self.cache_utility)

    def close(self):
        self.invalidate = 0
        self.hit = 0
        self.miss = 0
        self.manual_evicts = 0
        self.should_cache_true = 0
        self.should_cache_false = 0

    def __str__(self):
        return json.dumps({"Invalidation": self.invalidate,
                           "Hits": self.hit,
                           "Misses": self.miss,
                           "Hit rate (%)": self.hit_ratio * 100,
                           "Should cache": self.should_cache_true,
                           "Shouldn't cache": self.should_cache_false,
                           "Should cache ratio (%)": self.should_cache_ratio * 100,
                           "Manual Evicts": self.manual_evicts,
                           "Size": self.size,
                           "capacity": self.max_capacity
                           })
