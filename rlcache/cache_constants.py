from enum import Enum
from typing import Callable


class OperationType(Enum):
    New = 0
    Miss = 1
    Update = 2


class CacheInformation(object):
    """Class for keeping track of the environment information across all strategies."""

    def __init__(self, max_capacity: int, size_check_func: Callable[[], int]):
        self.invalidate = 0
        self.hit = 0
        self.miss = 0
        self.manual_evicts = 0
        self.max_capacity = max_capacity
        self._size_check_func = size_check_func

    @property
    def size(self):
        return self._size_check_func()

    def __str__(self):
        return "Invalidation: {}, " \
               "Hit: {}, " \
               "Miss: {}, " \
               "Manual Evicts: {}, " \
               "Size: {}/{}".format(self.invalidate, self.hit, self.miss,
                                    self.manual_evicts,
                                    self.size, self.max_capacity)
