from enum import Enum


class OperationType(Enum):
    New = 0
    Miss = 1
    Update = 2


class CacheInformation(object):
    """Class for keeping track of the environment information across all strategies."""

    def __init__(self, size_check_func):
        self.invalidate = 0
        self.hit = 0
        self.miss = 0
        self._size_check_func = size_check_func

    @property
    def size(self):
        return self._size_check_func()

    def __repr__(self):
        return "Invalidation: {}, Hit: {}, Miss: {}, Size: {}".format(self.invalidate, self.hit, self.miss, self.size)
