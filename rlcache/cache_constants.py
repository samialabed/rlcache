from enum import Enum


# TODO I don't like this
class CacheStatus(Enum):
    Hit = 0
    Miss = 1
    Invalidation = 2
    New = 3
