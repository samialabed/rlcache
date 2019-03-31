from enum import Enum


class CacheStatus(Enum):
    Hit = 0
    Miss = 1
    Invalidation = 2
    New = 3
