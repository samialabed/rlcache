from abc import ABC
from enum import Enum

from typing import List, Dict

from rlcache.backend import TTLCache


class ObservationType(Enum):
    Hit = 1
    Miss = 2
    Invalidate = 3
    Expiration = 4  # Signal terminal


class Observer(ABC):
    def __init__(self, shared_stats: Dict[str, int]):
        self.shared_stats = shared_stats

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        raise NotImplementedError


class ObserversOrchestrator(object):
    def __init__(self, expiring_memory: TTLCache, observers: List[Observer]):
        self.expiring_memory = expiring_memory
        self.observers = observers

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any] = None):
        for observer in self.observers:
            observer.observe(key, observation_type, info)
