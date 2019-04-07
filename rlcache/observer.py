from abc import ABC
from enum import Enum
from typing import List, Dict


class ObservationType(Enum):
    Hit = 1
    Miss = 2  # Caching strategy signal terminal (also eviction strategy to punish misses if it was chosen as evict)
    Invalidate = 3  # Caching strategy signal terminal
    Expiration = 4  # Caching strategy signal terminal
    InvalidateNotInCache = 5
    EvictionPolicy = 6
    Read = 7  # Eviction strategy
    Write = 8  # Eviction strategy
    Delete = 9  # Eviction strategy


class Observer(ABC):
    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        raise NotImplementedError


class ObserversOrchestrator(Observer):
    def __init__(self, observers: List[Observer]):
        self.observers = observers

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any] = None):
        for observer in self.observers:
            observer.observe(key=key, observation_type=observation_type, info=info)
