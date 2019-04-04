from abc import ABC
from enum import Enum

from typing import List, Dict


class ObservationType(Enum):
    Read = 1
    Write = 2
    Update = 3  # TODO do I need update?
    Eviction = 4
    Expiration = 5  # Signal terminal


class Observer(ABC):
    def __init__(self, shared_stats: Dict[str, int]):
        self.shared_stats = shared_stats

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        raise NotImplementedError


class ObserverContainer(object):
    def __init__(self, observers: List[Observer]):
        self.observers = observers

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any] = None):
        for observer in self.observers:
            observer.observe(key, observation_type, info)
