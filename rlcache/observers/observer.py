from abc import ABC
from enum import Enum

from typing import List


class ObservationType(Enum):
    Read = 1
    Write = 2
    Update = 3  # TODO do I need update?
    Eviction = 4


class Observer(ABC):
    def observe(self, key: str, observation_type: ObservationType):
        raise NotImplementedError


class ObserverContainer(object):
    def __init__(self, observers: List[Observer]):
        self.observers = observers

    def observe(self, key: str, observation_type: ObservationType):
        for observer in self.observers:
            observer.observe(key, observation_type)
