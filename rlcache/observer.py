from abc import ABC
from enum import Enum
from typing import Dict


class ObservationType(Enum):
    Hit = 1
    Miss = 2
    Invalidate = 3
    Expiration = 4
    EvictionPolicy = 5
    Write = 6


class Observer(ABC):
    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        raise NotImplementedError
