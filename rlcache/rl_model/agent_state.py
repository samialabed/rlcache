import copy
import dataclasses
from abc import ABC

import numpy as np


@dataclasses.dataclass
class AgentSystemState(ABC):
    def to_numpy(self) -> np.ndarray:
        return np.asarray(dataclasses.astuple(self))

    @classmethod
    def from_numpy(cls, encoded: np.ndarray):
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)
