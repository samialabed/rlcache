import copy
import dataclasses
from abc import ABC

import numpy as np


@dataclasses.dataclass
class AgentSystemState(ABC):
    # TODO simplify the class as well as let it use python DTO until to_numpy and from_numpy
    def to_numpy(self) -> np.ndarray:
        return np.array(dataclasses.astuple(self))

    @classmethod
    def from_numpy(cls, encoded: np.ndarray):
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)
