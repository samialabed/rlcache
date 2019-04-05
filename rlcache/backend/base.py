from abc import ABC
from typing import Dict

"""
TODO:
    - Write tests once things settled and you have a RL env
"""


class Storage(ABC):
    def get(self, key: str, default=None) -> Dict[str, any]:
        raise NotImplementedError

    def set(self, key: str, value) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def contains(self, key: str) -> bool:
        raise NotImplementedError


class OutOfMemoryError(Exception):
    pass
