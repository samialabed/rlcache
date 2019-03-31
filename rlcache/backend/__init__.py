from abc import ABC
from typing import Dict

"""
TODO:
    - Write tests once things settled and you have a RL env
"""


class Storage(ABC):
    def __init__(self, capacity):
        self.capacity = capacity

    def get(self, key: str, default=None) -> Dict[str, any]:
        raise NotImplementedError

    def set(self, key: str, value, ttl: int = 500) -> bool:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def clear(self) -> bool:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def contains(self, key: str) -> bool:
        raise NotImplementedError

    def is_full(self) -> bool:
        raise NotImplementedError


class OutOfMemoryError(Exception):
    pass
