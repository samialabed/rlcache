from abc import ABC
from typing import Dict

"""
TODO:
    - Write tests once things settled and you have a RL env
"""


class Storage(ABC):
    def __init__(self, capacity: int):
        self.capacity = capacity

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

    def __contains__(self, key):
        return self.contains(key)

    def __setitem__(self, key, value):
        return self.set(key, value)

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        return self.size()

    def __delitem__(self, key):
        return self.delete(key)

    def __iter__(self):
        raise NotImplementedError


class OutOfMemoryError(Exception):
    pass
