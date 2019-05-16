from abc import ABC
from typing import Dict


class Storage(ABC):
    def __init__(self, capacity: int):
        self.capacity = capacity

    def is_full(self) -> bool:
        """ Determine if can put more items. If capacity isn't specified then assumed no space limit."""
        if self.capacity is None:
            return False
        return self.size() + 1 > self.capacity

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

    def items(self):
        raise NotImplementedError

    def keys(self):
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
