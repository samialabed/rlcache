from abc import ABC

"""
TODO:
    - Incorporate a max-size to force more ejection
    
"""


class Cache(ABC):
    def __init__(self, capacity):
        self.capacity = capacity

    def get(self, key, default=None):
        raise NotImplementedError

    def set(self, key, value, ttl=500):
        raise NotImplementedError

    def delete(self, key):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def size(self):
        raise NotImplementedError

    def contains(self, key):
        raise NotImplementedError


class OutOfMemoryError(Exception):
    pass
