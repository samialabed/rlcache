import time
from collections import namedtuple

from backend import Cache, OutOfMemoryError

CachedEntry = namedtuple('CacheEntry', ['value', 'expiration'])


class InMemoryCache(Cache):

    def __init__(self, capacity):
        super().__init__(capacity)
        self.memory = {}
        self.expiration = {}

    def get(self, key, default=None):
        if time.time() > self.memory[key].expiration:
            self.delete(key)
        cached_entry = self.memory.get(key)
        if cached_entry:
            return cached_entry.value
        return default

    def set(self, key, value, ttl=500):
        if len(self.memory) > self.capacity:
            raise OutOfMemoryError
        expiration_ttl = time.time() + ttl
        self.memory[key] = CachedEntry(value, expiration_ttl)
        return True

    def delete(self, key):
        if key in self.memory:
            del self.memory[key]
        return True

    def clear(self):
        del self.memory
        self.memory = {}
        return True

    def size(self):
        return len(self.memory)

    def contains(self, key):
        return key in self.memory
