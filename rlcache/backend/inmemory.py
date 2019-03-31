from collections import namedtuple

import time

from backend.base import Storage

CachedEntry = namedtuple('CacheEntry', ['value', 'expiration'])


class InMemoryStorage(Storage):

    def __init__(self, capacity=1024):
        super().__init__(capacity)
        self.memory = {}

    def get(self, key, default=None):
        if time.time() > self.memory[key].expiration:
            self.delete(key)
        cached_entry = self.memory.get(key)
        if cached_entry:
            return cached_entry.value
        return default

    def set(self, key, value, ttl=500):
        if len(self.memory) > self.capacity:
            raise False
        expiration_ttl = time.time() + ttl
        self.memory[key] = CachedEntry(value, expiration_ttl)
        return True

    def delete(self, key):
        if key in self.memory:
            del self.memory[key]
            return True
        return False

    def clear(self):
        self.memory.clear()
        return True

    def size(self):
        return len(self.memory)

    def contains(self, key):
        return key in self.memory
