from collections import namedtuple
from typing import Dict

import time

from rlcache.backend.base import Storage

# BLOCKED FOR TTL IMPLEMENTATION
# TODO make this expires https://stackoverflow.com/questions/3927166/automatically-expiring-variable
# TODO Make permanent memory and expirying one
MemoryEntry = namedtuple('MemoryEntry', ['value', 'expiration'])


class InMemoryStorage(Storage):

    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
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
        self.memory[key] = MemoryEntry(value, expiration_ttl)
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
