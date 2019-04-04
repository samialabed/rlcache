from typing import Dict

from rlcache.backend.base import Storage


# TODO is this basically a dict?
class InMemoryStorage(Storage):
    """Long lasting memory storage."""

    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.memory = {}

    def get(self, key, default=None):
        cached_entry = self.memory.get(key)
        if cached_entry:
            return cached_entry.value
        return default

    def set(self, key, value):
        if len(self.memory) > self.capacity:
            raise False  # todo shouldn't this raise exception
        self.memory[key] = value
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
