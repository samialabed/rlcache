from rlcache.backend.base import Storage, OutOfMemoryError


# TODO is this basically a dict?

class InMemoryStorage(Storage):
    """Long lasting memory storage."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.memory = {}

    def get(self, key, default=None):
        cached_entry = self.memory.get(key)
        if cached_entry:
            return cached_entry
        return default

    def set(self, key, value):
        if key not in self.memory and len(self.memory) + 1 > self.capacity:
            raise OutOfMemoryError
        self.memory[key] = value

    def delete(self, key):
        if key in self.memory:
            del self.memory[key]

    def clear(self):
        self.memory.clear()

    def size(self):
        return len(self.memory)

    def contains(self, key):
        return key in self.memory
