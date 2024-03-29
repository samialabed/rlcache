from rlcache.backend.base import Storage, OutOfMemoryError


class InMemoryStorage(Storage):
    """Long lasting memory storage."""

    def __init__(self, capacity: int = None):
        super().__init__(capacity)
        self.memory = {}

    def get(self, key, default=None):
        return self.memory.get(key, default)

    def set(self, key, value):
        # key not in memory: update won't increase the size
        if key not in self.memory and self.is_full():
            raise OutOfMemoryError
        self.memory[key] = value

    def items(self):
        return self.memory.items()

    def delete(self, key):
        if key in self.memory:
            del self.memory[key]

    def clear(self):
        self.memory.clear()

    def keys(self):
        return self.memory.keys()

    def size(self):
        return len(self.memory)

    def contains(self, key):
        return key in self.memory

    def __iter__(self):
        return self.memory.__iter__()

    def __repr__(self):
        return self.memory.__repr__()
