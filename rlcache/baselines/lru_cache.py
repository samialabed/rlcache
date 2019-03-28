from backend import Cache


# TODO naive implementation of LRU, this does full scan at set
# Maintain doubly linked list for LRU
# What about the distributed cache case?

class LRUCache:
    def __init__(self, cache: Cache):
        self.tm = 0
        self.cache = cache
        self.lru = {}

    def get(self, key):
        self.lru[key] = self.tm
        self.tm += 1
        return self.cache.get(key)

    def set(self, key, value):
        if len(self.cache.size()) >= self.cache.capacity:
            # find the LRU entry
            old_key = min(self.lru.keys(), key=lambda k: self.lru[k])
            self.cache.delete(old_key)
            self.lru.pop(old_key)
        self.cache.set(key, value)
        self.lru[key] = self.tm
        self.tm += 1
