import collections

from time import monotonic

from rlcache.backend.base import Storage
from rlcache.observers.observer import Observer, ObservationType


class ExpiredKeyError(KeyError):
    pass


"""
TODO Take in storage and use it.
TODO refactor a bit more to allow generalising for other type of backends.
TODO [LP] refactor to use Python Magic.

"""


class TTLCache(object):
    """LRU Cache implementation with per-item time-to-live (TTL) value."""

    def __init__(self, memory: Storage, observer: Observer):
        self.observer = observer
        self.memory = memory
        self.__root = root = _Link()
        root.prev = root.next = root
        self.__links = collections.OrderedDict()
        self.__timer = _Timer(monotonic)

    def is_full(self):
        return self.memory.is_full()

    def clear(self):
        with self.__timer as time:
            self.expire(time)
            self.memory.clear()

    def set(self, key: str, value, ttl: int = 500) -> bool:
        with self.__timer as time:
            self.expire(time)
            self.memory.set(key, value)
        try:
            link = self.__getlink(key)
        except KeyError:
            self.__links[key] = link = _Link(key)
        else:
            link.unlink()
        link.expire = time + ttl
        link.next = root = self.__root
        link.prev = prev = root.prev
        prev.next = root.prev = link
        return True

    def delete(self, key: str) -> bool:
        self.memory.delete(key)
        link = self.__links.pop(key)
        link.unlink()
        return True

    def size(self) -> int:
        pass

    def contains(self, key: str) -> bool:
        try:
            link = self.__links[key]  # no reordering
        except KeyError:
            return False
        else:
            return not (link.expire < self.__timer())

    def get(self, key: str, default=None):
        try:
            link = self.__getlink(key)
        except KeyError:
            expired = False
        else:
            expired = link.expire < self.__timer()
        if expired:
            raise ExpiredKeyError(key)
        else:
            return self.memory.get(key, default)

    def __iter__(self):
        root = self.__root
        curr = root.next
        while curr is not root:
            # "freeze" time for iterator access
            with self.__timer as time:
                if not (curr.expire < time):
                    yield curr.key
            curr = curr.next

    def __len__(self):
        root = self.__root
        curr = root.next
        time = self.__timer()
        count = len(self.__links)
        while curr is not root and curr.expire < time:
            count -= 1
            curr = curr.next
        return count

    def __setstate__(self, state):
        self.__dict__.update(state)
        root = self.__root
        root.prev = root.next = root
        for link in sorted(self.__links.values(), key=lambda obj: obj.expire):
            link.next = root
            link.prev = prev = root.prev
            prev.next = root.prev = link
        self.expire(self.__timer())

    def __repr__(self):
        with self.__timer as time:
            self.expire(time)
            return self.memory.__repr__

    def expire(self, time=None):
        """Remove expired items from the cache."""
        if time is None:
            time = self.__timer()
        root = self.__root
        curr = root.next
        links = self.__links
        while curr is not root and curr.expire < time:
            self.memory.delete(curr.key)
            self.observer.observe(curr.key, ObservationType.Expiration, {'expire_at': curr.expire})
            del links[curr.key]
            next_link = curr.next
            curr.unlink()
            curr = next_link

    if hasattr(collections.OrderedDict, 'move_to_end'):
        def __getlink(self, key):
            value = self.__links[key]
            self.__links.move_to_end(key)
            return value
    else:
        def __getlink(self, key):
            value = self.__links.pop(key)
            self.__links[key] = value
            return value


class _Link(object):
    __slots__ = ('key', 'expire', 'next', 'prev')

    def __init__(self, key=None, expire=None):
        self.key = key
        self.expire = expire

    def __reduce__(self):
        return _Link, (self.key, self.expire)

    def unlink(self):
        next_link = self.next
        prev = self.prev
        prev.next = next_link
        next_link.prev = prev


class _Timer(object):

    def __init__(self, timer):
        self.__timer = timer
        self.__nesting = 0

    def __call__(self):
        if self.__nesting == 0:
            return self.__timer()
        else:
            return self.__time

    def __enter__(self):
        if self.__nesting == 0:
            self.__time = time = self.__timer()
        else:
            time = self.__time
        self.__nesting += 1
        return time

    def __exit__(self, *exc):
        self.__nesting -= 1

    def __reduce__(self):
        return _Timer, (self.__timer,)

    def __getattr__(self, name):
        return getattr(self.__timer, name)
