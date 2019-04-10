import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import time

from rlcache.backend.base import Storage
from rlcache.observer import ObservationType

KeyType = str
InfoType = Dict[str, any]


@dataclass
class _ExpirationListEntry(object):
    eviction_time: time
    dirty_delete: bool

    key: str = field(compare=False)  # sort base only on eviction_time

    def copy(self):
        return _ExpirationListEntry(self.eviction_time, self.dirty_delete, self.key)


class TTLCacheV2(object):
    """
    Cache TTL wrapper on top of Storage objects that ensures objects are evicted after ttl is up.
    TODO refactor a bit more to allow generalising for other type of backends for now using python dict.
    TODO [LP] refactor to use Python Magic.
    """

    def __init__(self, memory: Storage):
        self.memory = memory
        self.expiration_time_list = []  # type: List[_ExpirationListEntry]
        self.evict_hook_func = []
        self.key_to_expiration_item = {}  # type: Dict[str, _ExpirationListEntry]

    def register_hook_func(self, hook: Callable[[KeyType, ObservationType, InfoType], None]):
        """register hooks that are called upon evictions."""
        self.evict_hook_func.append(hook)

    def delete(self, key: str):
        # do I want to raise key error if key not in cache?
        self.memory.delete(key)

    def size(self) -> int:
        return self.memory.size()

    def contains(self, key: str):
        self.expire(time.time())
        return self.memory.contains(key)

    def get(self, key: str, default=None):
        self.expire(time.time())
        return self.memory.get(key, default)

    def set(self, key: str, values: any, ttl: int) -> None:
        """
        :param key: key to set.
        :param values: values.
        :param ttl: time to live in seconds.
        """
        current_time = time.time()
        self.expire(current_time)
        self.memory.set(key, values)
        if key in self.key_to_expiration_item:  # update pointer
            stored_value = self.key_to_expiration_item[key]
            stored_value.eviction_time = current_time + ttl
            stored_value.dirty_delete = False
        else:
            expiration_entry = _ExpirationListEntry(eviction_time=current_time + ttl, key=key, dirty_delete=False)
            self.key_to_expiration_item[key] = expiration_entry
            heapq.heappush(self.expiration_time_list, expiration_entry)

    def expire(self, cur_time):
        for expiration_entry in self.expiration_time_list:
            eviction_time = expiration_entry.eviction_time
            key = expiration_entry.key
            stored_value = self.key_to_expiration_item[key]

            if cur_time < eviction_time:
                break
            # self.delete(key) leaves the expiration queue as is, as a trade-off between speed and memory. cleanup here.
            if not stored_value.dirty_delete and self.memory.contains(key):
                stored_values = self.memory.get(key)
                self.invoke_hooks(key, stored_values, eviction_time)
                # remove entries from cache and expiration queue
                self.memory.delete(key)
            heapq.heappop(self.expiration_time_list)

    def invoke_hooks(self, key, stored_values, eviction_time):
        for hook in self.evict_hook_func:
            hook(key, ObservationType.Expiration, {'value': stored_values, 'expire_at': eviction_time})
