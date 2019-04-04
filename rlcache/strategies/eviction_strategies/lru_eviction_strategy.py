from collections import OrderedDict
from typing import Dict

from rlcache.backend.base import Storage
from rlcache.observers.observer import Observer, ObservationType
from rlcache.strategies.eviction_strategies.eviction_strategy_base import EvictionStrategy


class LRUObserver(Observer):
    def __init__(self, shared_stats: Dict[str, int]):
        super().__init__(shared_stats)
        self.lru = OrderedDict()

    def observe(self, key: str, observation_type: ObservationType, **kwargs):
        try:
            self.lru.pop(key)
        except KeyError:
            pass  # item not observed in cache
        if not observation_type.Eviction:
            self.lru[key] = observation_type


class LRUEvictionStrategy(EvictionStrategy):
    def __init__(self, config: Dict[str, any]):
        super().__init__(config)

    def observer(self, shared_stats):
        if self._observer:
            return self._observer
        self._observer = LRUObserver(shared_stats)
        return self._observer

    def trim_cache(self, cache: Storage):
        eviction_key = self._observer.lru.popitem(last=False)[0]
        cache.delete(eviction_key)
