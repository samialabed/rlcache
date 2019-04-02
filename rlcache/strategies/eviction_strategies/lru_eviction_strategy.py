from collections import OrderedDict
from typing import Dict

from backend.base import Storage
from observers.observer import Observer, ObservationType
from strategies.eviction_strategies.eviction_strategy_base import EvictionStrategy


class LRUObserver(Observer):
    def __init__(self):
        self.lru = OrderedDict()

    def observe(self, key: str, observation_type: ObservationType):
        try:
            self.lru.pop(key)
        except KeyError:
            pass  # item not observed in cache
        if not observation_type.Eviction:
            self.lru[key] = observation_type


class LRUEvictionStrategy(EvictionStrategy):
    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self._observer = LRUObserver()

    def observer(self):
        return self._observer

    def trim_cache(self, cache: Storage):
        eviction_key = self._observer.lru.popitem(last=False)[0]
        cache.delete(eviction_key)