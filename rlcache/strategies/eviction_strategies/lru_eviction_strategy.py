import logging
from collections import OrderedDict
from typing import Dict

from rlcache.backend import TTLCacheV2
from rlcache.observer import ObservationType
from rlcache.strategies.eviction_strategies.eviction_strategy_base import EvictionStrategy


class LRUEvictionStrategy(EvictionStrategy):
    def __init__(self, config: Dict[str, any], results_dir: str):
        super().__init__(config, results_dir)
        self.lru = OrderedDict()
        self.logger = logging.getLogger(__name__)
        self.renewable_ops = {ObservationType.Read, ObservationType.Write}

    def observe(self, key: str, observation_type: ObservationType, *args, **kwargs):
        try:
            self.lru.pop(key)
        except KeyError:
            self.logger.debug("Key: {} not in LRU monitor. Current LRU size: {}".format(key, len(self.lru)))
            pass  # item not observed in cache before.
        # add/refresh lru if hit
        if observation_type in self.renewable_ops:
            self.lru[key] = observation_type
        if observation_type in {ObservationType.Expiration, ObservationType.Invalidate}:
            self.logger.debug("Key {} expired, deleting".format(key))
            assert key not in self.lru, "Expired key should have been deleted."

    def trim_cache(self, cache: TTLCacheV2):
        eviction_key = self.lru.popitem(last=False)[0]
        assert cache.contains(eviction_key), "Key: {} is in LRU but not in cache.".format(eviction_key)

        cache.delete(eviction_key)
        return eviction_key

    def end_episode(self, *args, **kwargs):
        self.lru.clear()

    def save_results(self):
        pass
