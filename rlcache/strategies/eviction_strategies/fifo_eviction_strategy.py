import logging
from collections import OrderedDict
from typing import Dict, List

import time

from rlcache.backend import TTLCache, InMemoryStorage
from rlcache.cache_constants import CacheInformation
from rlcache.observer import ObservationType
from rlcache.strategies.eviction_strategies.base_eviction_strategy import EvictionStrategy
from rlcache.utils.loggers import create_file_logger


class FIFOEvictionStrategy(EvictionStrategy):
    def __init__(self, config: Dict[str, any], result_dir: str, cache_stats: CacheInformation):
        super().__init__(config, result_dir, cache_stats)
        self.fifo = OrderedDict()
        self.logger = logging.getLogger(__name__)
        self.renewable_ops = {ObservationType.Hit, ObservationType.Write}
        name = 'fifo_eviction_strategy'
        self.performance_logger = create_file_logger(name=f'{name}_performance_logger', result_dir=result_dir)

        self._incomplete_experiences = TTLCache(InMemoryStorage())
        self._incomplete_experiences.expired_entry_callback(self._observe_expired_incomplete_experience)

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        if observation_type == ObservationType.Write:
            ttl = info['ttl']
            observation_time = time.time()
            self.fifo[key] = {'ttl': ttl, 'observation_time': observation_time}
        elif observation_type in {ObservationType.Expiration, ObservationType.Invalidate}:
            self.logger.debug(f"Key {key} expired")
            del self.fifo[key]

        action_taken = self._incomplete_experiences.get(key)
        if action_taken is not None:
            if observation_type == ObservationType.Invalidate:
                # eviction followed by invalidation.
                self.performance_logger.info(f'{self.episode_num},TrueEvict')
            elif observation_type == ObservationType.Miss:
                self.performance_logger.info(f'{self.episode_num},FalseEvict')
                # Miss after making an eviction decision
            self._incomplete_experiences.delete(key)

    def _observe_expired_incomplete_experience(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        self.performance_logger.info(f'{self.episode_num},TrueEvict')

    def trim_cache(self, cache: TTLCache) -> List[str]:
        eviction_item = self.fifo.popitem(last=False)
        eviction_key = eviction_item[0]
        eviction_value = eviction_item[1]
        assert cache.contains(eviction_key), f'Key: {eviction_key} is in FIFO but not in cache.'
        decision_time = time.time()
        ttl_left = (eviction_value['observation_time'] + eviction_value['ttl']) - decision_time
        self._incomplete_experiences.set(eviction_key, 'evict', ttl_left)
        cache.delete(eviction_key)
        return [eviction_key]
