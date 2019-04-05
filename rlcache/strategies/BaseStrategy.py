from typing import Dict

from rlcache.cache_constants import CacheInformation
from rlcache.observer import Observer, ObservationType


class BaseStrategy(Observer):
    def __init__(self, config: Dict[str, any], shared_stats: CacheInformation):
        self.shared_stats = shared_stats
        self.config = config

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        """ Observe the effect on the key. """
        pass
