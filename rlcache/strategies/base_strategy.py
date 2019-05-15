from abc import ABC
from typing import Dict

from rlcache.cache_constants import CacheInformation
from rlcache.observer import Observer


class BaseStrategy(Observer, ABC):
    def __init__(self, config: Dict[str, any], result_dir: str, cache_stats: CacheInformation):
        self.cache_stats = cache_stats
        self.result_dir = result_dir
        self.config = config

    def close(self):
        pass
