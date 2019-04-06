from abc import ABC
from typing import Dict

from rlcache.cache_constants import CacheInformation
from rlcache.observer import Observer


class BaseStrategy(Observer, ABC):
    def __init__(self, config: Dict[str, any], shared_stats: CacheInformation):
        self.shared_stats = shared_stats
        self.config = config
