from abc import ABC
from typing import Dict

from rlcache.observer import Observer


class BaseStrategy(Observer, ABC):
    def __init__(self, config: Dict[str, any], result_dir):
        self.result_dir = result_dir
        self.config = config

    def close(self):
        pass