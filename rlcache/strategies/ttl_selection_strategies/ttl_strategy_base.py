from abc import ABC
from typing import Dict

from observers.observer import Observer


class TtlStrategy(ABC):
    def __init__(self, config: Dict[str, any]):
        self.config = config

    def estimate_ttl(self, key: str) -> int:
        """Estimates a time to live based on the key."""
        raise NotImplementedError

    def observer(self) -> Observer:
        """Returns an observer implementation that can be called on various methods."""
        pass
