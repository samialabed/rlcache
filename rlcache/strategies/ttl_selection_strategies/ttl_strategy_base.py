from abc import ABC

from rlcache.strategies.base_strategy import BaseStrategy


class TtlStrategy(BaseStrategy, ABC):
    def estimate_ttl(self, key: str) -> int:
        """Estimates a time to live based on the key."""
        raise NotImplementedError
