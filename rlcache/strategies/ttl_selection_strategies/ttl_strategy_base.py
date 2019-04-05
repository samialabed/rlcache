from rlcache.strategies.BaseStrategy import BaseStrategy


class TtlStrategy(BaseStrategy):
    def estimate_ttl(self, key: str) -> int:
        """Estimates a time to live based on the key."""
        raise NotImplementedError
