from observers.observer import ObservationType
from ttl_selection_strategies.ttl_strategy_base import TtlStrategy


class FixedTtlStrategy(TtlStrategy):
    """Fixed strategy that returns a preconfigured ttl."""

    def observer(self, key: str, observation_type: ObservationType):
        pass

    def __init__(self, ttl: int):
        self.ttl = ttl

    def estimate_ttl(self, key) -> int:
        return self.ttl
