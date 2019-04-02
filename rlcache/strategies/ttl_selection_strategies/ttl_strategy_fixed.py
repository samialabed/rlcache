from typing import Dict

from strategies.ttl_selection_strategies.ttl_strategy_base import TtlStrategy


class FixedTtlStrategy(TtlStrategy):
    """Fixed strategy that returns a preconfigured ttl."""

    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.ttl = self.config['ttl']

    def estimate_ttl(self, key) -> int:
        return self.ttl
