from typing import Dict

from rlcache.observer import ObservationType
from rlcache.strategies.ttl_selection_strategies.ttl_strategy_base import TtlStrategy


class FixedTtlStrategy(TtlStrategy):
    """Fixed strategy that returns a preconfigured ttl."""

    def end_episode(self, *args, **kwargs):
        pass

    def save_results(self):
        pass

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        pass

    def __init__(self, config: Dict[str, any], results_dir: str):
        super().__init__(config, results_dir)
        self.ttl = self.config['ttl']

    def estimate_ttl(self, key) -> int:
        return self.ttl
