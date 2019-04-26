from typing import Dict

import time

from rlcache.observer import ObservationType
from rlcache.strategies.ttl_selection_strategies.ttl_strategy_base import TtlStrategy
from rlcache.utils.loggers import create_file_logger


class FixedTtlStrategy(TtlStrategy):
    """Fixed strategy that returns a preconfigured ttl."""

    def __init__(self, config: Dict[str, any], result_dir: str):
        super().__init__(config, result_dir)
        self.ttl = self.config['ttl']
        self.ttl_logger = create_file_logger(name=f'{__name__}_ttl_logger', result_dir=self.result_dir)
        self.observed_keys = {}

    def observe(self, key: str, observation_type: ObservationType, *args, **kwargs):
        current_time = time.time()
        if key not in self.observed_keys:
            if observation_type == ObservationType.Write:
                self.observed_keys[key] = {'observation_time': current_time, 'estimated_ttl': self.ttl, 'hits': 0}
            return

        stored_values = self.observed_keys[key]
        if observation_type == ObservationType.Hit:
            self.observed_keys[key]['hits'] = stored_values['hits'] + 1
        else:
            # Include updates
            first_observation_time = stored_values['observation_time']
            estimated_ttl = stored_values['estimated_ttl']
            hits = stored_values['hits']
            real_ttl = current_time - first_observation_time
            # log the difference between the estimated ttl and real ttl
            self.ttl_logger.info(f'{observation_type.name},{key},{estimated_ttl},{real_ttl},{hits}')
            del self.observed_keys[key]
            if observation_type == ObservationType.Write:
                self.observed_keys[key] = {'observation_time': current_time, 'estimated_ttl': self.ttl, 'hits': 0}

    def estimate_ttl(self, key, *args, **kwargs) -> int:
        return self.ttl
