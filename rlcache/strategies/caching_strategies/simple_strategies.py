from typing import Dict

from rlcache.cache_constants import OperationType
from rlcache.observer import ObservationType
from rlcache.strategies.caching_strategies.base_caching_strategy import CachingStrategy
from rlcache.utils.loggers import create_file_logger


class OnReadWriteCacheStrategy(CachingStrategy):
    def __init__(self, config: Dict[str, any], result_dir):
        super().__init__(config, result_dir)
        name = 'read_write_caching_strategy_'
        self.observation_logger = create_file_logger(name=f'{name}_observation_logger', result_dir=self.result_dir)
        self.entry_hits_logger = create_file_logger(name=f'{name}_entry_hits_logger', result_dir=self.result_dir)
        self.observed_entries = {}

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        self.observation_logger.info(f'{key},{observation_type.name}')
        if key not in self.observed_entries:
            return

        if observation_type == ObservationType.Hit:
            self.observed_entries[key] += 1
        else:
            hits = self.observed_entries[key]
            self.entry_hits_logger.info(f'{key},{hits}')
            del self.observed_entries[key]

    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        self.observed_entries[key] = 0  # hits
        return True


class OnReadOnlyCacheStrategy(CachingStrategy):
    def __init__(self, config: Dict[str, any], result_dir):
        super().__init__(config, result_dir)
        name = 'read_only_caching_strategy_'
        self.observation_logger = create_file_logger(name=f'{name}_observation_logger', result_dir=self.result_dir)
        self.entry_hits_logger = create_file_logger(name=f'{name}_entry_hits_logger', result_dir=self.result_dir)
        self.observed_entries = {}

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        self.observation_logger.info(f'{key},{observation_type.name}')
        if key not in self.observed_entries:
            return

        if observation_type == ObservationType.Hit:
            self.observed_entries[key] += 1
        else:
            hits = self.observed_entries[key]
            self.entry_hits_logger.info(f'{key},{hits}')
            del self.observed_entries[key]

    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        should = operation_type == OperationType.Miss
        if should:
            self.observed_entries[key] = 0  # hits
        return should
