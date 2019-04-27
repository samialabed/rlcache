from typing import Dict

from rlcache.cache_constants import OperationType
from rlcache.observer import ObservationType
from rlcache.strategies.caching_strategies.base_caching_strategy import CachingStrategy
from rlcache.utils.loggers import create_file_logger


class OnReadWriteCacheStrategy(CachingStrategy):
    def __init__(self, config: Dict[str, any], result_dir):
        super().__init__(config, result_dir)
        self.observation_logger = create_file_logger(name=f'{__name__}_observation_logger', result_dir=self.result_dir)

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        self.observation_logger.info(f'{key},{observation_type.name}')

    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        return True


class OnReadOnlyCacheStrategy(CachingStrategy):
    def __init__(self, config: Dict[str, any], result_dir):
        super().__init__(config, result_dir)
        self.observation_logger = create_file_logger(name=f'{__name__}_observation_logger', result_dir=self.result_dir)

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        self.observation_logger.info(f'{key},{observation_type.name}')

    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        return operation_type == OperationType.Miss
