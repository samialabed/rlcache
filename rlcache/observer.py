from abc import ABC
from enum import Enum
from typing import Dict, List

from rlcache.cache_constants import CacheInformation
from rlcache.utils.loggers import create_file_logger


class ObservationType(Enum):
    New = 0
    Hit = 1
    Miss = 2
    Invalidate = 3
    Expiration = 4
    EvictionPolicy = 5
    Write = 6
    EndOfEpisode = 7
    SetNotInCache = 8
    DeleteNotInCache = 9


class Observer(ABC):
    def __init__(self):
        self.supported_observations = {}

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        raise NotImplementedError


class ObserversOrchestrator(object):
    def __init__(self, observers: List[Observer], results_dir: str, cache_stats: CacheInformation):
        self.observers = observers
        self.episode_num = 0
        self.cache_stats = cache_stats
        self.evaluation_logger = create_file_logger(result_dir=results_dir, name='evaluation_logger')
        self.end_of_episode_logger = create_file_logger(result_dir=results_dir, name='end_of_episode_logger')

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any] = None):
        if observation_type != ObservationType.Write:
            self.evaluation_logger.info(f'{key},{observation_type.name},{self.episode_num}')

        for observer in self.observers:
            # if observation_type in {ObservationType.SetNotInCache, ObservationType.DeleteNotInCache}:
            #     # TODO refactor the strategies to handle this
            #     observation_type = ObservationType.Invalidate

            if observation_type in observer.supported_observations:
                observer.observe(key=key, observation_type=observation_type, info=info)

    def close(self):
        self.end_of_episode_logger.info(f'{self.episode_num},{self.cache_stats.to_log()}')
        self.episode_num += 1
