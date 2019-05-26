import logging

import numpy as np

from rlcache.observer import ObservationType
from rlcache.rl_model.converter import RLConverter
from rlcache.strategies.eviction_strategies.rl_eviction_state import EvictionAgentIncompleteExperienceEntry
from rlcache.utils.loggers import create_file_logger
from rlcache.utils.vocabulary import Vocabulary


class EvictionStrategyRLConverter(RLConverter):
    def __init__(self, result_dir: str):
        self.vocabulary = Vocabulary(add_pad=True, add_unk=False)
        self.logger = logging.getLogger(__name__)
        name = 'rl_eviction_strategy'
        self.performance_logger = create_file_logger(name=f'{name}_performance_logger', result_dir=result_dir)

    def system_to_agent_state(self, *args, **kwargs) -> np.ndarray:
        pass

    def system_to_agent_action(self, should_evict: bool) -> np.ndarray:
        return np.ones(1, dtype='int32') if should_evict else np.zeros(1, dtype='int32')

    def agent_to_system_action(self, actions: np.ndarray, **kwargs) -> bool:
        return (actions.flatten() == 1).item()

    def system_to_agent_reward(self,
                               stored_experience: EvictionAgentIncompleteExperienceEntry,
                               observation_type: ObservationType,
                               episode_num: int) -> int:
        # TODO consider using the hit count as scalar?

        should_evict = self.agent_to_system_action(stored_experience.agent_action)

        if observation_type == ObservationType.Expiration:
            if should_evict:
                # reward if should evict didn't observe any follow up miss
                self.performance_logger.info(f'{episode_num},TrueEvict')
                return 1
            # else didn't evict
            else:
                # reward for not evicting a key that received more hits.
                # or 0 if it didn't evict but also didn't get any hits

                gain_for_not_evicting = stored_experience.state.hit_count - stored_experience.starting_state.hit_count
                if gain_for_not_evicting > 0:
                    self.performance_logger.info(f'{episode_num},TrueMiss')
                else:
                    self.performance_logger.info(f'{episode_num},MissEvict')

                return gain_for_not_evicting

        if observation_type == ObservationType.Invalidate:
            # Set/Delete, remove entry from the cache.
            # reward an eviction followed by invalidation.
            if should_evict:
                self.performance_logger.info(f'{episode_num},TrueEvict')
                return 1
            else:
                # punish not evicting a key that got invalidated after.
                self.performance_logger.info(f'{episode_num},MissEvict')
                return -1

        if observation_type == ObservationType.Miss:
            assert should_evict, 'Observation miss even without making an eviction nor expire decision'

            self.performance_logger.info(f'{episode_num},FalseEvict')
            # Miss after making an eviction decision
            # Punish, a read after an eviction decision
            return -1

        # reward for evicting key that followed by causes that isn't covered up
        return 1
