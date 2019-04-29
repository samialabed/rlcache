import logging
from typing import Dict

import numpy as np

from rlcache.observer import ObservationType
from rlcache.rl_model.converter import RLConverter
from rlcache.strategies.caching_strategies.rl_caching_state import CachingAgentIncompleteExperienceEntry
from rlcache.utils.vocabulary import Vocabulary


class CachingStrategyRLConverter(RLConverter):
    def __init__(self):
        self.vocabulary = Vocabulary()
        self.logger = logging.getLogger(__name__)

    def system_to_agent_state(self, key, values, operation_type, info: Dict[str, any]) -> np.ndarray:
        pass

    def system_to_agent_action(self, should_cache: bool) -> np.ndarray:
        return np.ones(1, dtype='int32') if should_cache else np.zeros(1, dtype='int32')

    def agent_to_system_action(self, actions: np.ndarray, **kwargs) -> bool:
        return (actions.flatten() == 1).item()

    def system_to_agent_reward(self, experience: CachingAgentIncompleteExperienceEntry) -> int:
        should_cache = self.agent_to_system_action(experience.agent_action)
        if should_cache:
            # 1- Should cache -> multiple hits -> expires/invalidates: Complete experience, reward
            # 2- Should cache -> no hits -> expires/invalidates: complete experience, punish
            reward = experience.state.hit_count
        else:
            # 4- Shouldn't cache -> hit(s): complete experience, punish
            # 3- shouldn't cache -> no hits or miss: reward with 1
            assert experience.state.step_code != ObservationType.Hit, \
                'Logical conflict: terminal state for should not cache, with a hit stepcode. '

            if experience.state.step_code == ObservationType.Miss:
                reward = -1
            else:
                reward = 1
        return reward
