import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox, IntBox

from rlcache.backend import TTLCache, InMemoryStorage
from rlcache.cache_constants import OperationType
from rlcache.observer import ObservationType
from rlcache.rl_model.converter import RLConverter
from rlcache.strategies.caching_strategies.caching_strategy_base import CachingStrategy
from rlcache.utils.loggers import create_file_logger
from rlcache.utils.vocabulary import Vocabulary


@dataclass
class _IncompleteExperienceEntry(object):
    __slots__ = ['state', 'agent_action', 'ttl', 'hits', 'miss']
    state: np.ndarray
    agent_action: np.ndarray
    ttl: int
    hits: int
    miss: int

    def __repr__(self):
        return f'state: {self.state}, agent_action: {self.agent_action}, ' \
            f'ttl: {self.ttl}, hits: {self.hits}, miss: {self.miss}'


class RLCachingStrategy(CachingStrategy):

    def __init__(self, config: Dict[str, any], result_dir):
        super().__init__(config, result_dir)
        # evaluation specific variables
        self.observation_seen = 0
        self.episode_reward = 0
        self.checkpoint_steps = config['checkpoint_steps']

        self.logger = logging.getLogger(__name__)
        self.reward_logger = create_file_logger(name='reward_logger', result_dir=self.result_dir)
        self.loss_logger = create_file_logger(name='loss_logger', result_dir=self.result_dir)
        self.observation_logger = create_file_logger(name='observation_logger', result_dir=self.result_dir)

        agent_config = config['agent_config']
        self.converter = CachingStrategyRLConverter(0)
        # action space: should cache: true or false
        # state space: [capacity (1), query key(1), query result set(num_indexes)]
        fields_in_state = 1 + 1 + 1 + 1  # key +  cache_status + hits + miss
        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(shape=(fields_in_state,)),
                                     action_space=IntBox(2))
        self._incomplete_experiences = TTLCache(InMemoryStorage())
        self._incomplete_experiences.register_hook_func(self._observe_expired_incomplete_experience)

    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        # TODO what about the case of a cache key that exist already in the incomplete exp?
        assert self._incomplete_experiences.get(key) is None, \
            "should_cache is assumed to be first call and key shouldn't be in the cache"

        # TODO Add ttl to state
        state = self.converter.system_to_agent_state(key, values, operation_type, {'hits': 0, 'miss': 0})
        agent_action = self.agent.get_action(state)
        incomplete_experience_entry = _IncompleteExperienceEntry(state=state,
                                                                 agent_action=agent_action,
                                                                 ttl=ttl,
                                                                 hits=0,
                                                                 miss=0)
        action = self.converter.agent_to_system_action(agent_action)
        self._incomplete_experiences.set(key,
                                         incomplete_experience_entry,
                                         incomplete_experience_entry.ttl)
        return action

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        self.observation_logger.info(f'{key},{observation_type.name}')
        # TODO include stats/capacity information in the info dict
        experience = self._incomplete_experiences.get(key)  # type: _IncompleteExperienceEntry
        if experience is None:
            return  # if I haven't had to make a decision on this, ignore it.

        self._reward_experience(key, experience, observation_type)

    def _observe_expired_incomplete_experience(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        """Observe decisions taken that hasn't been observed by main cache. e.g. don't cache -> ttl up -> no miss"""
        assert observation_type == ObservationType.Expiration
        experience = info['value']
        self._reward_experience(key, experience, observation_type)

    def _reward_experience(self, key: str, experience: _IncompleteExperienceEntry, observation_type: ObservationType):
        cache_hit = False
        cache_miss = False
        if observation_type == ObservationType.Hit:
            cache_hit = True
            self.logger.debug(f'Key hit: {key}')
        else:
            if observation_type == ObservationType.Miss:
                cache_miss = True
                # experience.miss += 1
            self.logger.debug(f'Key: {key} is in terminal state because: {str(observation_type)}')
            self._incomplete_experiences.delete(key)

        next_state = experience.state.copy()  # type: np.ndarray
        next_state[2] = experience.hits + 1 if cache_hit else experience.hits
        next_state[3] = experience.miss + 1 if cache_miss else experience.miss
        reward = self.converter.system_to_agent_reward(experience)

        self.agent.observe(experience.state,
                           experience.agent_action,
                           [],
                           reward,
                           next_state,
                           terminals=False)
        self.episode_reward += reward

        self.reward_logger.info(f'{reward}')

        # TODO replace with update scheduler
        loss = self.agent.update()
        if loss is not None:
            self.loss_logger.info(f'{loss[0]}')
        self.observation_seen += 1
        if self.observation_seen % self.checkpoint_steps == 0:
            self.logger.info(f'Observation seen so far: {self.observation_seen}, reward so far: {self.episode_reward}')


class CachingStrategyRLConverter(RLConverter):
    def __init__(self, padding_size):
        self.padding_size = padding_size
        self.vocabulary = Vocabulary()
        self.logger = logging.getLogger(__name__)
        self.value_fields_vocabulary = Vocabulary(add_pad=False, add_unk=False)
