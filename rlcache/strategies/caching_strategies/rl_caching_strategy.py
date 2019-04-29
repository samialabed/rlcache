import logging
from typing import Dict

import time

from rlcache.backend import TTLCache, InMemoryStorage
from rlcache.cache_constants import OperationType
from rlcache.observer import ObservationType
from rlcache.strategies.caching_strategies.base_caching_strategy import CachingStrategy
from rlcache.strategies.caching_strategies.rl_caching_state import CachingAgentIncompleteExperienceEntry, \
    CachingAgentSystemState
from rlcache.strategies.caching_strategies.rl_caching_state_converter import CachingStrategyRLConverter
from rlcache.utils.loggers import create_file_logger
from rlcache.utils.vocabulary import Vocabulary
from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox, IntBox


class RLCachingStrategy(CachingStrategy):

    def __init__(self, config: Dict[str, any], result_dir):
        super().__init__(config, result_dir)
        # evaluation specific variables
        self.observation_seen = 0
        self.episode_reward = 0
        self.checkpoint_steps = config['checkpoint_steps']

        self._incomplete_experiences = TTLCache(InMemoryStorage())
        self._incomplete_experiences.expired_entry_callback(self._observe_expired_incomplete_experience)

        self.experimental_reward = config.get('experimental_reward', False)
        agent_config = config['agent_config']
        self.converter = CachingStrategyRLConverter()
        # action space: should cache: true or false
        # state space: [capacity (1), query key(1), query result set(num_indexes)]
        fields_in_state = len(CachingAgentSystemState.__slots__)
        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(shape=(fields_in_state,)),
                                     action_space=IntBox(2))

        self.logger = logging.getLogger(__name__)
        name = 'rl_caching_strategy'
        self.reward_logger = create_file_logger(name=f'{name}_reward_logger', result_dir=self.result_dir)
        self.loss_logger = create_file_logger(name=f'{name}_loss_logger', result_dir=self.result_dir)
        self.observation_logger = create_file_logger(name=f'{name}_observation_logger', result_dir=self.result_dir)

        self.key_vocab = Vocabulary()

    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        # TODO what about the case of a cache key that exist already in the incomplete exp?
        assert self._incomplete_experiences.get(key) is None, \
            "should_cache is assumed to be first call and key shouldn't be in the cache"
        observation_time = time.time()

        encoded_key = self.key_vocab.add_or_get_id(key)
        state = CachingAgentSystemState(encoded_key=encoded_key,
                                        ttl=ttl,
                                        hit_count=0,
                                        step_code=0,
                                        operation_type=operation_type.value)

        agent_action = self.agent.get_action(state.to_numpy())
        incomplete_experience_entry = CachingAgentIncompleteExperienceEntry(state=state,
                                                                            agent_action=agent_action,
                                                                            starting_state=state.copy(),
                                                                            observation_time=observation_time)

        action = self.converter.agent_to_system_action(agent_action)
        self._incomplete_experiences.set(key, incomplete_experience_entry, ttl)

        return action

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        self.observation_logger.info(f'{key},{observation_type.name}')
        # TODO include stats/capacity information in the info dict
        experience = self._incomplete_experiences.get(key)  # type: CachingAgentIncompleteExperienceEntry
        if experience is None:
            return  # if I haven't had to make a decision on this, ignore it.

        if observation_type == ObservationType.Hit:
            experience.state.hit_count += 1

        else:
            self._reward_experience(key, experience, observation_type)

        self.observation_seen += 1
        if self.observation_seen % self.checkpoint_steps == 0:
            self.logger.info(f'Observation seen so far: {self.observation_seen}, reward so far: {self.episode_reward}')

    def _observe_expired_incomplete_experience(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        """Observe decisions taken that hasn't been observed by main cache. e.g. don't cache -> ttl up -> no miss"""
        assert observation_type == ObservationType.Expiration
        experience = info['value']
        self._reward_experience(key, experience, observation_type)

    def _reward_experience(self,
                           key: str,
                           experience: CachingAgentIncompleteExperienceEntry,
                           observation_type: ObservationType):
        state = experience.state
        state.step_code = observation_type.value

        self._incomplete_experiences.delete(key)

        reward = self.converter.system_to_agent_reward(experience)
        if self.experimental_reward:
            # TODO add cache utility to state and reward
            pass

        self.agent.observe(preprocessed_states=experience.starting_state.to_numpy(),
                           actions=experience.agent_action,
                           internals=[],
                           rewards=reward,
                           next_states=experience.state.to_numpy(),
                           terminals=False)

        self.episode_reward += reward
        self.reward_logger.info(f'{reward}')
        self.logger.debug(f'Key: {key} is in terminal state because: {str(observation_type)}')
        loss = self.agent.update()
        if loss is not None:
            self.loss_logger.info(f'{loss[0]}')
