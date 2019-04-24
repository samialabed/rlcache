import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import time
from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox, IntBox

from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObservationType
from rlcache.rl_model.converter import RLConverter
from rlcache.strategies.ttl_selection_strategies.ttl_strategy_base import TtlStrategy
from rlcache.utils.loggers import create_file_logger
from rlcache.utils.vocabulary import Vocabulary


@dataclass
class _ObservedExperience(object):
    __slots__ = ['state', 'agent_action', 'estimated_ttl', 'hits', 'observation_time', 'cache_utility']
    state: np.ndarray
    agent_action: np.ndarray
    estimated_ttl: int
    hits: int
    observation_time: time
    cache_utility: int


class RLTtlStrategy(TtlStrategy):
    """RL driven TTL estimation strategy."""

    def __init__(self, config: Dict[str, any], result_dir: str):
        super().__init__(config, result_dir)
        self.observation_seen = 0
        self.episode_reward = 0
        self.checkpoint_steps = config['checkpoint_steps']
        self.logger = logging.getLogger(__name__)
        self.reward_logger = create_file_logger(name='reward_logger', result_dir=self.result_dir)
        self.loss_logger = create_file_logger(name='loss_logger', result_dir=self.result_dir)
        self.observation_logger = create_file_logger(name='observation_logger', result_dir=self.result_dir)

        maximum_ttl = config['max_ttl']
        agent_config = config['agent_config']
        self.converter = TTLStrategyConverter(0)
        # action space: should cache: true or false
        # state space: [capacity (1), query key(1), query result set(num_indexes)]
        fields_in_state = 1 + 1 + 1 + 1 + 1  # key +  cache_status + hits + miss + initial ttl

        self.ttl_logger = create_file_logger(name='ttl_logger', result_dir=self.result_dir)
        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(shape=(fields_in_state,)),
                                     action_space=IntBox(low=1, high=maximum_ttl))

        self.observed_keys = {}  # type: Dict[str, _ObservedExperience]

    def observe(self, key: str, observation_type: ObservationType, *args, **kwargs):
        if observation_type == ObservationType.Write:
            # ignore writes (TODO refactor cache manager this shouldn't happen)
            return

        if key not in self.observed_keys:
            self.logger.debug(f'key: {key} is has not been observed in {__name__}, '
                              f'observation_type:{observation_type.name}')
            return  # haven't had to make a decision on it
        current_time = time.time()
        observed_experience = self.observed_keys[key]
        # record hits
        if observation_type == ObservationType.Hit:
            self.observed_keys[key].hits = observed_experience.hits + 1
        else:
            # Include updates, invalidation, and miss
            first_observation_time = observed_experience.observation_time
            estimated_ttl = observed_experience.estimated_ttl
            hits = observed_experience.hits
            real_ttl = current_time - first_observation_time
            # log the difference between the estimated ttl and real ttl
            self.ttl_logger.info(f'{observation_type.name},{key},{estimated_ttl},{real_ttl},{hits}')

            next_state = observed_experience.state.copy()
            next_state[2] = hits
            next_state[3] = 1 if observation_type == ObservationType.Miss else 0  # miss
            next_state[4] = real_ttl - estimated_ttl

            reward = self.converter.system_to_agent_reward(observed_experience, real_ttl)

            self.agent.observe(preprocessed_states=observed_experience.state,
                               actions=observed_experience.agent_action,
                               internals=[],
                               rewards=reward,
                               next_states=next_state,
                               terminals=False)
            self.episode_reward += reward
            self.reward_logger.info(f'{reward}')
            del self.observed_keys[key]
            # TODO replace with update scheduler
            loss = self.agent.update()
            if loss is not None:
                self.loss_logger.info(f'{loss[0]}')
            self.observation_seen += 1
            if self.observation_seen % self.checkpoint_steps == 0:
                self.logger.info(
                    f'Observation seen so far: {self.observation_seen}, reward so far: {self.episode_reward}')

    def estimate_ttl(self, key: str,
                     values: Dict[str, any],
                     operation_type: OperationType,
                     cache_information: CacheInformation) -> int:
        observation_time = time.time()
        state = self.converter.system_to_agent_state(key, values, operation_type, {})
        agent_action = self.agent.get_action(state)
        action = self.converter.agent_to_system_action(agent_action)
        cache_utility = cache_information.size / cache_information.max_capacity
        self.observed_keys[key] = _ObservedExperience(state=state,
                                                      agent_action=agent_action,
                                                      estimated_ttl=action,
                                                      observation_time=observation_time,
                                                      hits=0,
                                                      cache_utility=cache_utility)

        return action


class TTLStrategyConverter(RLConverter):
    def __init__(self, padding_size):
        self.padding_size = padding_size
        self.vocabulary = Vocabulary()
        self.logger = logging.getLogger(__name__)
        self.value_fields_vocabulary = Vocabulary(add_pad=False, add_unk=False)

    def agent_to_system_state(self, state: np.ndarray):
        key_encoded = state[0]
        operation_type_encoded = state[1]
        cache_hits = state[2]
        cache_miss = state[3]
        values_encoded = state[4:]

        key = self.vocabulary.get_name_for_id(key_encoded)
        operation_type = self.vocabulary.get_name_for_id(operation_type_encoded)
        values = self._recover_encoded_values(values_encoded)

        return key, values, operation_type, cache_hits, cache_miss

    def system_to_agent_state(self, key, values, operation_type, info: Dict[str, any]) -> np.ndarray:
        # TODO get this from observer
        key_encoded = self.vocabulary.add_or_get_id(key)
        operation_type_encoded = self.vocabulary.add_or_get_id(operation_type)
        hits = info.get('hits')
        miss = info.get('miss')
        initial_ttl = 0

        # TODO get capacity
        return np.concatenate([np.array([key_encoded, operation_type_encoded, hits, miss, initial_ttl])])

    def system_to_agent_action(self, should_cache: bool) -> np.ndarray:
        return np.ones(1, dtype='int32') if should_cache else np.zeros(1, dtype='int32')

    def agent_to_system_action(self, actions: np.ndarray, **kwargs) -> int:
        return actions.flatten().item()

    def system_to_agent_reward(self, experience: _ObservedExperience, real_ttl: time) -> int:
        # reward more utilisation of the cache capacity given more hits

        difference_in_ttl = real_ttl - experience.estimated_ttl
        # TODO consider doing (reward - difference in ttl) 
        reward = experience.hits * (1 + experience.cache_utility)
        self.logger.debug(f'{__name__}: Hits: {experience.hits}, ttl diff: {difference_in_ttl}, Reward: {reward}')
        return reward

    def _extract_and_encode_values(self, values: Dict[str, any]) -> np.ndarray:
        extracted = np.zeros(self.padding_size, dtype='int32')
        assert len(extracted) == len(values)
        for k, v in values.items():
            k_encoded = self.value_fields_vocabulary.add_or_get_id(k)
            extracted[k_encoded] = self.vocabulary.add_or_get_id(v)
        return extracted

    def _recover_encoded_values(self, values: np.ndarray) -> Dict[str, any]:
        output = {}
        for idx in range(len(values)):
            decoded_value = self.vocabulary.get_name_for_id(values[idx])
            decoded_key = self.value_fields_vocabulary.get_name_for_id(idx)
            output[decoded_key] = decoded_value
        assert len(output) == len(values)
        return output
