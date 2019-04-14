import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox, IntBox
from time import time

from rlcache.backend import TTLCacheV2, InMemoryStorage
from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObservationType
from rlcache.rl_model.converter import RLConverter
from rlcache.strategies.caching_strategies.caching_strategy_base import CachingStrategy
from rlcache.utils.vocabulary import Vocabulary

"""

How to know a non-cached entry, that didn't get any more reads or write, is a good entry to not cache. 

    TODOs:
        - [LP] this should be refactored once second agent is developed and common functionality is taken out
        - Result set is encoded as one id, Future work can use Language model to extract key information and decide
        based on that whether to cache or not.
    
    * initially - I don't need to pass the result set because YCSB generates rubbish, until I build my own workload
    I can remove that.
"""


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


@dataclass
class _MonitoringEntry(object):
    __slots__ = ['timestamp', 'key', 'cache_hit', 'cache_miss', 'episode']
    key: str
    cache_hit: bool
    cache_miss: bool
    timestamp: time
    episode: int

    def __repr__(self) -> str:
        return f'timestamp: {self.timestamp}, key: {self.key}, ' \
            f'cache_hit: {self.cache_hit}, cache_miss: {self.cache_miss}, episode: {self.episode}'

    def __str__(self) -> str:
        return f'{self.timestamp},{self.key},{self.cache_hit},{self.cache_miss},{self.episode}'


class RLCachingStrategy(CachingStrategy):

    def __init__(self, config: Dict[str, any], results_dir):
        super().__init__(config, results_dir)
        self.episode_rewards = []
        agent_config = config['agent_config']
        self.logger = logging.getLogger(__name__)
        self.converter = CachingStrategyRLConverter(0)
        flattened_num_cols = 1 + 1 + 1 + 1  # key +  cache_status + hits + miss
        # action space: should cache: true or false
        # state space: [capacity (1), query key(1), query result set(num_indexes)]
        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(shape=(flattened_num_cols,)),
                                     action_space=IntBox(2))
        self._incomplete_experience_storage = TTLCacheV2(InMemoryStorage(capacity=config['observer_storage_capacity']))
        self._incomplete_experience_storage.register_hook_func(self._observe_expired_incomplete_experience)

        # evaluation specific variables
        self.episode_reward = 0  # type: int
        self.episode_stats = []  # type: List[str]
        self.episode_num = 0  # type: int
        self.losses = []  # type: List[Tuple[int, float]]

    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        # TODO what about the case of a cache key that exist already in the incomplete exp?
        assert self._incomplete_experience_storage.get(key) is None, \
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
        self._incomplete_experience_storage.set(key,
                                                incomplete_experience_entry,
                                                incomplete_experience_entry.ttl)
        return action

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        # TODO needs a serious refactoring this is ugly and hacky
        # TODO include stats/capacity information in the info dict
        experience = self._incomplete_experience_storage.get(key)  # type: _IncompleteExperienceEntry
        if experience is None:
            return  # if I haven't had to make a decision on this, ignore it.

        self._reward_experience(key, experience, observation_type)

    def _observe_expired_incomplete_experience(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        assert observation_type == ObservationType.Expiration
        experience = info['value']
        self._reward_experience(key, experience, observation_type)

    def _reward_experience(self, key: str, experience: _IncompleteExperienceEntry, observation_type: ObservationType):
        if observation_type == ObservationType.Hit:
            experience.hits += 1
            self.episode_stats.append(str(_MonitoringEntry(key=key,
                                                           cache_hit=True,
                                                           cache_miss=False,
                                                           timestamp=time(),
                                                           episode=self.episode_num)))
            self.logger.debug(f'Key hit: {key}')
        else:
            if observation_type == ObservationType.Miss:
                experience.miss += 1
                self.episode_stats.append(str(_MonitoringEntry(key=key,
                                                               cache_hit=False,
                                                               cache_miss=True,
                                                               timestamp=time(),
                                                               episode=self.episode_num)))
            self.logger.debug(f'Key: {key} is in terminal state because: {str(observation_type)}')
            self._incomplete_experience_storage.delete(key)

        next_state = experience.state.copy()  # type: np.ndarray
        next_state[2] = experience.hits
        next_state[3] = experience.miss
        reward = self.converter.system_to_agent_reward(experience)
        self.agent.observe(experience.state,
                           experience.agent_action,
                           [],
                           reward,
                           next_state,
                           terminals=False)
        self.episode_reward += reward
        # TODO replace with update scheduler
        loss = self.agent.update()[0]
        if loss is not None:
            self.losses.append((self.episode_num, loss))

    def end_episode(self, cache_information: CacheInformation):
        # Include final system configs
        hits = cache_information.hit
        miss = cache_information.miss

        self.logger.info(f'Finished episode {self.episode_num}. Reward: {self.episode_reward}.')
        end_episode_state = self.converter.system_to_agent_state('END', '', OperationType.EndEpisode, {'hits': hits,
                                                                                                       'miss': miss})
        final_reward = hits - miss
        self.episode_reward += final_reward
        self.agent.observe(
            preprocessed_states=end_episode_state,
            actions=self.converter.system_to_agent_action(False),
            next_states=end_episode_state,
            internals=[],
            rewards=final_reward,
            terminals=True
        )
        self.episode_rewards.append(self.episode_reward)
        # TODO save agent weights
        self.episode_num += 1
        self._incomplete_experience_storage.clear()
        self.episode_reward = 0

    def save_results(self):
        np.savetxt(f'{self.result_dir}/episode_rewards.txt', np.asarray(self.episode_rewards), delimiter=',')
        np.savetxt(f'{self.result_dir}/losses.txt', np.asarray(self.losses, dtype='float32'), delimiter=',')
        np.savetxt(f'{self.result_dir}/stats.txt', np.asarray(self.episode_stats), delimiter=',', fmt='%s')
        # TODO add more debug information: agent action time, system wait time, evaluation time, episode duration.


class CachingStrategyRLConverter(RLConverter):
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
        # TODO get capacity
        return np.concatenate([np.array([key_encoded, operation_type_encoded, hits, miss])])

    def system_to_agent_action(self, should_cache: bool) -> np.ndarray:
        return np.ones(1, dtype='int32') if should_cache else np.zeros(1, dtype='int32')

    def agent_to_system_action(self, actions: np.ndarray, **kwargs) -> bool:
        return (actions.flatten() == 1).item()

    def system_to_agent_reward(self, experience: _IncompleteExperienceEntry) -> float:
        hits = experience.hits
        miss = experience.miss
        should_cache = self.agent_to_system_action(experience.agent_action)
        if should_cache:
            # 1- Should cache -> multiple hits -> expires/invalidates: Complete experience, reward
            #     2- Should cache -> no hits -> expires/invalidates: complete experience, punish
            reward = hits
        else:
            #     4- Shouldn't cache -> hit(s): complete experience, punish
            #   3- shouldn't cache -> no hits or miss: reward with 1
            reward = -miss if miss > 0 else 1
        self.logger.debug("Hits: {}, Miss: {}, Reward: {}".format(experience.hits, experience.miss, reward))
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
