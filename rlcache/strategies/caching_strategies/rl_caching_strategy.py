import logging
from collections import namedtuple
from typing import Dict, List

import numpy as np
from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox, IntBox

from rlcache.backend import TTLCache, InMemoryStorage
from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObservationType
from rlcache.rl_model.converter import RLConverter
from rlcache.strategies.caching_strategies.caching_strategy_base import CachingStrategy
from rlcache.utils.vocabulary import Vocabulary

"""

How to know a non-cached entry, that didn't get any more reads or write, is a good entry to not cache. 

    TODOs:
        - shared_stats: extract information for should_cache from shared_stats
        - Reward - calculate reward based on hits and misses for the key
        - Pass more information to the state: cache_stats and capacity
        - [LP] Sunday: this should be refactored once second agent is developed and common functionality is taken out
        - Debugging: states_per_key: key -> [hits, miss] and possibly used in should_cache
    initially - I don't need to pass the result set because YCSB generates rubbish, until I build my own workload
    I can remove that.
"""


class RLCachingStrategy(CachingStrategy):
    IncompleteExperienceEntry = namedtuple('IncompleteExperienceEntry', ('state',
                                                                         'agent_action',
                                                                         'ttl',
                                                                         'hits',
                                                                         'miss'))

    def __init__(self, config: Dict[str, any], shared_stats: CacheInformation):
        super().__init__(config, shared_stats)
        self.logger = logging.getLogger(__name__)
        self.shared_stats = shared_stats
        self.converter = CachingStrategyRLConverter()

        num_indexes = config['num_fields']
        agent_config = config['agent_config']
        flattened_num_cols = 1 + num_indexes + 1 + 1 + 1  # num_indexes + key +  cache_status + hits + miss

        # action space: should cache: true or false
        # state space: [capacity (1), query key(1), query result set(num_indexes)]
        # NOTE: state space and action_space are floatbox and intbox because bug in rlgraph.
        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(shape=(flattened_num_cols,)),
                                     action_space=IntBox(2, shape=(1,)))

        self._incomplete_experience_storage = TTLCache(InMemoryStorage(capacity=config['observer_storage_capacity']))
        self._incomplete_experience_storage.register_hook_func(self._observe_expired_incomplete_experience)

    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        # TODO what about the case of a cache key that exist already in the incomplete exp?
        assert self._incomplete_experience_storage.get(key) is None, \
            "should_cache is assumed to be first call and key shouldn't be in the cache"

        state = self.converter.system_to_agent_state(key, values, operation_type, {'hits': 0, 'miss': 0})
        agent_action = self.agent.get_action(state)
        action = self.converter.agent_to_system_action(agent_action)

        incomplete_experience_entry = self.IncompleteExperienceEntry(state=state,
                                                                     agent_action=agent_action,
                                                                     ttl=ttl,
                                                                     hits=0,
                                                                     miss=0)

        self._incomplete_experience_storage.set(key,
                                                incomplete_experience_entry,
                                                incomplete_experience_entry.ttl)
        return action

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        """
        # should maintain a dict of incomplete experiences, once it is complete, pop and put it in the agent memory.
        # queues are used by apex worker to train the agent in different threads.
        # call to when experience is done self.agent.observe()
        """
        experience = self._incomplete_experience_storage.get(key)  # type: RLCachingStrategy.IncompleteExperienceEntry
        if observation_type == ObservationType.Hit:
            experience.hits += 1
            # renew the cache ttl entry
            self._incomplete_experience_storage.set(key, experience, experience.ttl)
        else:
            if observation_type == ObservationType.Miss:
                experience.miss += 1
            # renew the cache ttl entry
            self.converter.system_to_agent_reward(experience)
            self._incomplete_experience_storage.delete(key)

    def _observe_expired_incomplete_experience(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        # should reward
        experience = self._incomplete_experience_storage.get(key)
        self.converter.system_to_agent_reward(experience)


class CachingStrategyRLConverter(RLConverter):
    def __init__(self, padding_size: int = 10):
        self.padding_size = padding_size
        self.vocabulary = Vocabulary()
        # values_key_vocab: TODO this is to maintain to and from dict that translate result sets' keys
        # self.values_key_vocabulary = Vocabulary(add_pad=False, add_unk=False)

    def agent_to_system_state(self, state: np.ndarray):
        key_encoded = state[0]
        operation_type_encoded = state[1]
        cache_hits = state[2]
        cache_miss = state[3]
        values_encoded = state[4:]

        key = self.vocabulary.get_name_for_id(key_encoded)
        operation_type = self.vocabulary.get_name_for_id(operation_type_encoded)
        values = self.vocabulary.translate_tokenized_array_to_list_words(values_encoded)

        return key, values, operation_type, cache_hits, cache_miss

    def system_to_agent_state(self, key, values, operation_type, info: Dict[str, any]) -> np.ndarray:
        # TODO get this from observer
        key_encoded = self.vocabulary.add_or_get_id(key)
        values_encoded = self.vocabulary.get_id_or_add_multiple(self._extracted_values_ordered(values),
                                                                self.padding_size)
        operation_type_encoded = self.vocabulary.add_or_get_id(operation_type)
        hits = info.get('hits')
        miss = info.get('miss')
        # TODO get capacity
        return np.concatenate([np.array([key_encoded, operation_type_encoded, hits, miss]), values_encoded])

    def system_to_agent_action(self, should_cache: bool) -> np.ndarray:
        return np.ones(1, dtype='int32') if should_cache else np.zeros(1, dtype='int32')

    def agent_to_system_action(self, actions: np.ndarray, **kwargs) -> bool:
        return (actions.flatten() == 1).item()

    def system_to_agent_reward(self, experience: RLCachingStrategy.IncompleteExperienceEntry) -> float:

        # incomplete experience:

        # reward more hits
        # if no hits, punish
        # if not cached but hits, punish
        hits = experience.hits
        miss = experience.miss
        should_cache = self.agent_to_system_action(experience.agent_action)
        if should_cache:
            # 1- Should cache -> multiple hits -> expires/invalidates: Complete experience, reward
            #     2- Should cache -> no hits -> expires/invalidates: complete experience, punish
            return hits
        else:
            #     4- Shouldn't cache -> hit(s): complete experience, punish
            return -miss

    def _extracted_values_ordered(self, values: Dict[str, any]) -> List[any]:
        sorted_values_by_key = []
        for k in sorted(values.keys()):
            sorted_values_by_key.append(values[k])
        return sorted_values_by_key
