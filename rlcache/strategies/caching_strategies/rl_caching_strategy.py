from collections import namedtuple
from typing import Dict, List

import numpy as np
from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox, IntBox

from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observers.observer import Observer, ObservationType
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
        - [LP] Reward - Look into using other metrics than hits and miss
        - Debugging: states_per_key: key -> [hits, miss] and possibly used in should_cache
    initially - I don't need to pass the result set because YCSB generates rubbish, until I build my own workload
    I can remove that.
"""


class RLCachingStrategy(CachingStrategy, Observer):
    IncompleteExperienceEntry = namedtuple('IncompleteExperienceEntry', ('state', 'agent_action', 'hits', 'miss'))

    def __init__(self, config: Dict[str, any], shared_stats: CacheInformation):
        super().__init__(config)
        self.shared_stats = shared_stats
        self.converter = CachingStrategyRLConverter()

        num_indexes = config['num_fields']
        agent_config = config['agent_config']
        flattened_num_cols = 1 + num_indexes + 1  # num_indexes + key +  cache_status

        # action space: should cache: true or false
        # state space: [capacity (1), query key(1), query result set(num_indexes)]
        # NOTE: state space and action_space are floatbox and intbox because bug in rlgraph.
        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(shape=(flattened_num_cols,)),
                                     action_space=IntBox(2, shape=(1,)))

        self._incomplete_experience = {}  # type: Dict[str, RLCachingStrategy.IncompleteExperienceEntry]

    def should_cache(self, key: str, values: Dict[str, str], cache_status: OperationType) -> bool:
        # save the action taken, and the experience

        state = self.converter.system_to_agent_state(key, values, cache_status)
        agent_action = self.agent.get_action(state)
        action = self.converter.agent_to_system_action(agent_action)
        self._incomplete_experience[key] = self.IncompleteExperienceEntry(state=state, agent_action=agent_action)
        return action

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        """
        # should maintain a dict of incomplete experiences, once it is complete, pop and put it in the agent memory.
        # queues are used by apex worker to train the agent in different threads.
        # call to when experience is done self.agent.observe()

        incomplete experience:
            1- Should cache -> multiple hits -> expires/invalidates: Complete experience, reward
            2- Should cache -> no hits -> expires/invalidates: complete experience, punish
            3- Shouldn't cache -> no hits -> expires/invalidates: complete experience, reward
            4- Shouldn't cache -> hit(s): complete experience, punish
            :param shared_stats:
        """

        pass


class CachingStrategyRLConverter(RLConverter):
    def __init__(self, padding_size: int = 10):
        self.padding_size = padding_size
        self.vocabulary = Vocabulary()
        # values_key_vocab: TODO this is to maintain to and from dict that translate result sets' keys
        # self.values_key_vocabulary = Vocabulary(add_pad=False, add_unk=False)

    def system_to_agent_state(self, key, values, cache_status) -> np.ndarray:
        # TODO get this from observer
        key_encoded = self.vocabulary.add_or_get_id(key)
        values_encoded = self.vocabulary.get_id_or_add_multiple(self._extracted_values_ordered(values),
                                                                self.padding_size)
        cache_status_encoded = self.vocabulary.add_or_get_id(cache_status)
        # TODO get capacity
        return np.concatenate([np.array([key_encoded, cache_status_encoded]), values_encoded])

    def system_to_agent_action(self, should_cache: bool) -> np.ndarray:
        return np.ones(1, dtype='int32') if should_cache else np.zeros(1, dtype='int32')

    def agent_to_system_action(self, actions: np.ndarray, **kwargs) -> bool:
        return (actions.flatten() == 1).item()

    def system_to_agent_reward(self, *args, **kwargs):
        # reward more hits
        # if no hits, punish
        # if not cached but hits, punish
        pass

    def _extracted_values_ordered(self, values: Dict[str, any]) -> List[any]:
        sorted_values_by_key = []
        for k in sorted(values.keys()):
            sorted_values_by_key.append(values[k])
        return sorted_values_by_key
