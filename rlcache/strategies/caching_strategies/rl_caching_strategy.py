from typing import Dict, List

import numpy as np
from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox, IntBox

from rlcache.cache_constants import CacheStatus
from rlcache.observers.observer import Observer
from rlcache.rl_model.converter import RLConverter
from rlcache.strategies.caching_strategies.caching_strategy_base import CachingStrategy
from rlcache.utils.vocabulary import Vocabulary


class RLCachingStrategy(CachingStrategy):
    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.converter = CachingStrategyRLConverter()

        num_indexes = config['num_fields']  # TODO for now only 1 state, the key, only
        agent_config = config['agent_config']  # DQN agent config

        flattened_num_cols = 1 + num_indexes + 1 + 1  # num_indexes + key + capacity + cache_status

        # TODO encode field = 0 capacity, field 1 = key, field 2 until num_indexes = values.
        # action space: should cache: true or false
        # state space:

        # TODO state space and action_space are floatbox and intbox because bug in rlgraph.
        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(shape=(flattened_num_cols,)),
                                     action_space=IntBox(2, shape=(1,)))

    def observer(self) -> Observer:
        # should maintain a dict of incomplete experiences, once it is complete, pop and put it in the agent memory.
        # queues are used by apex worker to train the agent in different threads.
        # call to when experience is done self.agent.observe()
        # self.agent.observe()
        pass

    def should_cache(self, key: str, values: Dict[str, str], cache_status: CacheStatus) -> bool:
        # query the agent.
        # transform agent action to system results (bool).
        # return that.
        # save the action taken, and the experience

        # call converter to transform the args to states.
        state = self.converter.system_to_agent_state(key=key, values=values, cache_status=cache_status)
        agent_prediction = self.agent.get_action(state)

        return self.converter.agent_to_system_action(agent_prediction)


class CachingStrategyRLConverter(RLConverter):
    def __init__(self, padding_size: int = 10):
        self.padding_size = padding_size
        self.vocabulary = Vocabulary()
        # values_key_vocab: TODO this is to maintain to and from dict that translate result sets' keys
        # self.values_key_vocabulary = Vocabulary(add_pad=False, add_unk=False)

    def system_to_agent_state(self, key, values, cache_status) -> np.ndarray:
        capacity = 50  # TODO get this from observer
        key_encoded = self.vocabulary.add_or_get_id(key)
        values_encoded = self.vocabulary.get_id_or_add_multiple(self._extracted_values_ordered(values),
                                                                self.padding_size)
        cache_status_encoded = self.vocabulary.add_or_get_id(cache_status)
        # return np.random.randint(13, size=13)
        return np.concatenate([np.array([key_encoded, capacity, cache_status_encoded]), values_encoded])

    def system_to_agent_action(self, *args, **kwargs) -> np.ndarray:
        return np.ones(1)

    def agent_to_system_action(self, actions: np.ndarray, **kwargs) -> bool:
        return (actions.flatten() == 1).item()

    def system_to_agent_reward(self, *args, **kwargs):
        pass

    def _extracted_values_ordered(self, values: Dict[str, any]) -> List[any]:
        sorted_values_by_key = []
        for k in sorted(values.keys()):
            sorted_values_by_key.append(values[k])
        return sorted_values_by_key
