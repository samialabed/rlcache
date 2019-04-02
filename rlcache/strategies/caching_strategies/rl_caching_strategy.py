from typing import Dict

import numpy as np
from rlgraph.agents import Agent
from rlgraph.spaces import BoolBox, IntBox, TextBox
from rlgraph.spaces import Dict as RLGraphDict

from rlcache.cache_constants import CacheStatus
from rlcache.observers.observer import Observer
from rlcache.rl_model.converter import RLConverter
from rlcache.strategies.caching_strategies.caching_strategy_base import CachingStrategy


class RLCachingStrategy(CachingStrategy):
    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.converter = CachingStrategyRLConverter()

        num_indexes = config['num_fields']  # TODO for now only 1 state, the key, only
        agent_config = config['agent_config']  # DQN agent config
        max_capacity = config['max_capacity']

        # TODO encode field = 0 capacity, field 1 = key, field 2 until num_indexes = values.
        # action space: should cache: true or false
        # state space:
        self.agent = Agent.from_spec(agent_config,
                                     state_space=RLGraphDict({'key': TextBox(shape=(1, 1)),
                                                              'values': TextBox(shape=(num_indexes, 1)),
                                                              'capacity': IntBox(max_capacity)
                                                              }),
                                     action_space=BoolBox())

    def observer(self) -> Observer:
        # should maintain a dict of incomplete experiences, once it is complete, pop and put it in the agent memory.
        # queues are used by apex worker to train the agent in different threads.
        # call to when experience is done self.agent.observe()
        # self.agent.observe()
        pass

    def should_cache(self, key: str, values: Dict[str, str], cache_status: CacheStatus) -> bool:
        # call converter to transform the args to states.
        # query the agent.
        # transform agent action to system results (bool).
        # return that.
        # save the action taken, and the experience
        state = self.converter.system_to_agent_state(key=key, values=values, cache_status=cache_status)
        return self.agent.get_action(state)


class CachingStrategyRLConverter(RLConverter):
    def system_to_agent_state(self, *args, **kwargs) -> np.ndarray:
        # TODO replace with meaningful state representation
        capacity = 50  # TODO get this from observer
        return np.array({'key': kwargs.get('key'),
                         'values': kwargs.get('values'),
                         'capacity': capacity
                         })

    def system_to_agent_action(self, *args, **kwargs):
        return np.ones(1)

    def agent_to_system_action(self, actions, **kwargs):
        pass

    def system_to_agent_reward(self, *args, **kwargs):
        pass
