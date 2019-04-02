from typing import Dict

from rlcache.cache_constants import CacheStatus
from rlcache.observers.observer import Observer
from rlcache.rl_model.converter import RLConverter
from rlcache.strategies.caching_strategies.caching_strategy_base import CachingStrategy


class RLCachingStrategy(CachingStrategy):
    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.converter = RLConverter

    def observer(self) -> Observer:
        # should maintain a dict of incomplete experiences, once it is complete, pop and put it in the agent memory.
        # queues are used by apex worker to train the agent in different threads.
        pass

    def should_cache(self, key: str, values: Dict[str, str], invalidation: CacheStatus) -> bool:
        # call converter to transform the args to states.
        # query the agent.
        # transform agent action to system results (bool).
        # return that.
        return True


class CachingStrategyRLConverter(RLConverter):
    def system_to_agent_state(self, *args, **kwargs):
        pass

    def system_to_agent_action(self, *args, **kwargs):
        pass

    def agent_to_system_action(self, actions, **kwargs):
        pass

    def system_to_agent_reward(self, *args, **kwargs):
        pass
