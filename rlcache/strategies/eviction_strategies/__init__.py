from typing import Dict

from rlcache.strategies.eviction_strategies.lru_eviction_strategy import LRUEvictionStrategy


def eviction_strategy_from_config(config: Dict[str, any], shared_cache_stats):
    # This is super hacky but anything more complicated requires better python experience.
    # TODO interesting summer refactoring.

    eviction_strategy_type = config['type']
    if eviction_strategy_type == "lru":
        return LRUEvictionStrategy(config, shared_cache_stats)
    else:
        raise NotImplementedError
