from typing import Dict

from rlcache.strategies.ttl_selection_strategies.ttl_strategy_fixed import FixedTtlStrategy


def ttl_strategy_from_config(config: Dict[str, any], shared_cache_stats):
    # This is super hacky but anything more complicated requires better python experience.
    # TODO interesting summer refactoring.

    ttl_strategy_type = config['type']
    if ttl_strategy_type == "fixed":
        return FixedTtlStrategy(config, shared_cache_stats)
    else:
        raise NotImplementedError
