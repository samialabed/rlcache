from typing import Dict

from rlcache.strategies.caching_strategies.caching_strategy_base import CachingStrategy
from rlcache.strategies.caching_strategies.rl_caching_strategy import RLCachingStrategy
from rlcache.strategies.caching_strategies.simple_strategies import OnReadWriteCacheStrategy, OnReadOnlyCacheStrategy

_supported_type = ['read_write', 'read_only', 'rl_driven']


def caching_strategy_from_config(config: Dict[str, any], shared_cache_stats) -> CachingStrategy:
    # This is super hacky but anything more complicated requires better python experience.
    # TODO interesting summer refactoring.

    caching_strategy_type = config['type']
    if caching_strategy_type == "read_write":
        return OnReadWriteCacheStrategy(config, shared_cache_stats)
    elif caching_strategy_type == "read_only":
        return OnReadOnlyCacheStrategy(config, shared_cache_stats)
    elif caching_strategy_type == 'rl_driven':
        return RLCachingStrategy(config, shared_cache_stats)
    else:
        raise NotImplementedError("Type passed isn't one of the supported types: {}".format(_supported_type))
