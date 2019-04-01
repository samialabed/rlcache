from typing import Dict

from caching_strategies.simple_strategies import OnReadWriteCacheStrategy, OnReadOnlyCacheStrategy


def caching_strategy_from_config(config: Dict[str, any]):
    # This is super hacky but anything more complicated requires better python experience.
    # TODO interesting summer refactoring.

    caching_strategy_type = config['type']
    if caching_strategy_type == "read_write":
        return OnReadWriteCacheStrategy(config)
    elif caching_strategy_type == "read_only":
        return OnReadOnlyCacheStrategy(config)
