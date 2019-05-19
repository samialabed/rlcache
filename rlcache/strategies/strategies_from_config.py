import json
import os
from typing import Dict

from rlcache.cache_constants import CacheInformation
from rlcache.strategies.caching_strategies.base_caching_strategy import CachingStrategy
from rlcache.strategies.caching_strategies.rl_caching_strategy import RLCachingStrategy
from rlcache.strategies.caching_strategies.simple_strategies import OnReadWriteCacheStrategy, OnReadOnlyCacheStrategy
from rlcache.strategies.eviction_strategies.base_eviction_strategy import EvictionStrategy
from rlcache.strategies.eviction_strategies.fifo_eviction_strategy import FIFOEvictionStrategy
from rlcache.strategies.eviction_strategies.lru_eviction_strategy import LRUEvictionStrategy
from rlcache.strategies.eviction_strategies.rl_eviction_strategy import RLEvictionStrategy
from rlcache.strategies.multi_task.rl_multi_task_cache_strategy import RLMultiTasksStrategy
from rlcache.strategies.ttl_estimation_strategies.base_ttl_strategy import TtlStrategy
from rlcache.strategies.ttl_estimation_strategies.fixed_ttl_strategy import FixedTtlStrategy
from rlcache.strategies.ttl_estimation_strategies.rl_ttl_strategy import RLTtlStrategy


def strategies_from_config(config: Dict[str, any],
                           results_dir: str,
                           cache_stats: CacheInformation) -> [CachingStrategy, EvictionStrategy, TtlStrategy]:
    if 'multi_strategy_settings' in config:
        caching_strategy = eviction_strategy = ttl_strategy = multi_from_config(config['multi_strategy_settings'],
                                                                                results_dir,
                                                                                cache_stats)

    else:
        caching_strategy = caching_strategy_from_config(config['caching_strategy_settings'], results_dir, cache_stats)
        eviction_strategy = eviction_strategy_from_config(config['eviction_strategy_settings'], results_dir,
                                                          cache_stats)
        ttl_strategy = ttl_strategy_from_config(config['ttl_strategy_settings'], results_dir, cache_stats)

    return [caching_strategy, eviction_strategy, ttl_strategy]


def caching_strategy_from_config(config: Dict[str, any],
                                 results_dir: str,
                                 cache_stats: CacheInformation) -> CachingStrategy:
    _supported_type = ['read_write', 'read_only', 'rl_driven']

    results_dir += '/caching_strategy/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    caching_strategy_type = config['type']
    if caching_strategy_type == "read_write":
        return OnReadWriteCacheStrategy(config, results_dir, cache_stats)
    elif caching_strategy_type == "read_only":
        return OnReadOnlyCacheStrategy(config, results_dir, cache_stats)
    elif caching_strategy_type == 'rl_driven':
        # load the agent config file into the dict
        with open(config['agent_config'], 'r') as fp:
            agent_config = json.load(fp)
            config['agent_config'] = agent_config
        # copy agent config to result directory for reproducibility
        with open(f'{results_dir}/agent_config.json', 'w') as outfile:
            json.dump(agent_config, outfile, indent=2)
        return RLCachingStrategy(config, results_dir, cache_stats)
    else:
        raise NotImplementedError("Type passed isn't one of the supported types: {}".format(_supported_type))


def eviction_strategy_from_config(config: Dict[str, any],
                                  results_dir: str,
                                  cache_stats: CacheInformation) -> EvictionStrategy:
    _supported_type = ['lru', 'rl_driven']

    results_dir += '/eviction_strategy/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    eviction_strategy_type = config['type']
    if eviction_strategy_type == "lru":
        return LRUEvictionStrategy(config, results_dir, cache_stats)
    elif eviction_strategy_type == "fifo":
        return FIFOEvictionStrategy(config, results_dir, cache_stats)
    elif eviction_strategy_type == 'rl_driven':
        # load the agent config file into the dict
        with open(config['agent_config'], 'r') as fp:
            agent_config = json.load(fp)
            config['agent_config'] = agent_config
        # copy agent config to result directory for reproducibility
        with open(f'{results_dir}/agent_config.json', 'w') as outfile:
            json.dump(agent_config, outfile, indent=2)
        return RLEvictionStrategy(config, results_dir, cache_stats)
    else:
        raise NotImplementedError("Type passed isn't one of the supported types: {}".format(_supported_type))


def ttl_strategy_from_config(config: Dict[str, any], results_dir: str, cache_stats: CacheInformation) -> TtlStrategy:
    _supported_type = ['fixed']
    results_dir += '/ttl_strategy/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    ttl_strategy_type = config['type']
    if ttl_strategy_type == "fixed":
        return FixedTtlStrategy(config, results_dir, cache_stats)
    elif ttl_strategy_type == 'rl_driven':
        # load the agent config file into the dict
        with open(config['agent_config'], 'r') as fp:
            agent_config = json.load(fp)
            config['agent_config'] = agent_config
        # copy agent config to result directory for reproducibility
        with open(f'{results_dir}/agent_config.json', 'w') as outfile:
            json.dump(agent_config, outfile, indent=2)
        return RLTtlStrategy(config, results_dir, cache_stats)
    else:
        raise NotImplementedError("Type passed isn't one of the supported types: {}".format(_supported_type))


def multi_from_config(config: Dict[str, any], results_dir: str, cache_stats: CacheInformation) -> RLMultiTasksStrategy:
    # load the agent config file into the dict
    results_dir += '/multi_strategy/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(config['agent_config'], 'r') as fp:
        agent_config = json.load(fp)
        config['agent_config'] = agent_config
    # copy agent config to result directory for reproducibility
    with open(f'{results_dir}/agent_config.json', 'w') as outfile:
        json.dump(agent_config, outfile, indent=2)
    return RLMultiTasksStrategy(config, results_dir, cache_stats)
