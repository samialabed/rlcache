from typing import Dict

from rlcache.backend.inmemory import InMemoryStorage
from rlcache.backend.ttl_cache_v2 import TTLCacheV2


def storage_from_config(config: Dict[str, any]):
    # This is super hacky but anything more complicated requires better python experience.
    # TODO interesting summer refactoring.

    storage_type = config['type']
    if storage_type == "inmemory":
        return InMemoryStorage(capacity=config['capacity'])
    elif storage_type == "cache_inmemory":
        return TTLCacheV2(InMemoryStorage(capacity=config['capacity']))
    else:
        raise NotImplementedError("Storage: {} isn't implemented.".format(storage_type))
