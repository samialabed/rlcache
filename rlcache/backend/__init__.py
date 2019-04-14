from typing import Dict

from rlcache.backend.inmemory import InMemoryStorage
from rlcache.backend.ttl_cache import TTLCache


def storage_from_config(config: Dict[str, any]):
    storage_type = config['type']
    if storage_type == "inmemory":
        return InMemoryStorage(capacity=config.get('capacity'))  # get or assume no limit
    elif storage_type == "cache_inmemory":
        return TTLCache(InMemoryStorage(capacity=config['capacity']))
    else:
        raise NotImplementedError("Storage: {} isn't implemented.".format(storage_type))
