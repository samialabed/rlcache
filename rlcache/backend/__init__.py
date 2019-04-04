from typing import Dict

from rlcache.backend.inmemory import InMemoryStorage
from rlcache.backend.inmemory_ttl import TTLCache
from rlcache.observers.observer import ObserversOrchestrator


def storage_from_config(config: Dict[str, any]):
    # This is super hacky but anything more complicated requires better python experience.
    # TODO interesting summer refactoring.

    storage_type = config['type']
    if storage_type == "inmemory":
        return InMemoryStorage(config)
    else:
        raise NotImplementedError("Storage: {} isn't implemented.".format(storage_type))


def cache_from_config(config: Dict[str, any], observers_orchestrator: ObserversOrchestrator):
    storage_type = config['type']
    if storage_type == "cache_inmemory":
        return TTLCache(InMemoryStorage(config), observers_orchestrator)
    else:
        raise NotImplementedError("Storage: {} isn't implemented.".format(storage_type))
