from typing import Dict

from rlcache.backend.inmemory import InMemoryStorage


def storage_from_config(config: Dict[str, any]):
    # This is super hacky but anything more complicated requires better python experience.
    # TODO interesting summer refactoring.

    storage_type = config['type']
    if storage_type == "inmemory":
        return InMemoryStorage(config)
