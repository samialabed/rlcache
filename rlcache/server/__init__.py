import json
import os
from collections import Counter

from flask import Flask

from rlcache.backend import storage_from_config
from rlcache.cache_manager import CacheManager

# TODO this is ugly, but flask refactoring is pain and not worth the time.

config_file = os.environ['CONFIG_FILE']
with open(config_file, 'r') as fp:
    CONFIG = json.load(fp)

DATABASE_BACKEND = storage_from_config(CONFIG['database_backend_settings'])
CACHE_BACKEND = storage_from_config(CONFIG['cache_backend_settings'])
CACHE_MANAGER = CacheManager(config=CONFIG['cache_manager_settings'],
                             cache=CACHE_BACKEND,
                             backend=DATABASE_BACKEND)
REQUESTS_COUNTER = Counter()

app = Flask('cache_manager_server')
