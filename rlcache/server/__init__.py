import json
import os
from collections import Counter

import time
from flask import Flask

from rlcache.backend import storage_from_config
from rlcache.cache_manager import CacheManager

config_file = os.environ['CONFIG_FILE']
with open(config_file, 'r') as fp:
    CONFIG = json.load(fp)

cache_capacity = CONFIG['cache_backend_settings'].get('capacity', 'unlimited')
results_dir = f"results/{CONFIG['experiment_name']}/cache_capacity_{cache_capacity}/{time.strftime('%Y_%m_%d_%H_%M')}"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
with open(f'{results_dir}/config_file.json', 'w') as outfile:
    json.dump(CONFIG, outfile, indent=2)

DATABASE_BACKEND = storage_from_config(CONFIG['database_backend_settings'])
CACHE_BACKEND = storage_from_config(CONFIG['cache_backend_settings'])
CACHE_MANAGER = CacheManager(config=CONFIG['cache_manager_settings'],
                             cache=CACHE_BACKEND,
                             backend=DATABASE_BACKEND,
                             result_dir=results_dir)
REQUESTS_COUNTER = Counter()

app = Flask('cache_manager_server')
