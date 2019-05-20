#!/bin/bash

export FLASK_APP="rlcache.server.cache_manager_server"
export FLASK_ENV="development"
export PYTHONPATH=$(pwd)
# create a backend endpoint that takes in the config?
export CONFIG_FILE="$(pwd)/configs/rl_ttl_strategy.json"
flask run --no-reload --without-threads
#python rlcache/server/cache_manager_server.py
