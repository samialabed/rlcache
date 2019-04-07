#!/bin/bash

export FLASK_APP="server.cache_manager_server"
export FLASK_ENV="development"
export PYTHONPATH=$(pwd)
# create a backend endpoint that takes in the config?
#export CONFIG_FILE="$(pwd)/../configs/simple_config.json"
export CONFIG_FILE="$(pwd)/../configs/rl_caching_strategy.json"
# I don't like this config init thing, TODO figure out how to pass arguments to flask...
flask run --no-reload --without-threads
#python rlcache/server/cache_manager_server.py