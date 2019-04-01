#!/bin/bash

export FLASK_APP="server.cache_manager_server"
export FLASK_ENV="development"
export PYTHONPATH=$(pwd)
export CONFIG_FILE="$(pwd)/../configs/simple_config.json"
# I don't like this config init thing, TODO figure out how to pass arguments to flask...
flask run
