#!/bin/bash

export FLASK_APP="cache_manager_server"
export FLASK_ENV="development"

#flask run --port 5000

uwsgi --http 127.0.0.1:8000 --wsgi-file cache_manager_server.py --callable app_dispatch
