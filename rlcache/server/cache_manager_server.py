import logging
import logging.config as log_cfg

from flask import request, jsonify

from rlcache.server import app, CONFIG, REQUESTS_COUNTER, CACHE_MANAGER, DATABASE_BACKEND
from rlcache.utils.loggers import LOG_CONFIG

log_cfg.dictConfig(LOG_CONFIG)

logger = logging.getLogger(__name__)
skip_cache = True  # start with cache being skipped


@app.route('/')
def hello_world():
    return 'Hello, World!, I am RLCache'


@app.route('/delete', methods=['DELETE'])
def delete():
    path = str(request.path)
    REQUESTS_COUNTER[path] += 1
    req_data = request.get_json()
    key = req_data['key']

    CACHE_MANAGER.delete(key)
    DATABASE_BACKEND.delete(key)

    response = {'key': key, 'values': {'deleted': 'True'}}
    logger.debug("delete: {}".format(response))
    return jsonify(response)


@app.route('/close', methods=['DELETE'])
def close():
    # TODO record an end of an episode
    path = str(request.path)
    REQUESTS_COUNTER[path] += 1

    return 'Success'


@app.route('/get', methods=['POST'])
def get():
    path = str(request.path)
    REQUESTS_COUNTER[path] += 1
    req_data = request.get_json()
    key = req_data['key']
    results = CACHE_MANAGER.get(key)

    response = {'key': key, 'values': results}
    logger.debug("get: {}".format(response))
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    path = str(request.path)
    REQUESTS_COUNTER[path] += 1
    req_data = request.get_json()
    key = req_data['key']
    values_to_update = req_data['values']
    values = CACHE_MANAGER.get(key)
    logger.debug("update: key: {}, values to update: {}, previous values: {}".format(key, values_to_update, values))

    for k, v in values_to_update.items():
        values[k] = v

    DATABASE_BACKEND.set(key, values)
    CACHE_MANAGER.set(key, values)

    return 'Success'


@app.route('/insert', methods=['POST'])
def insert():
    global skip_cache
    path = str(request.path)
    REQUESTS_COUNTER[path] += 1

    req_data = request.get_json()
    key = req_data['key']
    values = req_data['values']

    logger.debug("insert: key: {}, values: {}".format(key, values))
    if not skip_cache:  # exist for loading phase
        CACHE_MANAGER.set(key, values)
    DATABASE_BACKEND.set(key, req_data['values'])

    return 'Success'


@app.route('/end_loading_phase', methods=['GET'])
def end_loading_phase():
    # TODO maybe make this a POST endpoint that can receive configs?
    global skip_cache
    skip_cache = True
    return 'Success'


@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({'cache_stats': CACHE_MANAGER.stats(),
                    'database_size': DATABASE_BACKEND.size(),
                    'requests_counter': REQUESTS_COUNTER,
                    'experiment_config': CONFIG
                    })
