import logging

from flask import request, jsonify

from rlcache.server import app, CONFIG, REQUESTS_COUNTER, CACHE_MANAGER, DATABASE_BACKEND

# TODO Replace print with logger
# TODO figure out difference between insert and update
# TODO Distinguish between /close for load and /close for workload

logger = logging.getLogger(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


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

    try:
        results = CACHE_MANAGER.get(key)

        response = {'key': key, 'values': results}
        logger.debug("get: {}".format(response))
        return jsonify(response)
    except Exception as e:
        logger.error("Failed to get on: {}. message: {}".format(key, e))


@app.route('/update', methods=['POST'])
def update():
    path = str(request.path)
    REQUESTS_COUNTER[path] += 1
    req_data = request.get_json()
    key = req_data['key']
    values = req_data['values']

    logger.debug("update: key: {}, values: {}".format(key, values))
    CACHE_MANAGER.set(key, values)
    DATABASE_BACKEND.set(key, values)

    return 'Success'


@app.route('/insert', methods=['POST'])
def insert():
    path = str(request.path)
    REQUESTS_COUNTER[path] += 1

    req_data = request.get_json()
    key = req_data['key']
    values = req_data['values']

    logger.debug("insert: key: {}, values: {}".format(key, values))
    CACHE_MANAGER.set(key, values)
    DATABASE_BACKEND.set(key, req_data['values'])

    return 'Success'


@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({'cache_stats': CACHE_MANAGER.stats(),
                    'database_size': DATABASE_BACKEND.size(),
                    'requests_counter': REQUESTS_COUNTER,
                    'experiment_config': CONFIG
                    })
