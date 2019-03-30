#!/usr/bin/env python
from flask import Flask, request, jsonify
from prometheus_client import make_wsgi_app
from werkzeug.wsgi import DispatcherMiddleware

from backend.inmemory import InMemoryCache
from baselines.lru_cache import LRUCache
from metrics import CACHE_SIZE_MONITOR, REQUEST_PER_ENDPOINT_COUNTER, CACHE_MISS_COUNTER, \
    CACHE_HIT_COUNTER, CACHE_INVALIDATION_COUNTER, EXPERIMENT_INFO_MONITOR

"""
TODOs:
    - Parse command line for different cache implementations.
    - FULL REFACTOR OF THIS, server stuff should be in a class.'
    - maybe better? https://github.com/philwinder/prometheus-python/blob/master/app.py
"""

capacity = 10000000
cache_backend = InMemoryCache(capacity)
cache = LRUCache(cache_backend)

app = Flask('cache_manager_server')
app_dispatch = DispatcherMiddleware(app, {
    '/metrics': make_wsgi_app()
})

# TODO make this contains a whole experiment config
EXPERIMENT_INFO_MONITOR.info({'version': '1', 'cache_fixed_capacity': str(capacity)})
CACHE_SIZE_MONITOR.set_function(cache.size)  # Set callback that monitors the cache size


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/delete', methods=['DELETE'])
def delete():
    path = str(request.path)
    verb = request.method
    label_dict = {"method": verb,
                  "endpoint": path}
    REQUEST_PER_ENDPOINT_COUNTER.labels(**label_dict).inc()

    req_data = request.get_json()
    key = req_data['key']
    status = cache.delete(key)
    response = {'key': key, 'deleted': status}
    print("Results of delete: {}".format(response))
    return jsonify(response)


@app.route('/close', methods=['DELETE'])
def close():
    path = str(request.path)
    verb = request.method
    label_dict = {"method": verb,
                  "endpoint": path}
    REQUEST_PER_ENDPOINT_COUNTER.labels(**label_dict).inc()

    print("Cache size before clear: {}".format(cache.size()))
    cache.clear()
    # TODO record an end of an episode
    print("Cache size after clear: {}".format(cache.size()))
    return 'Success'


@app.route('/get', methods=['POST'])
def get():
    path = str(request.path)
    verb = request.method
    label_dict = {"method": verb,
                  "endpoint": path}
    REQUEST_PER_ENDPOINT_COUNTER.labels(**label_dict).inc()

    req_data = request.get_json()
    key = req_data['key']
    saved_results = cache.get(key)
    if not saved_results:
        cache.set(key, {'cached_from_get': True})
        CACHE_MISS_COUNTER.inc()
        # TODO record a cache miss
    else:
        CACHE_HIT_COUNTER.inc()
        # record a cache hit, an end of an experience

    response = {'key': key, 'values': saved_results}
    print("Results of get: {}".format(response))
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    path = str(request.path)
    verb = request.method
    label_dict = {"method": verb,
                  "endpoint": path}
    REQUEST_PER_ENDPOINT_COUNTER.labels(**label_dict).inc()
    req_data = request.get_json()
    print("Requested data for update: {}".format(req_data))

    key = req_data['key']
    if cache.contains(key):
        # TODO punish for invalidation
        CACHE_INVALIDATION_COUNTER.inc()

    if cache.set(key, req_data['values']):
        return 'Success'
    else:
        return 'Fail'


@app.route('/insert', methods=['POST'])
def insert():
    path = str(request.path)
    verb = request.method
    label_dict = {"method": verb,
                  "endpoint": path}
    REQUEST_PER_ENDPOINT_COUNTER.labels(**label_dict).inc()
    req_data = request.get_json()
    print("Requested data for set: {}".format(req_data))

    key = req_data['key']
    if cache.contains(key):
        # TODO punish for invalidation
        CACHE_INVALIDATION_COUNTER.inc()
    if cache.set(key, req_data['values']):
        return 'Success'
    else:
        return 'Fail'


@app.route('/stats', methods=['GET'])
def stats():
    path = str(request.path)
    verb = request.method
    label_dict = {"method": verb,
                  "endpoint": path}
    REQUEST_PER_ENDPOINT_COUNTER.labels(**label_dict).inc()
    return jsonify(cache.stats())
