#!/usr/bin/env python
from collections import Counter

from flask import Flask, request, jsonify

from backend.inmemory import InMemoryStorage
from eviction_strategies.lru_cache import LRUCache

"""
TODOs:
    - Take in a config file and parse it for various cache strategy.
    - FULL REFACTOR OF THIS, server stuff should be in a class. The counters can be an aspect?
    - Replace print with logger
"""

database_capacity = 1000000
cache_capacity = 10000
cache_backend = InMemoryStorage(cache_capacity)
database_backend = InMemoryStorage(database_capacity)

cache = LRUCache(cache_backend)

app = Flask('cache_manager_server')

experiment_info = {'cache_fixed_capacity': str(cache_capacity),
                   'database_fixed_capacity': str(database_capacity)},
cache_metrics = Counter({'cache_hits': 0,
                         'cache_miss': 0,
                         'cache_invalidation': 0,
                         'cache_size': 0})
requests_counter = Counter()


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/delete', methods=['DELETE'])
def delete():
    path = str(request.path)
    cache_metrics['request_counter'][path] += 1
    req_data = request.get_json()
    key = req_data['key']
    if cache.contains(key):
        # TODO punish for invalidation
        cache_metrics['cache_invalidation'] += 1
    status = cache.delete(key)
    database_backend.delete(key)
    response = {'key': key, 'deleted': status}
    print("Results of delete: {}".format(response))
    return jsonify(response)


@app.route('/close', methods=['DELETE'])
def close():
    # TODO record an end of an episode
    path = str(request.path)
    requests_counter[path] += 1

    return 'Success'


@app.route('/get', methods=['POST'])
def get():
    path = str(request.path)
    requests_counter[path] += 1
    req_data = request.get_json()
    key = req_data['key']
    print("Get: {}".format(req_data))

    results = cache.get(key)
    if not results:
        # TODO this should be an interface and swap in RLCache vs dummy
        results = database_backend.get(key)  # retrieve from DB on failure of read
        cache.set(key, results)
        cache_metrics['cache_miss'] += 1
        # TODO record a cache miss
    else:
        cache_metrics['cache_hits'] += 1
        # record a cache hit, an end of an experience

    response = {'key': key, 'values': results}
    print("Results of get: {}".format(response))
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    path = str(request.path)
    requests_counter[path] += 1
    req_data = request.get_json()
    key = req_data['key']
    print("Update: {}".format(req_data))
    values = req_data['values']

    if cache.contains(key):
        # TODO punish for invalidation
        cache_metrics['cache_invalidation'] += 1

    # TODO caching strategy, cache on write?
    cache.set(key, values)
    database_backend.set(key, values)

    return 'Success'


@app.route('/insert', methods=['POST'])
def insert():
    path = str(request.path)
    requests_counter[path] += 1

    req_data = request.get_json()
    print("Insert: {}".format(req_data))
    key = req_data['key']
    if cache.contains(key):
        # TODO punish for invalidation
        cache_metrics['cache_invalidation'] += 1
    # TODO caching strategy, cache on write?
    cache.set(key, req_data['values'])
    database_backend.set(key, req_data['values'])

    return 'Success'


@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({'cache_state': cache.state(),
                    'database_size': database_backend.size(),
                    'stats': cache_metrics,
                    'requests_counter': requests_counter,
                    'experiment_info': experiment_info
                    })
