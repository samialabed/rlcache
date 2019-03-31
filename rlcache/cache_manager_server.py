#!/usr/bin/env python
from collections import Counter

from flask import Flask, request, jsonify

from cache_manager import CacheManager
from caching_strategies.simple_strategies import OnReadWriteCacheStrategy
from eviction_strategies.lru_eviction_strategy import LRUEvictionStrategy
from rlcache.backend.inmemory import InMemoryStorage
from ttl_selection_strategies.ttl_strategy_fixed import FixedTtlStrategy

"""
TODOs:
    - Take in a config file and parse it for various cache strategy. each strategy has settings dict.
    - FULL REFACTOR OF THIS, server stuff should be in a class. The counters can be an aspect?
    - Replace print with logger
    - Difference between insert and update?
    - Distinquish between /close for load and /close for workload
"""

database_capacity = 1000000
cache_capacity = 10000
cache_backend = InMemoryStorage(cache_capacity)
database_backend = InMemoryStorage(database_capacity)

cache_manager = CacheManager(caching_strategy=OnReadWriteCacheStrategy(),
                             eviction_strategy=LRUEvictionStrategy(),
                             ttl_strategy=FixedTtlStrategy(300),
                             cache=cache_backend,
                             backend=database_backend)

app = Flask('cache_manager_server')

experiment_info = {'cache_fixed_capacity': str(cache_capacity),
                   'database_fixed_capacity': str(database_capacity)},
requests_counter = Counter()


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/delete', methods=['DELETE'])
def delete():
    path = str(request.path)
    requests_counter[path] += 1
    req_data = request.get_json()
    key = req_data['key']

    cache_manager.delete(key)
    database_backend.delete(key)

    response = {'key': key, 'values': {'deleted': 'True'}}
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

    results = cache_manager.get(key)

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

    cache_manager.set(key, values)
    database_backend.set(key, values)

    return 'Success'


@app.route('/insert', methods=['POST'])
def insert():
    path = str(request.path)
    requests_counter[path] += 1

    req_data = request.get_json()
    print("Insert: {}".format(req_data))
    key = req_data['key']
    values = req_data['values']

    cache_manager.set(key, values)
    database_backend.set(key, req_data['values'])

    return 'Success'


@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({'cache_stats': cache_manager.stats(),
                    'cache_size': cache_manager.cache.size(),
                    'database_size': database_backend.size(),
                    'requests_counter': requests_counter,
                    'experiment_info': experiment_info
                    })
