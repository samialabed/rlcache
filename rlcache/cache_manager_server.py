#!/usr/bin/env python
from flask import Flask, request, jsonify

from backend.inmemory import InMemoryCache
from baselines.lru_cache import LRUCache

"""
TODOs:
    - Parse command line for different cache implementations.
    - Record metrics for cache hits and misses.
    - Add Delete /close to signal an end of an "episode"
    - FULL REFACTOR OF THIS, server stuff should be in a class.
    - Handle items not in a cache.
    - If get fails once, create a fake cached entry.
"""

cache_backend = InMemoryCache(10000000)
cache = LRUCache(cache_backend)
app = Flask('cache_manager_server')


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/delete', methods=['DELETE'])
def delete():
    req_data = request.get_json()
    key = req_data['key']
    status = cache.delete(key)
    response = {'key': key, 'deleted': status}
    print("Results of delete: {}".format(response))
    return jsonify(response)


@app.route('/close', methods=['DELETE'])
def close():
    print("Cache size before clear: {}".format(cache.size()))
    cache.clear()
    # TODO record an end of an episode
    print("Cache size after clear: {}".format(cache.size()))
    return 'Success'


@app.route('/get', methods=['POST'])
def get():
    req_data = request.get_json()
    key = req_data['key']
    saved_results = cache.get(key)
    if not saved_results:
        cache.set(key, {'cached_from_get': True})
        # TODO record a cache miss
    # else: record a cache hit, an end of an experience
    response = {'key': key, 'values': saved_results}
    print("Results of get: {}".format(response))
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    # TODO cache invalidation
    req_data = request.get_json()
    print("Requested data for update: {}".format(req_data))

    if cache.set(req_data['key'], req_data['values']):
        return 'Success'
    else:
        return 'Fail'


@app.route('/insert', methods=['POST'])
def insert():
    # TODO cache invalidation
    req_data = request.get_json()
    print("Requested data for set: {}".format(req_data))
    if cache.set(req_data['key'], req_data['values']):
        return 'Success'
    else:
        return 'Fail'


@app.route('/stats', methods=['GET'])
def stats():
    return cache.stats()
