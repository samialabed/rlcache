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
"""

cache_backend = InMemoryCache()
cache = LRUCache(cache_backend)
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/delete', methods=['POST'])
def delete():
    req_data = request.get_json()
    key = req_data['key']
    status = cache.delete(key)
    response = {'key': key, 'deleted': status}
    print("Results of delete: {}".format(response))
    return jsonify(response)


@app.route('/get', methods=['POST'])
def get():
    req_data = request.get_json()
    key = req_data['key']
    saved_results = cache.get(key)
    response = {'key': key, 'values': saved_results}
    print("Results of get: {}".format(response))
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    req_data = request.get_json()
    print("Requested data for update: {}".format(req_data))

    if cache.set(req_data['key'], req_data['values']):
        return 'Success'
    else:
        return 'Fail'


@app.route('/set', methods=['POST'])
def set():
    req_data = request.get_json()
    print("Requested data for set: {}".format(req_data))
    if cache.set(req_data['key'], req_data['values']):
        return 'Success'
    else:
        return 'Fail'
