#!/usr/bin/env python
from flask import Flask, request

from backend.inmemory import InMemoryCache
from baselines.lru_cache import LRUCache

"""
TODOs:
    - Parse command line for different cache implementations.
    - Record metrics for cache hits and misses.
    - FULL REFACTOR OF THIS, server stuff should be in a class.
"""

cache_backend = InMemoryCache()
cache = LRUCache(cache_backend)
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/get', methods=['POST'])
def get():
    req_data = request.get_json()
    return cache.get(req_data['key'])


@app.route('/update', methods=['POST'])
def update():
    req_data = request.get_json()
    if cache.set(req_data['key'], req_data['value']):
        return 'Success'
    else:
        return 'Fail'


@app.route('/set', methods=['POST'])
def set():
    req_data = request.get_json()
    if cache.set(req_data['key'], req_data['value']):
        return 'Success'
    else:
        return 'Fail'
