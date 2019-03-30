from prometheus_client import Counter, Gauge, Info, start_http_server

CACHE_HIT_COUNTER = Counter('cache_hits',
                            'Number of cache hits.')
REQUEST_PER_ENDPOINT_COUNTER = Counter('requests_for_host', 'Number of runs of the process_request method',
                                       ['method', 'endpoint'])

CACHE_MISS_COUNTER = Counter('cache_miss',
                             'Number of cache misses.')

CACHE_INVALIDATION_COUNTER = Counter('cache_invalidation',
                                     'Number of cache invalidates.')

CACHE_SIZE_MONITOR = Gauge('cache_size', 'Number of objects in the cache')
EXPERIMENT_INFO_MONITOR = Info('experiment', 'Description of experiment')

