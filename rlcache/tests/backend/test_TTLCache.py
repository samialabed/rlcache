from unittest import TestCase
from unittest.mock import Mock

from time import sleep

from rlcache.backend import InMemoryStorage
from rlcache.backend.ttl_cache import TTLCache


class TestTTLCacheV2(TestCase):

    def test_order_is_based_on_ttl(self):
        key1 = 'key'
        key2 = '000'
        value = 'whatever'
        storage = InMemoryStorage(10)
        cache = TTLCache(storage)

        cache.set(key1, value, 10000000000)

        cache.set(key2, value, 30)

        assert cache.expiration_time_list[0]

    def test_get(self):
        key = 'key'
        value = '5'

        storage = InMemoryStorage(10)
        storage.set(key, value)

        cache = TTLCache(storage)
        get_result = cache.get(key)
        assert value == get_result, f"Expected '{value}' but got {get_result}"

    def test_set(self):
        key = 'key'
        value = '5'
        storage = InMemoryStorage(10)
        cache = TTLCache(storage)

        cache.set(key, value, 10)

        get_storage_results = storage.get(key)
        get_cache_results = cache.get(key)
        assert get_storage_results == value, f"Expected '{value} to be stored in underline storage. got {get_storage_results}"
        assert get_cache_results == value, f"Expected '{value} to be stored in cache. got {get_cache_results}"

    def test_set_twice(self):
        key = 'key'
        value = 'old_value'
        storage = InMemoryStorage(10)
        cache = TTLCache(storage)

        cache.set(key, value, 10)

        ttl_set = cache.expiration_time_list[-1].copy()

        new_value = 'new_value'
        cache.set(key, new_value, 35)
        ttl_set_2 = cache.expiration_time_list[-1]

        assert ttl_set != ttl_set_2, f'Expected cache entry to update TTL. Was {ttl_set} and became {ttl_set_2}'
        sleep(10)
        cache_get_res = cache.get(key)
        assert cache_get_res == new_value, f'Expected new values {new_value} to still be cached. ' \
            f'instead retrieved {cache_get_res}'

    def test_get_after_expired_set(self):
        key = 'key'
        value = '5'

        storage = InMemoryStorage(10)
        cache = TTLCache(storage)
        cache.set(key, value, 3)

        sleep(4)

        expected_results = None
        get_result = cache.get(key)
        assert expected_results == get_result, f"Expected '{expected_results}' but got {get_result}"

        get_storage_results = storage.get(key)
        assert get_storage_results == expected_results, f"Expected '{expected_results}' but got {get_storage_results}"
    # def test_register_hook_func(self):
    #     self.fail()
