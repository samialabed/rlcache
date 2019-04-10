from unittest import TestCase
from unittest.mock import Mock

from time import sleep

from rlcache.backend import TTLCache


class TestTTLCache(TestCase):

    def test_get(self):
        storage = Mock()
        storage.get.return_value = '5'
        cache = TTLCache(storage)
        get_result = cache.get('key')
        assert '5' == get_result, f"Expected '5' but got {get_result}"
        storage.get.assert_called_once_with('key', None)

    def test_set(self):
        storage = Mock()
        cache = TTLCache(storage)
        cache.set('key', '5')
        storage.set.assert_called_once_with('key', '5')

    def test_get_after_expired_set(self):
        storage = Mock()
        cache = TTLCache(storage)
        cache.set('key', '5', 2)
        storage.set.assert_called_once_with('key', '5')
        storage.get.return_value = 'Shouldnt be called'
        sleep(4)  # maybe I can just mock time
        get_results = cache.get('key')
        storage.delete.assert_called_once_with('key')

        assert None is get_results, f'Expected no results is cached, expected got {get_results}'

    def test_contains_after_expired_set(self):
        storage = Mock()
        cache = TTLCache(storage)
        cache.set('key', '5', 2)
        storage.set.assert_called_once_with('key', '5')
        storage.get.return_value = 'Shouldnt be called'
        sleep(4)  # maybe I can just mock time
        contains_results = cache.contains('key')
        storage.delete.assert_called_once_with('key')

        assert contains_results is False, f'Expected cache.contains(key) = False is cached, got {contains_results}'

    # def test_register_hook_func(self):
    #     self.fail()
    #
    # def test_clear(self):
    #     self.fail()
    #
    # def test_set(self):
    #     self.fail()
    #
    # def test_delete(self):
    #     self.fail()
    #
    # def test_size(self):
    #     self.fail()
    #
    # def test_contains(self):
    #     self.fail()
    #
    # def test_get(self):
    #     self.fail()
    #
    # def test_expire(self):
    #     self.fail()
    #
    # def test_capacity(self):
    #     self.fail()
