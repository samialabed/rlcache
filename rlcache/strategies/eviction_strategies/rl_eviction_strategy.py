import logging
from dataclasses import dataclass
from typing import Dict, List

from pandas.tests.extension.numpy_.test_numpy_nested import np
from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox, IntBox

from rlcache.backend import TTLCache, InMemoryStorage
from rlcache.observer import ObservationType
from rlcache.rl_model.converter import RLConverter
from rlcache.strategies.eviction_strategies.eviction_strategy_base import EvictionStrategy
from rlcache.utils.loggers import create_file_logger
from rlcache.utils.vocabulary import Vocabulary


@dataclass
class _IncompleteExperienceEntry(object):
    __slots__ = ['state', 'agent_action', 'ttl', 'hits', 'miss']
    state: np.ndarray
    agent_action: np.ndarray
    ttl: int
    hits: int
    miss: int

    def __repr__(self):
        return f'state: {self.state}, agent_action: {self.agent_action}, ' \
            f'ttl: {self.ttl}, hits: {self.hits}, miss: {self.miss}'


class RLEvictionStrategy(EvictionStrategy):
    def __init__(self, config: Dict[str, any], results_dir: str):
        super().__init__(config, results_dir)
        self.logger = logging.getLogger(__name__)

        # evaluation specific variables
        self.observation_seen = 0
        self.episode_reward = 0
        self.checkpoint_steps = config['checkpoint_steps']

        self.logger = logging.getLogger(__name__)
        self.reward_logger = create_file_logger(name='reward_logger', result_dir=self.result_dir)
        self.loss_logger = create_file_logger(name='loss_logger', result_dir=self.result_dir)
        self.observation_logger = create_file_logger(name='observation_logger', result_dir=self.result_dir)

        agent_config = config['agent_config']
        # state_space = all keys in the cache
        max_cache_size = config['cache_capacity']
        fields_in_state = 4  # (key + hits + miss + ttl) * keys?
        self.converter = EvictionStrategyRLConverter(max_cache_size, 0)

        # Action: index of the key to evict
        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(low=-1, high=1e9,
                                                          shape=(max_cache_size * fields_in_state,)),
                                     action_space=IntBox(low=0, high=max_cache_size, shape=(1,)))

        self._incomplete_experiences = TTLCache(InMemoryStorage())
        self._incomplete_experiences.register_hook_func(self._observe_expired_incomplete_experience)
        self.view_of_the_cache = {}  # type: Dict[int, np.ndarray]

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        # If it is a miss/hit in the incomplete experience queue: punish (based on left TTL?)
        # If it is an expiration: reward
        # otherwise: a hit update the view of the cache wth hits, if a invalidate remove from view, if write put in

        observed_key = self.converter.vocabulary.add_or_get_id(key)
        if observation_type == ObservationType.Invalidate:
            if self._incomplete_experiences.contains(key):
                # reward an eviction followed by invalidation
                stored_state = self._incomplete_experiences.get(key)  # type: _IncompleteExperienceEntry
                reward = self.converter.system_to_agent_reward(observation_type)
                state = stored_state.state
                action = stored_state.agent_action
                self._reward_agent(state, action, reward)
                self._incomplete_experiences.delete(key)
            if observed_key in self.view_of_the_cache:
                del self.view_of_the_cache[observed_key]

        elif observation_type == ObservationType.Expiration:
            assert not self._incomplete_experiences.contains(key), \
                "Detected key in incomplete experience. An expiration from cache means it key wasn't evicted."
            if observed_key in self.view_of_the_cache:
                del self.view_of_the_cache[observed_key]

        elif observation_type == ObservationType.Miss:
            if self._incomplete_experiences.contains(key):
                # Punish, a read after an eviction decision
                stored_state = self._incomplete_experiences.get(key)  # type: _IncompleteExperienceEntry
                reward = self.converter.system_to_agent_reward(observation_type)
                state = stored_state.state
                action = stored_state.agent_action
                self._reward_agent(state, action, reward)
                self._incomplete_experiences.delete(key)
            assert observed_key in self.view_of_the_cache, \
                "Attempt to read from cache but doesnt exist in the eviction strategy view"
            stored_view = self.view_of_the_cache[observed_key]
            stored_view[1] = stored_view[1] + 1  # increment hits

        elif observation_type == ObservationType.Write:
            assert self._incomplete_experiences.contains(key) is False, \
                "Write observation should be precede with miss or hit."
            self.view_of_the_cache[observed_key] = self.converter.system_to_agent_state(key=observed_key,
                                                                                        hits=0,
                                                                                        miss=0,
                                                                                        ttl=info['ttl'])

        self.observation_seen += 1
        if self.observation_seen % self.checkpoint_steps == 0:
            self.logger.info(f'Observation seen so far: {self.observation_seen}, reward so far: {self.episode_reward}')

    def _reward_agent(self, state: np.ndarray, agent_action: np.ndarray, reward: int):
        new_cache_state = self.converter.whole_cache_to_agent_state(self.view_of_the_cache)
        self.agent.observe(preprocessed_states=state,
                           actions=agent_action,
                           internals=[],
                           rewards=reward,
                           next_states=new_cache_state,
                           terminals=False)

        self.reward_logger.info(f'{reward}')

        # TODO replace with update scheduler
        loss = self.agent.update()
        if loss is not None:
            self.loss_logger.info(f'{loss[0]}')

    def _observe_expired_incomplete_experience(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        """Observe decisions taken that hasn't been observed by main cache. e.g. don't cache -> ttl up -> no miss"""
        assert observation_type == ObservationType.Expiration
        experience = info['value']  # type: _IncompleteExperienceEntry
        reward = self.converter.system_to_agent_reward(observation_type)
        state = experience.state
        action = experience.agent_action
        self._reward_agent(state, action, reward)

    def trim_cache(self, cache: TTLCache) -> str:
        # trim cache isn't called often so the operation is ok to be expensive
        state = self.converter.whole_cache_to_agent_state(self.view_of_the_cache)
        while True:
            agent_action = self.agent.get_action(state)
            if agent_action.item() not in self.view_of_the_cache:
                # punish and call it again
                self.logger.debug(f'{__name__}: agent chose action not in cache: {agent_action}')
                self._reward_agent(state, agent_action, -1)

            else:
                eviction_key = self.converter.agent_to_system_action(agent_action)
                self.logger.debug(f'{__name__}: agent chose id: {agent_action} -> key: {eviction_key}')
                break
        stored_hits_miss = self.view_of_the_cache[agent_action.item()]
        incomplete_experience = _IncompleteExperienceEntry(state, agent_action,
                                                           hits=stored_hits_miss[1],
                                                           miss=stored_hits_miss[2],
                                                           ttl=stored_hits_miss[3])
        # TODO calculate the difference between ttl left and store that
        self._incomplete_experiences.set(eviction_key, incomplete_experience, incomplete_experience.ttl)
        del self.view_of_the_cache[agent_action.item()]
        assert cache.contains(eviction_key), "Key: {} is in Eviction Cache View but not in cache.".format(eviction_key)
        cache.delete(eviction_key)
        return eviction_key


class EvictionStrategyRLConverter(RLConverter):
    def __init__(self, max_cache_size: int, values_padding_size: int):
        self.vocabulary = Vocabulary(add_pad=True, add_unk=False)
        self.logger = logging.getLogger(__name__)
        self.value_fields_vocabulary = Vocabulary(add_pad=False, add_unk=False)
        self.max_cache_size = max_cache_size
        self.values_padding_size = values_padding_size

    def agent_to_system_state(self, state: np.ndarray):
        key_encoded = state[0]
        cache_hits = state[1]
        cache_miss = state[2]
        ttl = state[3]
        key = self.vocabulary.get_name_for_id(key_encoded)
        return key, cache_hits, cache_miss, ttl

    def whole_cache_to_agent_state(self, observed_cache: Dict[int, List[int]]) -> np.ndarray:
        cache_matrix = np.array([item for item in observed_cache.values()])
        needed_padding = self.max_cache_size - cache_matrix.shape[0]
        padded_cache_matrix = np.pad(cache_matrix, ((0, needed_padding), (0, 0)), 'constant')
        return padded_cache_matrix.flatten()

    def system_to_agent_state(self, key: int, hits: int, miss: int, ttl: int) -> np.ndarray:
        # TODO get values and capacity
        return np.array([key, hits, miss, ttl])

    def system_to_agent_action(self, key: str) -> np.ndarray:
        return np.asarray(self.vocabulary.add_or_get_id(key))

    def agent_to_system_action(self, actions: np.ndarray, **kwargs) -> str:
        return self.vocabulary.get_name_for_id(actions.item())

    def system_to_agent_reward(self, observation_type: ObservationType) -> int:
        return -1 if observation_type == ObservationType.Miss else 1

    def _extract_and_encode_values(self, values: Dict[str, any]) -> np.ndarray:
        extracted = np.zeros(self.values_padding_size, dtype='int32')
        assert len(extracted) == len(values)
        for k, v in values.items():
            k_encoded = self.value_fields_vocabulary.add_or_get_id(k)
            extracted[k_encoded] = self.vocabulary.add_or_get_id(v)
        return extracted

    def _recover_encoded_values(self, values: np.ndarray) -> Dict[str, any]:
        output = {}
        for idx in range(len(values)):
            decoded_value = self.vocabulary.get_name_for_id(values[idx])
            decoded_key = self.value_fields_vocabulary.get_name_for_id(idx)
            output[decoded_key] = decoded_value
        assert len(output) == len(values)
        return output
