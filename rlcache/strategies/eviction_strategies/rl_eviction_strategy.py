import logging
from typing import Dict, List

import time
from pandas.tests.extension.numpy_.test_numpy_nested import np

from rlcache.backend import TTLCache, InMemoryStorage
from rlcache.cache_constants import CacheInformation
from rlcache.observer import ObservationType
from rlcache.strategies.eviction_strategies.base_eviction_strategy import EvictionStrategy
from rlcache.strategies.eviction_strategies.rl_eviction_state import EvictionAgentSystemState, \
    EvictionAgentIncompleteExperienceEntry
from rlcache.strategies.eviction_strategies.rl_eviction_state_converter import EvictionStrategyRLConverter
from rlcache.utils.loggers import create_file_logger
from rlcache.utils.vocabulary import Vocabulary
from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox, IntBox


class RLEvictionStrategy(EvictionStrategy):
    def __init__(self, config: Dict[str, any], result_dir: str, cache_stats: CacheInformation):
        super().__init__(config, result_dir, cache_stats)
        # evaluation specific variables
        self.observation_seen = 0
        self.episode_reward = 0
        self.checkpoint_steps = config['checkpoint_steps']

        self._incomplete_experiences = TTLCache(InMemoryStorage())
        self._incomplete_experiences.expired_entry_callback(self._observe_expired_incomplete_experience)
        self.view_of_the_cache = {}  # type: Dict[str, Dict[str, any]]
        self._end_episode_observation = {ObservationType.Invalidate, ObservationType.Miss, ObservationType.Expiration}

        # TODO refactor into common RL interface for all strategies
        # Agent configuration (can be shared with others)
        agent_config = config['agent_config']
        fields_in_state = len(EvictionAgentSystemState.__slots__)
        self.converter = EvictionStrategyRLConverter(self.result_dir)

        # State: fields to observe in question
        # Action: to evict or not that key
        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(shape=(fields_in_state,)),
                                     action_space=IntBox(low=0, high=2))

        self.logger = logging.getLogger(__name__)
        name = 'rl_eviction_strategy'
        self.reward_logger = create_file_logger(name=f'{name}_reward_logger', result_dir=self.result_dir)
        self.loss_logger = create_file_logger(name=f'{name}_loss_logger', result_dir=self.result_dir)
        self.observation_logger = create_file_logger(name=f'{name}_observation_logger', result_dir=self.result_dir)
        self.key_vocab = Vocabulary()

    def trim_cache(self, cache: TTLCache) -> List[str]:
        # trim cache isn't called often so the operation is ok to be expensive
        # produce an action on the whole cache
        keys_to_evict = []
        keys_to_not_evict = []

        while True:
            for (key, cached_key) in self.view_of_the_cache.items():
                agent_system_state = cached_key['state']

                agent_action = self.agent.get_action(agent_system_state.to_numpy())
                should_evict = self.converter.agent_to_system_action(agent_action)

                decision_time = time.time()
                incomplete_experience = EvictionAgentIncompleteExperienceEntry(agent_system_state,
                                                                               agent_action,
                                                                               agent_system_state.copy(),
                                                                               decision_time)

                # observe the key for only the ttl period that is left for this key
                ttl_left = (cached_key['observation_time'] + agent_system_state.ttl) - decision_time
                self._incomplete_experiences.set(key=key, values=incomplete_experience, ttl=ttl_left)

                if should_evict:
                    keys_to_evict.append(key)
                else:
                    keys_to_not_evict.append(key)
            if len(keys_to_evict) > 0:
                break
            else:
                # didn't make any eviction decisions, remove observed stuff from memory and try again.
                # this won't happen when the cache is large enough.
                for key in keys_to_not_evict:
                    self._incomplete_experiences.delete(key)
                self.logger.error('No keys were chosen to be evicted. Retrying.')

        for key in keys_to_evict:
            # race condition: while in this loop a key expires and hit the observer pattern
            if key in self.view_of_the_cache:
                del self.view_of_the_cache[key]
            if not cache.contains(key):
                # race condition, clean up and move on
                self._incomplete_experiences.delete(key)
            cache.delete(key)

        return keys_to_evict

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        self.observation_logger.info(f'{self.episode_num},{key},{observation_type}')

        observed_key = self.converter.vocabulary.add_or_get_id(key)
        stored_experience = self._incomplete_experiences.get(key)
        if observation_type == ObservationType.Write:
            # New item to write into cache view and observe.
            assert stored_experience is None, \
                "Write observation should be precede with end of an episode operation."
            ttl = info['ttl']
            observation_time = time.time()
            self.view_of_the_cache[key] = {'state': EvictionAgentSystemState(encoded_key=observed_key,
                                                                             ttl=ttl,
                                                                             hit_count=0,
                                                                             step_code=observation_type.value),
                                           'observation_time': observation_time}

        elif observation_type == ObservationType.Hit:
            # Cache hit, update the hit record of this key in the cache
            stored_view = self.view_of_the_cache[key]['state']
            stored_view.hit_count += 1

        elif observation_type in self._end_episode_observation:
            if stored_experience:
                reward = self.converter.system_to_agent_reward(stored_experience, observation_type, self.episode_num)
                state = stored_experience.state
                action = stored_experience.agent_action
                new_state = state.copy()
                new_state.step_code = observation_type.value

                self._reward_agent(state.to_numpy(), new_state.to_numpy(), action, reward)
                self._incomplete_experiences.delete(key)

            if key in self.view_of_the_cache:
                del self.view_of_the_cache[key]

        self.observation_seen += 1
        if self.observation_seen % self.checkpoint_steps == 0:
            self.logger.info(f'Observation seen so far: {self.observation_seen}, reward so far: {self.episode_reward}')

    def _observe_expired_incomplete_experience(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        """Observe decisions taken that hasn't been observed by main cache. e.g. don't cache -> ttl up -> no miss"""
        assert observation_type == ObservationType.Expiration
        self.observation_logger.info(f'{self.episode_num},{key},{observation_type}')

        experience = info['value']  # type: EvictionAgentIncompleteExperienceEntry
        reward = self.converter.system_to_agent_reward(experience, observation_type, self.episode_num)
        starting_state = experience.starting_state
        action = experience.agent_action
        new_state = experience.state.copy()
        new_state.step_code = observation_type.value

        self._reward_agent(starting_state.to_numpy(), new_state.to_numpy(), action, reward)

    def _reward_agent(self,
                      state: np.ndarray,
                      new_state: np.ndarray,
                      agent_action: np.ndarray,
                      reward: int):
        self.agent.observe(preprocessed_states=state,
                           actions=agent_action,
                           internals=[],
                           rewards=reward,
                           next_states=new_state,
                           terminals=False)
        self.reward_logger.info(f'{self.episode_num},{reward}')

        loss = self.agent.update()
        if loss is not None:
            self.loss_logger.info(f'{self.episode_num},{loss[0]}')

    def close(self):
        super().close()
        self._incomplete_experiences.clear()
        self.agent.reset()
