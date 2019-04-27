import logging
import time
from typing import Dict, List

from pandas.tests.extension.numpy_.test_numpy_nested import np
from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox, IntBox

from rlcache.backend import TTLCache, InMemoryStorage
from rlcache.observer import ObservationType
from rlcache.strategies.eviction_strategies.base_eviction_strategy import EvictionStrategy
from rlcache.strategies.eviction_strategies.rl_eviction_agent_state import EvictionAgentSystemState, \
    EvictionAgentIncompleteExperienceEntry
from rlcache.strategies.eviction_strategies.rl_eviction_strategy_converter import EvictionStrategyRLConverter
from rlcache.utils.loggers import create_file_logger


class RLEvictionStrategy(EvictionStrategy):
    def __init__(self, config: Dict[str, any], results_dir: str):
        super().__init__(config, results_dir)
        # evaluation specific variables
        self.observation_seen = 0
        self.episode_reward = 0
        self.checkpoint_steps = config['checkpoint_steps']

        self.logger = logging.getLogger(__name__)
        self.reward_logger = create_file_logger(name=f'{__name__}_reward_logger', result_dir=self.result_dir)
        self.loss_logger = create_file_logger(name=f'{__name__}_loss_logger', result_dir=self.result_dir)
        self.observation_logger = create_file_logger(name=f'{__name__}_observation_logger', result_dir=self.result_dir)

        agent_config = config['agent_config']
        fields_in_state = len(EvictionAgentSystemState.__slots__)
        self.converter = EvictionStrategyRLConverter(self.result_dir)

        # State: fields to observe in question
        # Action: to evict or not that key
        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(low=0, high=1e9, shape=(fields_in_state,)),
                                     action_space=IntBox(low=0, high=2))

        self._incomplete_experiences = TTLCache(InMemoryStorage())
        self._incomplete_experiences.register_hook_func(self._observe_expired_incomplete_experience)

        self.view_of_the_cache = {}  # type: Dict[str, EvictionAgentSystemState]
        self._end_episode_observation = {ObservationType.Invalidate, ObservationType.Miss, ObservationType.Expiration}

    def trim_cache(self, cache: TTLCache) -> List[str]:
        # trim cache isn't called often so the operation is ok to be expensive
        # produce an action on the whole cache
        keys_to_evict = []
        keys_to_not_evict = []

        while True:
            for (key, agent_system_state) in self.view_of_the_cache.items():
                agent_action = self.agent.get_action(agent_system_state.to_numpy())
                should_evict = self.converter.agent_to_system_action(agent_action)
                self.logger.debug(f'{__name__}: {key} should be evicted: {should_evict}')
                stored_system_state = self.view_of_the_cache[key]
                incomplete_experience = EvictionAgentIncompleteExperienceEntry(stored_system_state,
                                                                               agent_action,
                                                                               stored_system_state.copy())
                ttl_left = stored_system_state.expiry_time - time.time()  # the ttl left on this key
                self._incomplete_experiences.set(key=key,
                                                 values=incomplete_experience,
                                                 ttl=ttl_left)
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
        self.observation_logger.info(f'{key},{observation_type}')

        observed_key = self.converter.vocabulary.add_or_get_id(key)
        stored_experience = self._incomplete_experiences.get(key)
        if observation_type == ObservationType.Write:
            # New item to write into cache view and observe.
            assert stored_experience is None, \
                "Write observation should be precede with end of an episode operation."
            ttl = info['ttl']
            expiry_time = time.time() + ttl
            self.view_of_the_cache[key] = EvictionAgentSystemState(encoded_key=observed_key,
                                                                   ttl=ttl,
                                                                   expiry_time=expiry_time,
                                                                   hit_count=0,
                                                                   miss_count=0,
                                                                   invalidate_count=0,
                                                                   expiration_count=0)

        elif observation_type == ObservationType.Hit:
            # Cache hit, update the hit record of this key in the cache
            stored_view = self.view_of_the_cache[key]
            stored_view.hit_count += 1

        elif observation_type in self._end_episode_observation:
            if stored_experience:
                reward = self.converter.system_to_agent_reward(stored_experience, observation_type)
                state = stored_experience.state
                action = stored_experience.agent_action
                new_state = state.copy()
                if observation_type == ObservationType.Invalidate:
                    new_state.invalidate_count += 1
                elif observation_type == ObservationType.Miss:
                    new_state.miss_count += 1

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
        self.observation_logger.info(f'{key},{observation_type}')

        experience = info['value']  # type: EvictionAgentIncompleteExperienceEntry
        reward = self.converter.system_to_agent_reward(experience, observation_type)
        starting_state = experience.starting_state
        action = experience.agent_action
        new_state = experience.state.copy()
        new_state.expiration_count += 1

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
        self.reward_logger.info(f'{reward}')

        loss = self.agent.update()
        if loss is not None:
            self.loss_logger.info(f'{loss[0]}')
