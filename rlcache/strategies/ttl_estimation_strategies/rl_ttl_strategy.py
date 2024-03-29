import logging
from typing import Dict

import time
from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox

from rlcache.backend import TTLCache, InMemoryStorage
from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObservationType
from rlcache.strategies.ttl_estimation_strategies.base_ttl_strategy import TtlStrategy
from rlcache.strategies.ttl_estimation_strategies.rl_ttl_state import TTLAgentObservedExperience, \
    TTLAgentSystemState
from rlcache.utils.loggers import create_file_logger
from rlcache.utils.vocabulary import Vocabulary


class RLTtlStrategy(TtlStrategy):
    """RL driven TTL estimation strategy."""

    def __init__(self, config: Dict[str, any], result_dir: str, cache_stats: CacheInformation):
        super().__init__(config, result_dir, cache_stats)
        self.observation_seen = 0
        self.cum_reward = 0
        self.checkpoint_steps = config['checkpoint_steps']

        self._incomplete_experiences = TTLCache(InMemoryStorage())
        self._incomplete_experiences.expired_entry_callback(self._observe_expiry_eviction)
        self.non_terminal_observations = {ObservationType.EvictionPolicy, ObservationType.Expiration}
        agent_config = config['agent_config']
        self.maximum_ttl = config['max_ttl']
        self.experimental_reward = config.get('experimental_reward', False)
        fields_in_state = len(TTLAgentSystemState.__slots__)
        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(shape=(fields_in_state,)),
                                     action_space=FloatBox(low=0, high=self.maximum_ttl, shape=(1,)))

        # TODO refactor into common RL interface for all strategies
        self.logger = logging.getLogger(__name__)
        name = 'rl_ttl_strategy'
        self.reward_logger = create_file_logger(name=f'{name}_reward_logger', result_dir=self.result_dir)
        self.loss_logger = create_file_logger(name=f'{name}_loss_logger', result_dir=self.result_dir)
        self.ttl_logger = create_file_logger(name=f'{name}_ttl_logger', result_dir=self.result_dir)
        self.observation_logger = create_file_logger(name=f'{name}_observation_logger', result_dir=self.result_dir)
        self.key_vocab = Vocabulary()
        self.errors = create_file_logger(name=f'{name}_error_logger', result_dir=self.result_dir)

    def estimate_ttl(self, key: str,
                     values: Dict[str, any],
                     operation_type: OperationType) -> float:
        observation_time = time.time()
        encoded_key = self.key_vocab.add_or_get_id(key)
        cache_utility = self.cache_stats.cache_utility

        state = TTLAgentSystemState(encoded_key=encoded_key,
                                    hit_count=0,
                                    step_code=0,
                                    cache_utility=cache_utility,
                                    operation_type=operation_type.value)

        state_as_numpy = state.to_numpy()
        agent_action = self.agent.get_action(state_as_numpy)
        action = agent_action.item()

        incomplete_experience = TTLAgentObservedExperience(state=state,
                                                           agent_action=agent_action,
                                                           starting_state=state.copy(),
                                                           observation_time=observation_time)
        self._incomplete_experiences.set(key, incomplete_experience, self.maximum_ttl)

        return action

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        observed_experience = self._incomplete_experiences.get(key)

        if observed_experience is None or observation_type == ObservationType.Expiration:
            return  # haven't had to make a decision on it

        current_time = time.time()
        stored_state = observed_experience.state
        if observation_type == ObservationType.Hit:
            stored_state.hit_count += 1
        # elif observation_type in self.non_terminal_observations:
        #     # it was evicted by another policy don't attempt to learn stuff from this
        #     pass

        estimated_ttl = observed_experience.agent_action.item()
        first_observation_time = observed_experience.observation_time
        real_ttl = current_time - first_observation_time
        stored_state.step_code = observation_type.value
        stored_state.cache_utility = self.cache_stats.cache_utility
        self.reward_agent(observation_type, observed_experience, real_ttl)

        if observation_type != ObservationType.Hit:
            self.ttl_logger.info(
                f'{self.episode_num},{observation_type.name},{key},{estimated_ttl},{real_ttl},{stored_state.hit_count}')
            self._incomplete_experiences.delete(key)

        self.observation_seen += 1
        if self.observation_seen % self.checkpoint_steps == 0:
            self.logger.info(
                f'Observation seen so far: {self.observation_seen}, reward so far: {self.cum_reward}')
        if observation_type not in self.non_terminal_observations:
            self.observation_logger.info(f'{key},{observation_type}')

    def _observe_expiry_eviction(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        """Observe decisions taken that hasn't been observed by main cache. e.g. don't cache -> ttl up -> no miss"""
        self.observation_logger.info(f'{self.episode_num},{key},{observation_type}')
        experience = info['value']  # type: TTLAgentObservedExperience
        self.ttl_logger.info(f'{self.episode_num},{observation_type.name},{key},{experience.agent_action.item()},'
                             f'{experience.agent_action.item()},{experience.state.hit_count}')
        experience.state.step_code = observation_type.value

        self.reward_agent(observation_type, experience, experience.agent_action.item())

    def reward_agent(self, observation_type: ObservationType,
                     experience: TTLAgentObservedExperience,
                     real_ttl: time) -> int:
        # reward more utilisation of the cache capacity given more hits
        final_state = experience.state

        difference_in_ttl = -abs((experience.agent_action.item() + 1) / max(min(real_ttl, self.maximum_ttl), 1))
        # reward = final_state.hit_count - abs(difference_in_ttl * self.cache_stats.cache_utility)

        if observation_type == ObservationType.Hit:
            reward = 1
            terminal = False
        # elif observation_type == ObservationType.EvictionPolicy:
        #     reward = 0
        #     terminal = True
        else:
            reward = difference_in_ttl
            if abs(difference_in_ttl) < 10:
                reward = 10
            terminal = True
            self.logger.debug(f'Hits: {final_state.hit_count}, ttl diff: {difference_in_ttl}, Reward: {reward}')

        self.agent.observe(preprocessed_states=experience.starting_state.to_numpy(),
                           actions=experience.agent_action,
                           internals=[],
                           rewards=reward,
                           next_states=final_state.to_numpy(),
                           terminals=terminal)

        self.cum_reward += reward
        self.reward_logger.info(f'{self.episode_num},{reward}')
        loss = self.agent.update()
        if loss is not None:
            self.loss_logger.info(f'{self.episode_num},{loss[0]}')

        return reward

    def close(self):
        super().close()
        for (k, v) in list(self._incomplete_experiences.items()):
            self.ttl_logger.info(f'{self.episode_num},{ObservationType.EndOfEpisode.name},{k},{v.agent_action.item()},'
                                 f'{v.agent_action.item()},{v.state.hit_count}')

        self._incomplete_experiences.clear()
        try:
            self.agent.reset()
        except Exception as e:
            self.errors.info(e)
