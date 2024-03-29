import logging
from typing import Dict

import time
from rlgraph.agents import Agent
from rlgraph.spaces import Dict as RLDict, IntBox
from rlgraph.spaces import FloatBox

from rlcache.backend import InMemoryStorage, TTLCache
from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObservationType
from rlcache.strategies.base_strategy import BaseStrategy
from rlcache.strategies.multi_task.rl_multi_task_cache_strategy_state import MultiTaskAgentSystemState, \
    MultiTaskAgentObservedExperience
from rlcache.utils.loggers import create_file_logger
from rlcache.utils.vocabulary import Vocabulary


class RLMultiTasksStrategy(BaseStrategy):
    """RL driven multi task strategy - Caching, eviction, and ttl estimation."""

    def __init__(self, config: Dict[str, any], result_dir: str, cache_stats: CacheInformation):
        super().__init__(config, result_dir, cache_stats)
        self.supported_observations = {ObservationType.Hit,
                                       ObservationType.Miss,
                                       ObservationType.Invalidate}

        # evaluation specific variables
        self.observation_seen = 0
        self.cum_reward = 0
        self.checkpoint_steps = config['checkpoint_steps']

        self._incomplete_experiences = TTLCache(InMemoryStorage())
        self._incomplete_experiences.expired_entry_callback(self._observe_expiry_eviction)
        self.non_terminal_observations = {ObservationType.EvictionPolicy, ObservationType.Expiration}

        agent_config = config['agent_config']
        self.maximum_ttl = config['max_ttl']

        fields_in_state = len(MultiTaskAgentSystemState.__slots__)

        action_space = RLDict({
            'ttl': IntBox(low=0, high=self.maximum_ttl),
            'eviction': IntBox(low=0, high=2)
        })

        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(shape=(fields_in_state,)),
                                     action_space=action_space)

        # TODO refactor into common RL interface for all strategies
        self.logger = logging.getLogger(__name__)
        name = 'rl_multi_strategy'
        self.reward_logger = create_file_logger(name=f'{name}_reward_logger', result_dir=self.result_dir)
        self.loss_logger = create_file_logger(name=f'{name}_loss_logger', result_dir=self.result_dir)
        self.ttl_logger = create_file_logger(name=f'{name}_ttl_logger', result_dir=self.result_dir)
        self.observation_logger = create_file_logger(name=f'{name}_observation_logger', result_dir=self.result_dir)
        self.performance_logger = create_file_logger(name=f'{name}_performance_logger', result_dir=self.result_dir)
        self.key_vocab = Vocabulary()

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        observed_experience = self._incomplete_experiences.get(key)

        if observed_experience is None:
            return  # haven't had to make a decision on it

        current_time = time.time()
        stored_state = observed_experience.state

        stored_state.step_code = observation_type.value
        stored_state.cache_utility = self.cache_stats.cache_utility

        if observation_type == ObservationType.Hit:
            stored_state.hit_count += 1
        else:
            # Include eviction, invalidation, and miss
            estimated_ttl = observed_experience.agent_action['ttl'].item()
            first_observation_time = observed_experience.observation_time
            real_ttl = current_time - first_observation_time
            # log the difference between the estimated ttl and real ttl
            self.ttl_logger.info(
                f'{self.episode_num},{observation_type.name},{key},{estimated_ttl},{real_ttl},{stored_state.hit_count}')
            self._incomplete_experiences.delete(key)

        self.reward_agent(observation_type, observed_experience)

        self.observation_seen += 1
        if self.observation_seen % self.checkpoint_steps == 0:
            self.logger.info(
                f'Observation seen so far: {self.observation_seen}, reward so far: {self.cum_reward}')
        if observation_type not in self.non_terminal_observations:
            self.observation_logger.info(f'{key},{observation_type}')

    def trim_cache(self, cache: TTLCache):
        # trim cache isn't called often so the operation is ok to be expensive
        # produce an action on the whole cache
        keys_to_evict = []

        for (key, stored_experience) in list(self._incomplete_experiences.items()):
            action = self.agent.get_action(stored_experience.state.to_numpy())['eviction']
            evict = (action.flatten() == 1).item()
            if evict:
                cache.delete(key)
                keys_to_evict.append(key)
            # update stored value for eviction action
            stored_experience.agent_action['eviction'] = action
            stored_experience.manual_eviction = True

        if len(keys_to_evict) == 0:
            self.logger.error('trim_cache No keys were evicted.')

        return keys_to_evict

    def should_cache(self, key: str, values: Dict[str, str], ttl: int, operation_type: OperationType) -> bool:
        # cache objects that have TTL more than 1 second (maybe make this configurable?)
        return ttl > 10

    def estimate_ttl(self, key: str, values: Dict[str, any], operation_type: OperationType) -> float:
        # TODO check if it is in the observed queue

        observation_time = time.time()
        encoded_key = self.key_vocab.add_or_get_id(key)
        cache_utility = self.cache_stats.cache_utility

        state = MultiTaskAgentSystemState(encoded_key=encoded_key,
                                          hit_count=0,
                                          ttl=0,
                                          step_code=0,
                                          cache_utility=cache_utility,
                                          operation_type=operation_type.value)

        state_as_numpy = state.to_numpy()
        agent_action = self.agent.get_action(state_as_numpy)
        action = agent_action['ttl'].item()

        incomplete_experience = MultiTaskAgentObservedExperience(state=state,
                                                                 agent_action=agent_action,
                                                                 starting_state=state.copy(),
                                                                 observation_time=observation_time,
                                                                 manual_eviction=False)
        self._incomplete_experiences.set(key, incomplete_experience, self.maximum_ttl)

        return action

    def reward_agent(self, observation_type: ObservationType, experience: MultiTaskAgentObservedExperience) -> int:
        # reward more utilisation of the cache capacity given more hits
        final_state = experience.state

        # difference_in_ttl = -abs((experience.agent_action.item() + 1) / min(real_ttl, self.maximum_ttl))
        reward = 0
        terminal = False

        if observation_type == observation_type.Invalidate and (
                experience.agent_action['ttl'] < 10 or (experience.agent_action['eviction'].flatten() == 1).item()):
            # if evicted or not cached, followed by an invalidate
            reward = 10
            terminal = True
        elif observation_type == observation_type.Invalidate or observation_type == observation_type.Miss:
            reward = -10
            terminal = True
        elif observation_type == ObservationType.Hit:
            terminal = False
            reward = 1

        if experience.manual_eviction:
            if observation_type == observation_type.Expiration:
                reward = -10
                terminal = True
            if observation_type == observation_type.Hit:
                reward = 2

            self.performance_metric_for_eviction(experience, observation_type)

        self.agent.observe(preprocessed_states=experience.starting_state.to_numpy(),
                           actions=experience.agent_action,
                           internals=[],
                           rewards=terminal,
                           next_states=final_state.to_numpy(),
                           terminals=True)

        self.cum_reward += reward
        self.reward_logger.info(f'{self.episode_num},{reward}')
        # TODO use self.agent.update_schedule to decide when to call update
        loss = self.agent.update()
        if loss is not None:
            self.loss_logger.info(f'{self.episode_num},{loss[0]}')

        return reward

    def _observe_expiry_eviction(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        """Observe decisions taken that hasn't been observed by main cache. e.g. don't cache -> ttl up -> no miss"""
        self.observation_logger.info(f'{self.episode_num},{key},{observation_type}')
        experience = info['value']  # type: MultiTaskAgentObservedExperience
        self.ttl_logger.info(
            f'{self.episode_num},{observation_type.name},{key},{experience.agent_action["ttl"].item()},'
            f'{experience.agent_action["ttl"].item()},{experience.state.hit_count}')
        experience.state.step_code = observation_type.value

        self.reward_agent(observation_type, experience)

    def performance_metric_for_eviction(self,
                                        stored_experience: MultiTaskAgentObservedExperience,
                                        observation_type: ObservationType) -> int:
        should_evict = (stored_experience.agent_action['eviction'].flatten() == 1).item()

        if observation_type == ObservationType.Expiration:
            if should_evict:
                # reward if should evict didn't observe any follow up miss
                self.performance_logger.info(f'{self.episode_num},TrueEvict')
            # else didn't evict
            else:
                # reward for not evicting a key that received more hits.
                # or 0 if it didn't evict but also didn't get any hits
                gain_for_not_evicting = stored_experience.state.hit_count - stored_experience.starting_state.hit_count
                if gain_for_not_evicting > 0:
                    self.performance_logger.info(f'{self.episode_num},TrueMiss')
                else:
                    self.performance_logger.info(f'{self.episode_num},MissEvict')

                return gain_for_not_evicting

        elif observation_type == ObservationType.Invalidate:
            # Set/Delete, remove entry from the cache.
            # reward an eviction followed by invalidation.
            if should_evict:
                self.performance_logger.info(f'{self.episode_num},TrueEvict')
            else:
                # punish not evicting a key that got invalidated after.
                self.performance_logger.info(f'{self.episode_num},MissEvict')

        elif observation_type == ObservationType.Miss:
            if should_evict:
                self.performance_logger.info(f'{self.episode_num},FalseEvict')
            # Miss after making an eviction decision
            # Punish, a read after an eviction decision

    def close(self):
        for (k, v) in list(self._incomplete_experiences.items()):
            self.ttl_logger.info(
                f'{self.episode_num},{ObservationType.EndOfEpisode.name},{k},{v.agent_action["ttl"].item()},'
                f'{v.agent_action["ttl"].item()},{v.state.hit_count}')
            self.performance_logger.info(f'{self.episode_num},TrueMiss')
        super().close()
        self._incomplete_experiences.clear()
        try:
            self.agent.reset()
        except Exception as e:
            pass
