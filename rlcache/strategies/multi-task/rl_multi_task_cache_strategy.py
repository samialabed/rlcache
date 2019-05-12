import logging
from typing import Dict

import time

from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObservationType
from rlcache.strategies.ttl_estimation_strategies.base_ttl_strategy import TtlStrategy
from rlcache.strategies.ttl_estimation_strategies.rl_ttl_state import TTLAgentObservedExperience, \
    TTLAgentSystemState
from rlcache.utils.loggers import create_file_logger
from rlcache.utils.vocabulary import Vocabulary
from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox
from rlgraph.spaces import Dict as RLDict


class RLMultiTasksStrategy(TtlStrategy):
    """RL driven multi task strategy - Caching, eviction, and ttl estimation."""

    def __init__(self, config: Dict[str, any], result_dir: str):
        super().__init__(config, result_dir)
        self.observation_seen = 0
        self.cum_reward = 0
        self.checkpoint_steps = config['checkpoint_steps']

        self.observed_keys = {}  # type: Dict[str, TTLAgentObservedExperience]

        agent_config = config['agent_config']
        maximum_ttl = config['max_ttl']
        self.experimental_reward = config.get('experimental_reward', False)

        fields_in_state = len(TTLAgentSystemState.__slots__)

        action_space = RLDict({
            'should_cache': FloatBox(low=0, high=2, shape=(1,)),
            'ttl': FloatBox(low=0, high=maximum_ttl, shape=(1,)),
            'eviction': FloatBox(low=0, high=2, shape=(1, ))

        }, add_batch_rank=True)

        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(shape=(fields_in_state,)),
                                     action_space=FloatBox(low=0, high=maximum_ttl, shape=(1,)))

        # TODO refactor into common RL interface for all strategies
        self.logger = logging.getLogger(__name__)
        name = 'rl_ttl_strategy'
        self.reward_logger = create_file_logger(name=f'{name}_reward_logger', result_dir=self.result_dir)
        self.loss_logger = create_file_logger(name=f'{name}_loss_logger', result_dir=self.result_dir)
        self.ttl_logger = create_file_logger(name=f'{name}_ttl_logger', result_dir=self.result_dir)
        self.key_vocab = Vocabulary()

    def observe(self, key: str, observation_type: ObservationType, info: Dict[str, any]):
        if key not in self.observed_keys:
            self.logger.error(f'key: {key} is has not been observed. observation_type:{observation_type.name}')
            return  # haven't had to make a decision on it

        current_time = time.time()
        observed_experience = self.observed_keys[key]
        stored_state = observed_experience.state
        stored_agent_action = observed_experience.agent_action
        first_observation_time = observed_experience.observation_time

        if observation_type == ObservationType.Hit:
            stored_state.hit_count += 1

        # TODO consider allowing to ignore interference from other agents
        # elif observation_type == ObservationType.EvictionPolicy:
        # it was evicted by another policy don't attempt to learn stuff from this?

        else:
            # Include eviction, invalidation, and miss
            estimated_ttl = stored_agent_action.item()
            real_ttl = current_time - first_observation_time
            stored_state.step_code = observation_type.value
            stored_state.cache_utility = info['cache_utility']

            # log the difference between the estimated ttl and real ttl
            self.ttl_logger.info(f'{observation_type.name},{key},{estimated_ttl},{real_ttl},{stored_state.hit_count}')
            self.reward_agent(observation_type, observed_experience, real_ttl)
            del self.observed_keys[key]

        self.observation_seen += 1
        if self.observation_seen % self.checkpoint_steps == 0:
            self.logger.info(
                f'Observation seen so far: {self.observation_seen}, reward so far: {self.cum_reward}')

    def estimate_ttl(self, key: str,
                     values: Dict[str, any],
                     operation_type: OperationType,
                     cache_information: CacheInformation) -> float:
        observation_time = time.time()
        encoded_key = self.key_vocab.add_or_get_id(key)
        cache_utility = cache_information.cache_utility

        state = TTLAgentSystemState(encoded_key=encoded_key,
                                    hit_count=0,
                                    step_code=0,
                                    cache_utility=cache_utility,
                                    operation_type=operation_type.value)

        state_as_numpy = state.to_numpy()
        agent_action = self.agent.get_action(state_as_numpy)
        action = agent_action.item()

        self.observed_keys[key] = TTLAgentObservedExperience(state=state,
                                                             agent_action=agent_action,
                                                             starting_state=state.copy(),
                                                             observation_time=observation_time)

        return action

    def reward_agent(self, observation_type: ObservationType,
                     experience: TTLAgentObservedExperience,
                     real_ttl: time) -> int:
        # reward more utilisation of the cache capacity given more hits
        final_state = experience.state

        difference_in_ttl = real_ttl - experience.agent_action.item()
        reward = (final_state.hit_count + difference_in_ttl) * (1 + final_state.cache_utility)

        # TODO Test various reward functions
        if self.experimental_reward:
            if observation_type == ObservationType.Invalidate:
                # if invalidation reward 0 for good invalidation, and negative value for how far away it was.
                reward = difference_in_ttl
            else:
                # miss\expire\eviction. reward 0 for no hits?
                reward = final_state.hit_count * (1 + final_state.cache_utility)

        self.logger.debug(f'Hits: {final_state.hit_count}, ttl diff: {difference_in_ttl}, Reward: {reward}')

        self.agent.observe(preprocessed_states=experience.starting_state.to_numpy(),
                           actions=experience.agent_action,
                           internals=[],
                           rewards=reward,
                           next_states=final_state.to_numpy(),
                           terminals=False)

        self.cum_reward += reward
        self.reward_logger.info(f'{reward}')
        # TODO use self.agent.update_schedule to decide when to call update
        loss = self.agent.update()
        if loss is not None:
            self.loss_logger.info(f'{loss[0]}')

        return reward
