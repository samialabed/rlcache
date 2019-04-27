import logging
import time
from typing import Dict

from rlgraph.agents import Agent
from rlgraph.spaces import FloatBox

from rlcache.cache_constants import OperationType, CacheInformation
from rlcache.observer import ObservationType
from rlcache.strategies.ttl_estimation_strategies.base_ttl_strategy import TtlStrategy
from rlcache.strategies.ttl_estimation_strategies.rl_ttl_agent_state import TTLAgentObservedExperience, \
    TTLAgentSystemState
from rlcache.utils.loggers import create_file_logger
from rlcache.utils.vocabulary import Vocabulary


class RLTtlStrategy(TtlStrategy):
    """RL driven TTL estimation strategy."""

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
        self.agent = Agent.from_spec(agent_config,
                                     state_space=FloatBox(shape=(fields_in_state,)),
                                     action_space=FloatBox(low=0, high=maximum_ttl, shape=(1,)))

        # TODO refactor into common RL interface for all strategies
        self.logger = logging.getLogger(__name__)
        self.reward_logger = create_file_logger(name=f'{__name__}_reward_logger', result_dir=self.result_dir)
        self.loss_logger = create_file_logger(name=f'{__name__}_loss_logger', result_dir=self.result_dir)
        self.observation_logger = create_file_logger(name=f'{__name__}_observation_logger', result_dir=self.result_dir)
        self.ttl_logger = create_file_logger(name=f'{__name__}_ttl_logger', result_dir=self.result_dir)
        self.key_vocab = Vocabulary()

    def observe(self, key: str, observation_type: ObservationType, *args, **kwargs):
        if key not in self.observed_keys:
            self.logger.error(f'key: {key} is has not been observed in {__name__}, '
                              f'observation_type:{observation_type.name}')
            return  # haven't had to make a decision on it

        current_time = time.time()
        observed_experience = self.observed_keys[key]
        stored_state = observed_experience.state
        stored_agent_action = observed_experience.agent_action

        if observation_type == ObservationType.Hit:
            stored_state.hit_count += 1

        # TODO consider allowing to ignore interference from other agents
        # elif observation_type == ObservationType.EvictionPolicy:
        # it was evicted by another policy don't attempt to learn stuff from this?

        else:
            # Include eviction, invalidation, and miss
            first_observation_time = stored_state.observation_time
            estimated_ttl = stored_agent_action.item()
            real_ttl = current_time - first_observation_time
            stored_state.step_code = observation_type.value

            # log the difference between the estimated ttl and real ttl
            self.ttl_logger.info(f'{observation_type.name},{key},{estimated_ttl},{real_ttl},{stored_state.hit_count}')

            reward = self.system_reward(observation_type, observed_experience, real_ttl)

            self.agent.observe(preprocessed_states=observed_experience.starting_state.to_numpy(),
                               actions=stored_agent_action,
                               internals=[],
                               rewards=reward,
                               next_states=stored_state.to_numpy(),
                               terminals=False)

            self.cum_reward += reward
            self.reward_logger.info(f'{reward}')
            del self.observed_keys[key]
            # TODO use self.agent.update_schedule to decide when to call update
            loss = self.agent.update()
            if loss is not None:
                self.loss_logger.info(f'{loss[0]}')
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
        cache_utility = cache_information.size / cache_information.max_capacity()

        state = TTLAgentSystemState(encoded_key=encoded_key,
                                    observation_time=observation_time,
                                    hit_count=0,
                                    step_code=0,
                                    cache_utility=cache_utility,
                                    operation_type=operation_type.value)

        agent_action = self.agent.get_action(state.to_numpy())
        action = agent_action.item()

        self.observed_keys[key] = TTLAgentObservedExperience(state=state,
                                                             agent_action=agent_action,
                                                             starting_state=state.copy())

        return action

    def system_reward(self, observation_type: ObservationType,
                      experience: TTLAgentObservedExperience,
                      real_ttl: time) -> int:
        # reward more utilisation of the cache capacity given more hits
        state = experience.state
        difference_in_ttl = real_ttl - experience.agent_action.item()
        reward = state.hit_count * (1 + state.cache_utility)

        # TODO Test various reward functions
        if self.experimental_reward:
            if observation_type == ObservationType.Invalidate:
                # if invalidation reward 0 for good invalidation, and negative value for how far away it was.
                reward = difference_in_ttl
            else:
                # miss\expire\eviction. reward 0 for no hits? or
                reward = state.hit_count * (1 + state.cache_utility)

        self.logger.debug(f'{__name__}: Hits: {state.hit_count}, ttl diff: {difference_in_ttl}, Reward: {reward}')
        return reward
