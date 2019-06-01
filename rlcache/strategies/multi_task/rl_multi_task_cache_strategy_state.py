from dataclasses import dataclass

import numpy as np

from rlcache.rl_model.agent_state import AgentSystemState


@dataclass
class MultiTaskAgentSystemState(AgentSystemState):
    __slots__ = ['encoded_key',
                 'hit_count',
                 'ttl',
                 'step_code',
                 'cache_utility',
                 'operation_type'
                 ]

    encoded_key: int
    hit_count: int
    ttl: int
    step_code: int  # code referring to the step the state is in {new -> invalidate/expired/evicted}
    cache_utility: int
    operation_type: int

    @classmethod
    def from_numpy(cls, encoded: np.ndarray):
        return cls(encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5])


@dataclass
class MultiTaskAgentObservedExperience(object):
    __slots__ = ['state', 'agent_action', 'starting_state', 'observation_time', 'manual_eviction']
    state: MultiTaskAgentSystemState
    agent_action: np.ndarray
    starting_state: MultiTaskAgentSystemState
    observation_time: float
    manual_eviction: bool
