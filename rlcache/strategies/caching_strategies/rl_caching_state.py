from dataclasses import dataclass

import numpy as np

from rlcache.rl_model.agent_state import AgentSystemState


# TODO worth writing tests for this

@dataclass
class CachingAgentSystemState(AgentSystemState):
    __slots__ = ['encoded_key',
                 'ttl',
                 'hit_count',
                 'step_code',
                 'operating_type']

    encoded_key: int
    ttl: int
    hit_count: int
    step_code: int  # code referring to the step the state is in {new -> invalidate/expired/evicted}
    operation_type: int

    @classmethod
    def from_numpy(cls, encoded: np.ndarray):
        return cls(encoded[0], encoded[1], encoded[2], encoded[3], encoded[4])


@dataclass
class CachingAgentIncompleteExperienceEntry(object):
    __slots__ = ['state', 'agent_action', 'starting_state', 'observation_time']
    state: CachingAgentSystemState
    agent_action: np.ndarray
    starting_state: CachingAgentSystemState
    observation_time: float
