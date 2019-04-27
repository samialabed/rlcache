import time
from dataclasses import dataclass

import numpy as np

from rlcache.rl_model.agent_state import AgentSystemState


@dataclass
class TTLAgentSystemState(AgentSystemState):
    __slots__ = ['encoded_key',
                 'observation_time',
                 'hit_count',
                 'step_code',
                 'cache_utility',
                 'operation_type']

    encoded_key: int
    observation_time: time
    hit_count: int
    step_code: int  # code referring to the step the state is in {new -> invalidate/expired/evicted}
    cache_utility: int
    operation_type: int

    @classmethod
    def from_numpy(cls, encoded: np.ndarray):
        return cls(encoded[0],
                   encoded[1],
                   encoded[2],
                   encoded[3],
                   encoded[4],
                   encoded[5])


@dataclass
class TTLAgentObservedExperience(object):
    __slots__ = ['state', 'agent_action', 'starting_state']
    state: TTLAgentSystemState
    agent_action: np.ndarray
    starting_state: TTLAgentSystemState

    def __repr__(self):
        return f'state: {self.state}, agent_action: {self.agent_action}, starting_state: {self.starting_state}'
