import time
from dataclasses import dataclass

import numpy as np

from rlcache.rl_model.agent_state import AgentSystemState


# TODO worth writing tests for this

@dataclass
class EvictionAgentSystemState(AgentSystemState):
    __slots__ = ['encoded_key',
                 'ttl',
                 'expiry_time',
                 'hit_count',
                 'miss_count',
                 'invalidate_count',
                 'expiration_count']

    encoded_key: int
    ttl: int
    expiry_time: time
    hit_count: int
    miss_count: int
    invalidate_count: int
    expiration_count: int

    @classmethod
    def from_numpy(cls, encoded: np.ndarray):
        return cls(encoded[0],
                   encoded[1],
                   encoded[2],
                   encoded[3],
                   encoded[4],
                   encoded[5],
                   encoded[6])


@dataclass
class EvictionAgentIncompleteExperienceEntry(object):
    __slots__ = ['state', 'agent_action', 'starting_state']
    state: EvictionAgentSystemState
    agent_action: np.ndarray
    starting_state: EvictionAgentSystemState

    def __repr__(self):
        return f'state: {self.state}, agent_action: {self.agent_action}, starting_state: {self.starting_state}'
