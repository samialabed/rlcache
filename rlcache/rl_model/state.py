from typing import Dict

import numpy as np


class State(object):

    def __init__(self, value: np.ndarray, meta_data: Dict[str, any] = None):
        """
        Represents a state and optional meta data required to process it.
        Args:
            value: State value.
            meta_data: Meta data dict, any additional state information used
                when processing the state.
        """
        self.state_value = value
        self.meta_data = meta_data

    def as_array(self) -> np.ndarray:
        """
        Returns state array.
        """
        assert isinstance(self.state_value, np.ndarray)
        return self.state_value

    def get_meta_data(self):
        return self.meta_data

    def set_meta_data(self, meta_data):
        if self.meta_data is not None:
            raise ValueError("Meta data object can not be overwritten.")
        self.meta_data = meta_data

    def update_meta_data(self, meta_data):
        for key, value in meta_data.items():
            self.meta_data[key] = value

    def as_dict(self):
        assert isinstance(self.state_value, dict)
        return self.state_value
