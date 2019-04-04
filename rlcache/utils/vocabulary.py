from typing import Sized, List, FrozenSet, Optional

import numpy as np


# Taken and edited from https://github.com/Microsoft/dpu-utils/blob/master/dpu_utils/mlutils/vocabulary.py
class Vocabulary(Sized):
    """
    A simple vocabulary that maps strings to unique ids (and back).
    """

    def __init__(self, add_unk: bool = True, add_pad: bool = True) -> None:
        self.token_to_id = {}
        self.id_to_token = []
        if add_pad:
            self.add_or_get_id(self.get_pad())
        if add_unk:
            self.add_or_get_id(self.get_unk())

    def is_equal_to(self, other) -> bool:
        """
        This would be __eq__, except that Python 3 insists that __hash__ is defined if __eq__ is
        defined, and we can't define __hash__ because the object is mutable.
        """
        if not isinstance(other, Vocabulary):
            return False
        return self.id_to_token == other.id_to_token

    def add_or_get_id(self, token: str) -> int:
        """
        Get the token's id. If the token does not exist in the
        dictionary, add it.
        """
        this_id = self.token_to_id.get(token)
        if this_id is not None:
            return this_id

        this_id = len(self.id_to_token)
        self.token_to_id[token] = this_id
        self.id_to_token.append(token)
        return this_id

    def is_unk(self, token: str) -> bool:
        return token not in self.token_to_id

    def get_id_or_unk(self, token: str) -> int:
        this_id = self.token_to_id.get(token)
        if this_id is not None:
            return this_id
        else:
            return self.token_to_id[self.get_unk()]

    def get_id_or_add_multiple(self,
                               tokens: List[str],
                               pad_to_size: Optional[int] = None,
                               padding_element: int = 0) -> List[int]:
        if pad_to_size is not None:
            tokens = tokens[:pad_to_size]

        ids = [self.add_or_get_id(t) for t in tokens]

        if pad_to_size is not None and len(ids) != pad_to_size:
            ids += [padding_element] * (pad_to_size - len(ids))

        return ids

    def get_id_or_unk_multiple(self,
                               tokens: List[str],
                               pad_to_size: Optional[int] = None,
                               padding_element: int = 0) -> List[int]:
        if pad_to_size is not None:
            tokens = tokens[:pad_to_size]

        ids = [self.get_id_or_unk(t) for t in tokens]

        if pad_to_size is not None and len(ids) != pad_to_size:
            ids += [padding_element] * (pad_to_size - len(ids))

        return ids

    def get_name_for_id(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __str__(self):
        return str(self.token_to_id)

    def get_all_names(self) -> FrozenSet[str]:
        return frozenset(self.token_to_id.keys())

    def translate_tokenized_array_to_list_words(self, token: np.ndarray) -> List[str]:
        """Helper function to translate numpy array tokens back to words"""
        return [self.get_name_for_id(n) for n in token[np.nonzero(token != self.get_id_or_unk(self.get_pad()))]]

    @staticmethod
    def get_unk() -> str:
        return '%UNK%'

    @staticmethod
    def get_pad() -> str:
        return '%PAD%'
