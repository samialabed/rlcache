from abc import ABC
from typing import Union

from observers.observer import Observer


class TtlStrategy(ABC):
    def estimate_ttl(self, key) -> int:
        raise NotImplementedError

    def observer(self) -> Union[Observer, None]:
        """Returns an observer implementation that can be called on various methods."""
        return None
