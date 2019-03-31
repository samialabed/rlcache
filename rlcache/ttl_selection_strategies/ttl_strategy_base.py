from abc import ABC

from observers.observer import Observer


class TtlStrategy(ABC):
    def estimate_ttl(self, key) -> int:
        raise NotImplementedError

    def observer(self) -> Observer:
        """Returns an observer implementation that can be called on various methods."""
        pass
