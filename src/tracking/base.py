from abc import ABC, abstractmethod
from typing import Any


class BaseTracker(ABC):

    @abstractmethod
    def update(self, frame: Any):
        """Update tracker using a new frame."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset tracker state."""
        raise NotImplementedError