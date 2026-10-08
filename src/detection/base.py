from abc import ABC, abstractmethod
from typing import Any


class BaseDetector(ABC):

    @abstractmethod
    def detect(self, frame: Any):
        """Detect objects in a frame."""
        raise NotImplementedError