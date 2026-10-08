
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseFaceEmbedder(ABC):
    """Abstract interface for face embedding extraction."""

    @abstractmethod
    def extract(self, face_crop: Any) -> np.ndarray:
        """Extract a normalized face embedding."""
        raise NotImplementedError

    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return embedding dimensionality."""
        raise NotImplementedError
