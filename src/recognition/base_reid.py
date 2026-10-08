
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseReIDExtractor(ABC):
    """
    Abstract interface for person Re-ID embedding extraction.
    """

    @abstractmethod
    def extract(
        self,
        person_crop: Any,
    ) -> np.ndarray:
        """
        Convert a person crop into a feature embedding.
        """
        raise NotImplementedError

    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        Return the dimensionality of the embedding.
        """
        raise NotImplementedError
