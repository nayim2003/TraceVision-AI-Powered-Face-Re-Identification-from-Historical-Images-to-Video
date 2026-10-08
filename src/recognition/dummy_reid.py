
import numpy as np

from src.recognition.base_reid import BaseReIDExtractor


class DummyReIDExtractor(BaseReIDExtractor):
    """
    Temporary extractor used for pipeline testing.
    """

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    def extract(self, person_crop) -> np.ndarray:

        # Temporary synthetic embedding
        embedding = np.random.rand(
            self.dimension
        ).astype(np.float32)

        # L2 normalization
        norm = np.linalg.norm(embedding)

        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embedding_dimension(self) -> int:
        return self.dimension
