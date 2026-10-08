

from typing import Optional

import numpy as np

from src.recognition.face_memory import FaceTrackMemory


class FaceEmbeddingAggregator:
    """
    Creates a robust track-level face embedding
    from multiple ArcFace observations.
    """

    def __init__(
        self,
        minimum_quality: float = 0.60,
        max_embeddings: int = 20,
    ):
        self.minimum_quality = minimum_quality
        self.max_embeddings = max_embeddings

    def aggregate(
        self,
        memory: FaceTrackMemory,
    ) -> Optional[np.ndarray]:

        valid_observations = [
            observation
            for observation in memory.embeddings
            if observation.quality >= self.minimum_quality
        ]

        if not valid_observations:
            return None

        valid_observations = valid_observations[
            -self.max_embeddings:
        ]

        embeddings = np.stack(
            [
                observation.embedding
                for observation in valid_observations
            ],
            axis=0,
        )

        qualities = np.asarray(
            [
                observation.quality
                for observation in valid_observations
            ],
            dtype=np.float32,
        )

        quality_sum = qualities.sum()

        if quality_sum <= 0:
            return None

        # Quality-weighted aggregation
        aggregated = np.average(
            embeddings,
            axis=0,
            weights=qualities,
        )

        # L2 normalization
        norm = np.linalg.norm(aggregated)

        if norm == 0:
            return None

        aggregated = aggregated / norm

        return aggregated.astype(np.float32)
