
from typing import Optional
import numpy as np

from src.tracking.models import Track


class TrackEmbeddingAggregator:

    def __init__(
        self,
        minimum_quality: float = 0.60,
        max_embeddings: int = 20,
    ):
        self.minimum_quality = minimum_quality
        self.max_embeddings = max_embeddings

    def aggregate(
        self,
        track: Track,
    ) -> Optional[np.ndarray]:

        observations = [
            obs
            for obs in track.memory.person_embeddings
            if obs.quality >= self.minimum_quality
        ]

        if not observations:
            return None

        observations = observations[-self.max_embeddings:]

        embeddings = np.stack(
            [obs.embedding for obs in observations]
        ).astype(np.float32)

        weights = np.asarray(
            [obs.quality for obs in observations],
            dtype=np.float32,
        )

        weight_sum = weights.sum()

        if weight_sum <= 0:
            return None

        aggregated = np.average(
            embeddings,
            axis=0,
            weights=weights,
        )

        norm = np.linalg.norm(aggregated)

        if norm > 0:
            aggregated = aggregated / norm

        return aggregated.astype(np.float32)
