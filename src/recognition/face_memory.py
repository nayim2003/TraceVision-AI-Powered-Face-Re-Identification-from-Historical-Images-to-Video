
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class FaceEmbeddingObservation:
    embedding: np.ndarray
    frame_number: int
    quality: float = 1.0


@dataclass
class FaceTrackMemory:

    embeddings: List[
        FaceEmbeddingObservation
    ] = field(default_factory=list)

    def add(
        self,
        embedding: np.ndarray,
        frame_number: int,
        quality: float = 1.0,
    ):

        self.embeddings.append(
            FaceEmbeddingObservation(
                embedding=np.asarray(
                    embedding,
                    dtype=np.float32,
                ),
                frame_number=frame_number,
                quality=quality,
            )
        )

    @property
    def size(self) -> int:
        return len(self.embeddings)

    def clear(self):
        self.embeddings.clear()

