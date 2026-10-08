
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class EmbeddingObservation:
    """
    One embedding observation associated with a track.
    """

    embedding: np.ndarray
    frame_number: int
    quality: float = 1.0


@dataclass
class TrackMemory:
    """
    Stores representation-level memory for a single track.
    """

    person_embeddings: List[EmbeddingObservation] = field(
        default_factory=list
    )

    face_embeddings: List[EmbeddingObservation] = field(
        default_factory=list
    )

    identity_candidates: List[dict] = field(
        default_factory=list
    )

    def add_person_embedding(
        self,
        embedding: np.ndarray,
        frame_number: int,
        quality: float = 1.0,
    ):
        self.person_embeddings.append(
            EmbeddingObservation(
                embedding=np.asarray(
                    embedding,
                    dtype=np.float32
                ),
                frame_number=frame_number,
                quality=quality,
            )
        )

    def add_face_embedding(
        self,
        embedding: np.ndarray,
        frame_number: int,
        quality: float = 1.0,
    ):
        self.face_embeddings.append(
            EmbeddingObservation(
                embedding=np.asarray(
                    embedding,
                    dtype=np.float32
                ),
                frame_number=frame_number,
                quality=quality,
            )
        )

    def add_identity_candidate(
        self,
        candidate: dict,
    ):
        self.identity_candidates.append(candidate)

    @property
    def person_memory_size(self) -> int:
        return len(self.person_embeddings)

    @property
    def face_memory_size(self) -> int:
        return len(self.face_embeddings)

    @property
    def identity_memory_size(self) -> int:
        return len(self.identity_candidates)
