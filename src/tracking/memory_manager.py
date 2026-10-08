
from typing import Optional
import numpy as np

from src.tracking.models import Track


class TrackMemoryManager:
    """
    Controls how representation observations are stored
    inside a Track.
    """

    def __init__(
        self,
        max_person_embeddings: int = 20,
        max_face_embeddings: int = 20,
        minimum_quality: float = 0.60,
    ):
        self.max_person_embeddings = max_person_embeddings
        self.max_face_embeddings = max_face_embeddings
        self.minimum_quality = minimum_quality

    def add_person_embedding(
        self,
        track: Track,
        embedding: np.ndarray,
        frame_number: int,
        quality: float = 1.0,
    ):

        if quality < self.minimum_quality:
            return False

        if track.memory.person_memory_size >= self.max_person_embeddings:
            self._remove_lowest_quality_person(track)

        track.memory.add_person_embedding(
            embedding=embedding,
            frame_number=frame_number,
            quality=quality,
        )

        return True

    def add_face_embedding(
        self,
        track: Track,
        embedding: np.ndarray,
        frame_number: int,
        quality: float = 1.0,
    ):

        if quality < self.minimum_quality:
            return False

        if track.memory.face_memory_size >= self.max_face_embeddings:
            self._remove_lowest_quality_face(track)

        track.memory.add_face_embedding(
            embedding=embedding,
            frame_number=frame_number,
            quality=quality,
        )

        return True

    def _remove_lowest_quality_person(
        self,
        track: Track,
    ):

        if not track.memory.person_embeddings:
            return

        lowest_index = min(
            range(len(track.memory.person_embeddings)),
            key=lambda i:
                track.memory.person_embeddings[i].quality
        )

        track.memory.person_embeddings.pop(
            lowest_index
        )

    def _remove_lowest_quality_face(
        self,
        track: Track,
    ):

        if not track.memory.face_embeddings:
            return

        lowest_index = min(
            range(len(track.memory.face_embeddings)),
            key=lambda i:
                track.memory.face_embeddings[i].quality
        )

        track.memory.face_embeddings.pop(
            lowest_index
        )
