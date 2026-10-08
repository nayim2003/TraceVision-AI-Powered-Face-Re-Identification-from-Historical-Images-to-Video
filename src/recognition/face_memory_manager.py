
import numpy as np

from src.recognition.face_memory import FaceTrackMemory


class FaceMemoryManager:
    """
    Manages temporal ArcFace embeddings for a face track.
    """

    def __init__(
        self,
        max_embeddings: int = 20,
        minimum_quality: float = 0.60,
    ):
        self.max_embeddings = max_embeddings
        self.minimum_quality = minimum_quality

    def add_embedding(
        self,
        memory: FaceTrackMemory,
        embedding: np.ndarray,
        frame_number: int,
        quality: float = 1.0,
    ) -> bool:

        if quality < self.minimum_quality:
            return False

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return False

        embedding = embedding / norm

        if memory.size >= self.max_embeddings:
            self._remove_lowest_quality(memory)

        memory.add(
            embedding=embedding,
            frame_number=frame_number,
            quality=quality,
        )

        return True

    @staticmethod
    def _remove_lowest_quality(
        memory: FaceTrackMemory,
    ) -> None:

        if not memory.embeddings:
            return

        lowest_index = min(
            range(len(memory.embeddings)),
            key=lambda i: memory.embeddings[i].quality,
        )

        memory.embeddings.pop(lowest_index)
