
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.storage.gallery import GalleryIdentity


class GalleryEmbeddingLoader:
    """
    Loads reference embeddings from disk.

    Expected embedding format:
        .npy

    Expected shape:
        (512,)
    """

    def __init__(self, dimension: int = 512):
        self.dimension = int(dimension)

    def load_embedding(self, path: str | Path) -> np.ndarray:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Embedding file not found: {path}"
            )

        embedding = np.load(path)

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"got {embedding.shape[0]} "
                f"for {path}"
            )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError(
                f"Zero-norm embedding: {path}"
            )

        embedding = embedding / norm

        return embedding.astype(np.float32)

    def load_person_embeddings(
        self,
        identities: List[GalleryIdentity],
    ) -> Tuple[np.ndarray, List[dict]]:

        embeddings = []
        metadata = []

        for identity in identities:

            if not identity.person_embedding_path:
                continue

            embedding = self.load_embedding(
                identity.person_embedding_path
            )

            embeddings.append(embedding)

            metadata.append(
                {
                    "subject_id": identity.subject_id,
                    "label": identity.label,
                    "modality": "person",
                }
            )

        if not embeddings:
            return (
                np.empty(
                    (0, self.dimension),
                    dtype=np.float32,
                ),
                [],
            )

        return (
            np.vstack(embeddings).astype(np.float32),
            metadata,
        )

    def load_face_embeddings(
        self,
        identities: List[GalleryIdentity],
    ) -> Tuple[np.ndarray, List[dict]]:

        embeddings = []
        metadata = []

        for identity in identities:

            if not identity.face_embedding_path:
                continue

            embedding = self.load_embedding(
                identity.face_embedding_path
            )

            embeddings.append(embedding)

            metadata.append(
                {
                    "subject_id": identity.subject_id,
                    "label": identity.label,
                    "modality": "face",
                }
            )

        if not embeddings:
            return (
                np.empty(
                    (0, self.dimension),
                    dtype=np.float32,
                ),
                [],
            )

        return (
            np.vstack(embeddings).astype(np.float32),
            metadata,
        )
