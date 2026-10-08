
from typing import List, Dict, Optional

import faiss
import numpy as np


class FaceFAISSIndex:
    """
    FAISS vector index for ArcFace embeddings.
    """

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.metadata: List[Dict] = []

    @staticmethod
    def _normalize(
        vectors: np.ndarray,
    ) -> np.ndarray:

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        faiss.normalize_L2(vectors)

        return vectors

    def add(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ):

        embeddings = self._normalize(embeddings)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Number of embeddings and metadata "
                "records must match."
            )

        self.index.add(embeddings)

        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ):

        if self.index.ntotal == 0:
            return []

        query = self._normalize(
            query_embedding
        )

        top_k = min(
            top_k,
            self.index.ntotal,
        )

        similarities, indices = self.index.search(
            query,
            top_k,
        )

        results = []

        for similarity, index in zip(
            similarities[0],
            indices[0],
        ):

            if index < 0:
                continue

            metadata = self.metadata[index]

            results.append(
                {
                    "rank": len(results) + 1,
                    "similarity": float(similarity),
                    **metadata,
                }
            )

        return results

    def reset(self):

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.metadata.clear()

    @property
    def size(self) -> int:
        return self.index.ntotal
