
from typing import List, Tuple

import faiss
import numpy as np


class FAISSRetriever:
    """
    FAISS-based vector retrieval engine.

    Uses:
        FAISS IndexFlatIP

    Since embeddings are L2-normalized before insertion
    and before search, inner product is equivalent to
    cosine similarity.

    Responsibilities:
        - Store reference embeddings
        - Store corresponding metadata
        - Normalize embeddings
        - Validate embedding dimensions
        - Perform top-k similarity search
        - Safely handle an empty index
        - Reset the index when required
    """

    def __init__(
        self,
        dimension: int,
    ):
        """
        Initialize the FAISS retriever.

        Parameters
        ----------
        dimension : int
            Embedding dimension.

        Example
        -------
        FAISSRetriever(dimension=512)
        """

        if dimension <= 0:
            raise ValueError(
                "Embedding dimension must be greater than zero."
            )

        self.dimension = int(dimension)

        # Inner Product index.
        # With normalized vectors:
        # Inner Product == Cosine Similarity
        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        # Metadata corresponding to FAISS vector positions.
        self.metadata: List[dict] = []

    # ==========================================================
    # Add Embeddings
    # ==========================================================

    def add(
        self,
        embeddings: np.ndarray,
        metadata: List[dict],
    ) -> None:
        """
        Add reference embeddings and their metadata.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape:
                (n_samples, dimension)

            A single vector of shape
                (dimension,)
            is also accepted.

        metadata : List[dict]
            One metadata dictionary for each embedding.

        Raises
        ------
        ValueError
            If dimensions or metadata count are invalid.
        """

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        # ------------------------------------------------------
        # Convert single embedding to batch format
        # ------------------------------------------------------

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(
                1,
                -1,
            )

        # ------------------------------------------------------
        # Validate shape
        # ------------------------------------------------------

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 1D or 2D NumPy array."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"got {embeddings.shape[1]}."
            )

        # ------------------------------------------------------
        # Validate metadata
        # ------------------------------------------------------

        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                "Number of metadata entries must match "
                "number of embeddings. "
                f"Got {len(metadata)} metadata entries "
                f"for {embeddings.shape[0]} embeddings."
            )

        # ------------------------------------------------------
        # Handle empty input
        # ------------------------------------------------------

        if embeddings.shape[0] == 0:
            return

        # ------------------------------------------------------
        # Ensure contiguous float32 array
        # ------------------------------------------------------

        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        # ------------------------------------------------------
        # Normalize embeddings
        # ------------------------------------------------------

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        # Reject zero vectors
        if np.any(norms == 0):
            raise ValueError(
                "Cannot add zero-norm embedding(s) "
                "to FAISS index."
            )

        faiss.normalize_L2(
            embeddings
        )

        # ------------------------------------------------------
        # Add vectors to FAISS
        # ------------------------------------------------------

        self.index.add(
            embeddings
        )

        # ------------------------------------------------------
        # Store corresponding metadata
        # ------------------------------------------------------

        self.metadata.extend(
            metadata
        )

    # ==========================================================
    # Search
    # ==========================================================

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[dict, float]]:
        """
        Search the reference gallery.

        Parameters
        ----------
        query_embedding : np.ndarray
            Query embedding of shape:

                (dimension,)

            or:

                (1, dimension)

        top_k : int
            Maximum number of results.

        Returns
        -------
        List[Tuple[dict, float]]
            Each result contains:

                (metadata, similarity_score)

            Similarity is cosine similarity because
            both query and reference vectors are normalized.

        Notes
        -----
        If the FAISS index is empty, an empty list is returned.
        """

        # ------------------------------------------------------
        # Validate top_k
        # ------------------------------------------------------

        if top_k <= 0:
            return []

        # ------------------------------------------------------
        # Convert query
        # ------------------------------------------------------

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        # ------------------------------------------------------
        # Convert single vector to batch
        # ------------------------------------------------------

        if query.ndim == 1:
            query = query.reshape(
                1,
                -1,
            )

        # ------------------------------------------------------
        # Validate query shape
        # ------------------------------------------------------

        if query.ndim != 2:
            raise ValueError(
                "Query embedding must be a 1D or "
                "2D NumPy array."
            )

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"got {query.shape[1]}."
            )

        # ------------------------------------------------------
        # Empty query protection
        # ------------------------------------------------------

        if query.shape[0] == 0:
            return []

        # ------------------------------------------------------
        # Empty FAISS index protection
        # ------------------------------------------------------

        if self.index.ntotal == 0:
            return []

        # ------------------------------------------------------
        # Ensure contiguous float32 array
        # ------------------------------------------------------

        query = np.ascontiguousarray(
            query,
            dtype=np.float32,
        )

        # ------------------------------------------------------
        # Validate query norm
        # ------------------------------------------------------

        norms = np.linalg.norm(
            query,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError(
                "Cannot search using a zero-norm "
                "query embedding."
            )

        # ------------------------------------------------------
        # Normalize query
        # ------------------------------------------------------

        faiss.normalize_L2(
            query
        )

        # ------------------------------------------------------
        # Determine actual search size
        # ------------------------------------------------------

        search_k = min(
            int(top_k),
            self.index.ntotal,
        )

        # Extra safety
        if search_k <= 0:
            return []

        # ------------------------------------------------------
        # FAISS search
        # ------------------------------------------------------

        scores, indices = self.index.search(
            query,
            search_k,
        )

        # ------------------------------------------------------
        # Convert FAISS output to application format
        # ------------------------------------------------------

        results: List[
            Tuple[dict, float]
        ] = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            # FAISS may return -1 for invalid entries.
            if index < 0:
                continue

            index = int(index)

            # Defensive metadata validation
            if index >= len(self.metadata):
                continue

            results.append(
                (
                    self.metadata[index],
                    float(score),
                )
            )

        return results

    # ==========================================================
    # Properties
    # ==========================================================

    @property
    def size(self) -> int:
        """
        Number of embeddings currently stored
        in the FAISS index.
        """

        return int(
            self.index.ntotal
        )

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self) -> None:
        """
        Completely reset the FAISS index and metadata.
        """

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.metadata.clear()

    # ==========================================================
    # Utility
    # ==========================================================

    def is_empty(self) -> bool:
        """
        Return True when the index contains
        no reference embeddings.
        """

        return self.index.ntotal == 0

    def __len__(self) -> int:
        """
        Allow:

            len(retriever)

        to return the number of indexed embeddings.
        """

        return self.size
