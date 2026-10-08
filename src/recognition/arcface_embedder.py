
from typing import Any

import numpy as np

from src.recognition.base_face_embedder import BaseFaceEmbedder


class ArcFaceEmbedder(BaseFaceEmbedder):
    """
    ArcFace-based face embedding extractor.

    Uses InsightFace as the backend.

    The embedder first checks whether the upstream
    FaceDetection already contains an ArcFace embedding.
    If not, it falls back to running InsightFace on
    the supplied face crop.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "CPU",
    ):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "InsightFace is not installed. "
                "Install it with: "
                "pip install insightface onnxruntime"
            ) from exc

        self.model_name = model_name
        self.device = device

        # Current implementation uses CPU execution.
        providers = [
            "CPUExecutionProvider"
        ]

        self.app = FaceAnalysis(
            name=model_name,
            providers=providers,
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640),
        )

        self._embedding_dimension = 512

    def extract(
        self,
        face_crop: Any,
    ) -> np.ndarray:
        """
        Extract a normalized ArcFace embedding.

        If the FaceCrop already carries an embedding from
        the upstream InsightFace detector, that embedding
        is reused directly.

        Otherwise, InsightFace is run on the supplied crop
        as a fallback.
        """

        # --------------------------------------------------
        # 1. Use Existing Embedding If Available
        # --------------------------------------------------

        existing_embedding = getattr(
            face_crop,
            "embedding",
            None,
        )

        if existing_embedding is not None:

            embedding = np.asarray(
                existing_embedding,
                dtype=np.float32,
            )

            return self._normalize_embedding(
                embedding
            )

        # --------------------------------------------------
        # 2. Fallback: Run InsightFace on Crop
        # --------------------------------------------------

        image = face_crop.image

        if image is None:
            raise ValueError(
                "Face crop image is empty."
            )

        faces = self.app.get(image)

        if not faces:
            raise ValueError(
                "No face detected inside the supplied face crop."
            )

        # Select the strongest detected face.
        face = max(
            faces,
            key=lambda item: float(
                item.det_score
            ),
        )

        embedding = np.asarray(
            face.embedding,
            dtype=np.float32,
        )

        return self._normalize_embedding(
            embedding
        )

    def _normalize_embedding(
        self,
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and L2-normalize an embedding.
        """

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if embedding.size == 0:
            raise ValueError(
                "ArcFace returned an empty embedding."
            )

        if embedding.shape[0] != self._embedding_dimension:
            raise ValueError(
                "Invalid ArcFace embedding dimension: "
                f"{embedding.shape[0]}. "
                f"Expected {self._embedding_dimension}."
            )

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            raise ValueError(
                "ArcFace returned a zero-norm embedding."
            )

        embedding = embedding / norm

        return embedding.astype(
            np.float32
        )

    def embedding_dimension(self) -> int:
        """
        Return the ArcFace embedding dimension.
        """

        return self._embedding_dimension
