
from typing import List

import numpy as np

from src.recognition.base_face_detector import BaseFaceDetector
from src.recognition.face_models import FaceDetection


class InsightFaceDetector(BaseFaceDetector):
    """
    InsightFace-based face detector.

    Performs face detection and extracts the corresponding
    ArcFace embedding in the same inference pass.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.50,
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
        self.det_size = det_size
        self.confidence_threshold = confidence_threshold

        # CPU backend for the current environment.
        self.providers = [
            "CPUExecutionProvider"
        ]

        self.app = FaceAnalysis(
            name=model_name,
            providers=self.providers,
        )

        self.app.prepare(
            ctx_id=0,
            det_size=det_size,
        )

    def detect(
        self,
        frame,
    ) -> List[FaceDetection]:
        """
        Detect faces and attach the native InsightFace
        ArcFace embedding to each FaceDetection.
        """

        if frame is None:
            return []

        faces = self.app.get(frame)

        detections = []

        for face in faces:

            confidence = float(
                face.det_score
            )

            if confidence < self.confidence_threshold:
                continue

            bbox = face.bbox

            embedding = getattr(
                face,
                "embedding",
                None,
            )

            if embedding is not None:

                embedding = np.asarray(
                    embedding,
                    dtype=np.float32,
                )

                norm = np.linalg.norm(
                    embedding
                )

                if norm > 0:
                    embedding = (
                        embedding / norm
                    )

                else:
                    embedding = None

            detection = FaceDetection(
                bbox=(
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ),
                confidence=confidence,
                embedding=embedding,
            )

            detections.append(
                detection
            )

        return detections
