
import cv2
import numpy as np
from src.recognition.face_models import FaceQualityResult


class FaceQualityAnalyzer:

    def __init__(
        self,
        min_width: int = 40,
        min_height: int = 40,
        min_blur_score: float = 50.0,
        minimum_quality: float = 0.60,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.min_blur_score = min_blur_score
        self.minimum_quality = minimum_quality

    def analyze(
        self,
        face_image,
        confidence: float = 1.0,
    ) -> FaceQualityResult:

        height, width = face_image.shape[:2]

        gray = cv2.cvtColor(
            face_image,
            cv2.COLOR_BGR2GRAY,
        )

        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        size_score = min(
            width / self.min_width,
            height / self.min_height,
            1.0,
        )

        blur_score_normalized = min(
            blur_score / self.min_blur_score,
            1.0,
        )

        confidence_score = min(
            max(confidence, 0.0),
            1.0,
        )

        quality = (
            0.35 * size_score
            + 0.40 * blur_score_normalized
            + 0.25 * confidence_score
        )

        accepted = (
            width >= self.min_width
            and height >= self.min_height
            and quality >= self.minimum_quality
        )

        return FaceQualityResult(
            score=float(quality),
            accepted=bool(accepted),
            width=width,
            height=height,
            blur_score=blur_score,
            confidence=confidence,
        )
