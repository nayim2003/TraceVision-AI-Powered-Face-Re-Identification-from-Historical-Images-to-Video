
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class QualityResult:
    score: float
    accepted: bool

    width: int
    height: int

    blur_score: float
    size_score: float
    aspect_score: float

    reason: str


class CropQualityAnalyzer:
    """
    Evaluates whether an image crop is suitable
    for representation extraction.
    """

    def __init__(
        self,
        min_width: int = 40,
        min_height: int = 80,
        min_blur_score: float = 50.0,
        minimum_quality: float = 0.60,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.min_blur_score = min_blur_score
        self.minimum_quality = minimum_quality

    def analyze(
        self,
        crop: Optional[np.ndarray],
    ) -> QualityResult:

        if crop is None or crop.size == 0:

            return QualityResult(
                score=0.0,
                accepted=False,
                width=0,
                height=0,
                blur_score=0.0,
                size_score=0.0,
                aspect_score=0.0,
                reason="empty_crop",
            )

        height, width = crop.shape[:2]

        # --------------------------------
        # Size score
        # --------------------------------

        width_score = min(
            width / self.min_width,
            1.0
        )

        height_score = min(
            height / self.min_height,
            1.0
        )

        size_score = (
            width_score + height_score
        ) / 2.0

        # --------------------------------
        # Blur score
        # --------------------------------

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )

        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        blur_quality = min(
            blur_score / self.min_blur_score,
            1.0
        )

        # --------------------------------
        # Aspect ratio
        # --------------------------------

        aspect_ratio = width / max(
            height,
            1
        )

        # Person crops are normally
        # taller than they are wide.
        if 0.20 <= aspect_ratio <= 1.20:
            aspect_score = 1.0
        else:
            aspect_score = 0.5

        # --------------------------------
        # Final quality score
        # --------------------------------

        score = (
            0.40 * size_score
            + 0.40 * blur_quality
            + 0.20 * aspect_score
        )

        accepted = (
            score >= self.minimum_quality
            and width >= self.min_width
            and height >= self.min_height
        )

        if not accepted:

            if width < self.min_width:
                reason = "crop_too_small"

            elif height < self.min_height:
                reason = "crop_too_small"

            elif blur_score < self.min_blur_score:
                reason = "too_blurry"

            else:
                reason = "low_quality"

        else:
            reason = "accepted"

        return QualityResult(
            score=float(score),
            accepted=accepted,
            width=width,
            height=height,
            blur_score=blur_score,
            size_score=float(size_score),
            aspect_score=float(aspect_score),
            reason=reason,
        )
