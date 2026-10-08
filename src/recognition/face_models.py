
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FaceDetection:
    """
    Represents one detected face.
    """

    bbox: tuple[float, float, float, float]

    confidence: float

    embedding: Optional[np.ndarray] = None

    track_id: Optional[int] = None


@dataclass
class FaceQualityResult:
    """
    Quality assessment for a face crop.
    """

    score: float

    accepted: bool

    width: int

    height: int

    blur_score: float

    confidence: float
