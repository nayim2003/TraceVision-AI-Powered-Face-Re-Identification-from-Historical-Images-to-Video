
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.recognition.face_models import FaceDetection


class BaseFaceDetector(ABC):
    """Abstract interface for face detection."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a frame."""
        raise NotImplementedError
