
from dataclasses import dataclass
import cv2
import numpy as np

from src.recognition.face_models import FaceDetection


@dataclass
class FaceCrop:
    image: np.ndarray
    detection: FaceDetection


class FaceCropExtractor:

    def __init__(
        self,
        target_size: tuple[int, int] = (112, 112),
        padding_ratio: float = 0.10,
    ):
        self.target_size = target_size
        self.padding_ratio = padding_ratio

    def extract(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
    ) -> FaceCrop | None:

        x1, y1, x2, y2 = detection.bbox

        height, width = frame.shape[:2]

        box_width = x2 - x1
        box_height = y2 - y1

        pad_x = box_width * self.padding_ratio
        pad_y = box_height * self.padding_ratio

        x1 = max(
            0,
            int(x1 - pad_x),
        )

        y1 = max(
            0,
            int(y1 - pad_y),
        )

        x2 = min(
            width,
            int(x2 + pad_x),
        )

        y2 = min(
            height,
            int(y2 + pad_y),
        )

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return None

        crop = cv2.resize(
            crop,
            self.target_size,
        )

        return FaceCrop(
            image=crop,
            detection=detection,
        )

