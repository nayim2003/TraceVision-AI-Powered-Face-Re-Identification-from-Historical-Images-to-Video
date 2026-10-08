
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.detection.models import Detection


@dataclass
class PersonCrop:
    image: np.ndarray

    x1: int
    y1: int
    x2: int
    y2: int

    original_width: int
    original_height: int

    track_id: Optional[int] = None


class PersonCropExtractor:
    """
    Extracts and preprocesses person crops from video frames.
    """

    def __init__(
        self,
        target_width: int = 128,
        target_height: int = 256,
        padding_ratio: float = 0.05,
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.padding_ratio = padding_ratio

    def extract(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> Optional[PersonCrop]:

        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]

        # --------------------------------
        # Original bounding box
        # --------------------------------

        x1 = int(detection.bbox.x1)
        y1 = int(detection.bbox.y1)
        x2 = int(detection.bbox.x2)
        y2 = int(detection.bbox.y2)

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            return None

        # --------------------------------
        # Add small padding
        # --------------------------------

        pad_x = int(
            box_width * self.padding_ratio
        )

        pad_y = int(
            box_height * self.padding_ratio
        )

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)

        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        # --------------------------------
        # Crop
        # --------------------------------

        crop = frame[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            return None

        # --------------------------------
        # Resize
        # --------------------------------

        resized = cv2.resize(
            crop,
            (
                self.target_width,
                self.target_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        return PersonCrop(
            image=resized,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            original_width=width,
            original_height=height,
            track_id=detection.track_id,
        )
