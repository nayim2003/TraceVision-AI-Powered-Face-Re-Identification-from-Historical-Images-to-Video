from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundingBox:
    """Bounding box coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2,
        )


@dataclass
class Detection:
    """Single object detection."""

    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None