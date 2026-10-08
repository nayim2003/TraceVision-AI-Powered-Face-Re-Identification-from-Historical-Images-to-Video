from typing import Any

from ultralytics import YOLO

from src.detection.models import BoundingBox, Detection
from src.detection.base import BaseDetector


class YOLODetector(BaseDetector):
    """YOLO-based object detector."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.50,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device

        self.model = YOLO(model_name)

    def detect(self, frame: Any) -> list[Detection]:
        """Run object detection on a frame."""

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
            device=None if self.device == "auto" else self.device,
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            for i in range(len(boxes)):

                xyxy = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().item())
                class_id = int(boxes.cls[i].cpu().item())

                class_name = self.model.names[class_id]

                bbox = BoundingBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                )

                detection = Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )

                detections.append(detection)

        return detections