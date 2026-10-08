from typing import Any

from ultralytics import YOLO

from src.detection.models import BoundingBox, Detection
from src.tracking.base import BaseTracker


class ByteTrackTracker(BaseTracker):
    """ByteTrack adapter using the Ultralytics tracking interface."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.50,
        tracker_config: str = "bytetrack.yaml",
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker_config

        self.model = YOLO(model_name)

    def update(self, frame: Any) -> list[Detection]:
        """Run person detection and ByteTrack tracking on a frame."""

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            for i in range(len(boxes)):

                class_id = int(
                    boxes.cls[i].cpu().item()
                )

                # Safety check: only person detections.
                if class_id != self.PERSON_CLASS_ID:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy()

                confidence = float(
                    boxes.conf[i].cpu().item()
                )

                class_name = self.model.names[class_id]

                track_id = None

                if boxes.id is not None:
                    track_id = int(
                        boxes.id[i].cpu().item()
                    )

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
                    track_id=track_id,
                )

                detections.append(detection)

        print(
            f"[ByteTrack] detections={len(detections)} "
            f"track_ids={[d.track_id for d in detections]}"
        )

        return detections

    def reset(self):
        """Reset tracker state."""

        self.model = YOLO(self.model_name)