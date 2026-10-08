
from typing import List

from src.detection.models import Detection
from src.recognition.face_models import FaceDetection


class FaceTrackAssociator:
    """
    Associates detected faces with existing person detections.

    Association is based primarily on:
    1. Face center being inside the person bounding box.
    2. Face/person spatial overlap.
    3. One face assigned to one person track per frame.
    """

    def __init__(
        self,
        minimum_iou: float = 0.01,
        center_inside_required: bool = True,
    ):
        self.minimum_iou = minimum_iou
        self.center_inside_required = center_inside_required

    @staticmethod
    def _calculate_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        intersection = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def _face_center(face_bbox):
        x1, y1, x2, y2 = face_bbox

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    @staticmethod
    def _point_inside_box(point, box):
        x, y = point
        x1, y1, x2, y2 = box

        return (
            x1 <= x <= x2
            and y1 <= y <= y2
        )

    def associate(
        self,
        faces: List[FaceDetection],
        person_detections: List[Detection],
    ):

        associated_faces = []

        # Prevent multiple faces from being assigned
        # to the same person track in one frame.
        used_track_ids = set()

        # Process higher-confidence faces first.
        ordered_faces = sorted(
            faces,
            key=lambda face: float(face.confidence),
            reverse=True,
        )

        for face in ordered_faces:

            face_bbox = face.bbox
            face_center = self._face_center(face_bbox)

            best_detection = None
            best_score = -1.0

            for detection in person_detections:

                track_id = detection.track_id

                if track_id is None:
                    continue

                # One face per person track per frame.
                if track_id in used_track_ids:
                    continue

                person_bbox = (
                    detection.bbox.x1,
                    detection.bbox.y1,
                    detection.bbox.x2,
                    detection.bbox.y2,
                )

                # Face center must belong to the person box.
                if self.center_inside_required:
                    if not self._point_inside_box(
                        face_center,
                        person_bbox,
                    ):
                        continue

                iou = self._calculate_iou(
                    face_bbox,
                    person_bbox,
                )

                if iou < self.minimum_iou:
                    continue

                # Prefer higher IoU.
                score = iou

                if score > best_score:
                    best_score = score
                    best_detection = detection

            if best_detection is None:
                continue

            face.track_id = best_detection.track_id

            used_track_ids.add(
                best_detection.track_id
            )

            associated_faces.append(face)

        return associated_faces
