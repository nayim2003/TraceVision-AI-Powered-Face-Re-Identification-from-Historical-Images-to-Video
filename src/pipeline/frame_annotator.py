
from typing import Optional
import cv2


class FrameAnnotator:
    """
    Draws TraceVision processing information
    onto video frames.

    This component is visualization-only.
    It does not make identity decisions.
    """

    def __init__(
        self,
        show_track_id: bool = True,
        show_quality: bool = True,
        show_candidate_status: bool = True,
        show_frame_number: bool = True,
    ):
        self.show_track_id = show_track_id
        self.show_quality = show_quality
        self.show_candidate_status = show_candidate_status
        self.show_frame_number = show_frame_number

    def annotate(
        self,
        frame,
        tracks,
        frame_number: int,
        pipeline=None,
    ):
        output = frame.copy()

        if self.show_frame_number:
            cv2.putText(
                output,
                f"Frame: {frame_number}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        for track in tracks:

            if not track.detections:
                continue

            detection = track.detections[-1]
            bbox = detection.bbox

            x1 = int(bbox.x1)
            y1 = int(bbox.y1)
            x2 = int(bbox.x2)
            y2 = int(bbox.y2)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            labels = []

            if self.show_track_id:
                labels.append(
                    f"Track {track.track_id}"
                )

            if self.show_candidate_status:
                candidates = []

                if pipeline is not None:
                    candidates = (
                        pipeline.get_ranked_candidates(
                            track.track_id
                        )
                    )

                if candidates:
                    labels.append(
                        f"Candidates: {len(candidates)}"
                    )
                else:
                    labels.append(
                        "Candidates: 0"
                    )

            if self.show_quality:
                memory_size = (
                    track.memory.person_memory_size
                )

                labels.append(
                    f"Embeddings: {memory_size}"
                )

            text_y = max(
                y1 - 10,
                20,
            )

            for label in reversed(labels):
                cv2.putText(
                    output,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                text_y -= 22

        return output
