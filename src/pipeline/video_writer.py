
from pathlib import Path
import cv2


class VideoWriter:
    """
    Writes processed TraceVision frames
    to an output video file.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: tuple[int, int],
    ):
        self.output_path = Path(
            output_path
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        width, height = frame_size

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not self.writer.isOpened():
            raise RuntimeError(
                f"Unable to open video writer: "
                f"{self.output_path}"
            )

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.release()
