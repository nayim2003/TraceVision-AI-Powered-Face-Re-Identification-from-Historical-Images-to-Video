from pathlib import Path
from typing import Union

import cv2


class VideoSource:
    """Unified video input interface."""

    def __init__(self, source: Union[str, int]):
        self.source = source
        self.capture = None

    def open(self):
        """Open the video source."""

        source = self.source

        if isinstance(source, str):
            source = str(Path(source))

        self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Unable to open video source: {self.source}"
            )

        return self

    def read(self):
        """Read the next frame."""

        if self.capture is None:
            raise RuntimeError(
                "Video source is not open. Call open() first."
            )

        ret, frame = self.capture.read()

        return ret, frame

    def release(self):
        """Release the video source."""

        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()