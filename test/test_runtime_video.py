
from pathlib import Path

import cv2

from src.pipeline.factory import build_tracevision_pipeline


VIDEO_PATH = Path(
    "data/raw/test_video.mp4"
)

MAX_FRAMES = 100


def test_real_video_smoke():

    assert VIDEO_PATH.exists(), (
        f"Video not found: {VIDEO_PATH}"
    )

    pipeline = build_tracevision_pipeline()

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    assert cap.isOpened(), (
        "Could not open video"
    )

    frame_count = 0

    try:

        while frame_count < MAX_FRAMES:

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            pipeline.process_frame(
                frame
            )

            if frame_count % 10 == 0:
                print(
                    f"Processed frame: "
                    f"{frame_count}"
                )

    finally:

        cap.release()

    results = pipeline.finish()

    print()
    print(
        f"Frames processed: {frame_count}"
    )

    print(
        f"Final tracks: {len(results)}"
    )

    for result in results:
        print(result)

    assert frame_count > 0, (
        "No video frames processed"
    )


if __name__ == "__main__":
    test_real_video_smoke()
