
from typing import Optional

from src.pipeline.video_source import VideoSource
from src.pipeline.frame_annotator import FrameAnnotator
from src.pipeline.video_writer import VideoWriter


class VideoRunner:
    '''
    Runs TraceVision over a video source and optionally
    produces an annotated output video.
    '''

    def __init__(
        self,
        pipeline,
        session_writer,
        source,
        max_frames: Optional[int] = None,
        output_path: Optional[str] = None,
        annotator=None,
    ):
        self.pipeline = pipeline
        self.session_writer = session_writer
        self.source = source
        self.max_frames = max_frames
        self.output_path = output_path

        self.annotator = (
            annotator
            if annotator is not None
            else FrameAnnotator()
        )

        self.total_frames = 0
        self.processed_frames = 0

    def run(self):

        video_source = VideoSource(
            self.source
        )

        writer = None

        try:
            video_source.open()

            fps = (
                video_source.capture.get(
                    5
                )
                or 20.0
            )

            width = int(
                video_source.capture.get(3)
            )

            height = int(
                video_source.capture.get(4)
            )

            if self.output_path is not None:

                writer = VideoWriter(
                    output_path=self.output_path,
                    fps=fps,
                    frame_size=(
                        width,
                        height,
                    ),
                )

            while True:

                if (
                    self.max_frames is not None
                    and self.processed_frames
                    >= self.max_frames
                ):
                    break

                ret, frame = (
                    video_source.read()
                )

                if not ret:
                    break

                self.total_frames += 1

                active_tracks = (
                    self.pipeline.process_frame(
                        frame
                    )
                )

                annotated_frame = (
                    self.annotator.annotate(
                        frame=frame,
                        tracks=active_tracks,
                        frame_number=(
                            self.pipeline.frame_number
                        ),
                        pipeline=self.pipeline,
                    )
                )

                if writer is not None:
                    writer.write(
                        annotated_frame
                    )

                self.processed_frames += 1

            session_path = (
                self.pipeline.finish_session(
                    self.session_writer
                )
            )

            return {
                "status": "completed",
                "total_frames": self.total_frames,
                "processed_frames": self.processed_frames,
                "finalized_tracks": (
                    self.pipeline.session
                    .finalized_tracks
                ),
                "session_id": (
                    self.pipeline.session
                    .session_id
                ),
                "session_path": str(
                    session_path
                ),
                "output_video": (
                    self.output_path
                ),
            }

        except Exception as error:

            self.pipeline.fail_session(
                self.session_writer,
                error=str(error),
            )

            raise

        finally:

            video_source.release()

            if writer is not None:
                writer.release()
