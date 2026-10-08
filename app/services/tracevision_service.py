
from pathlib import Path
from typing import Optional

import cv2

from src.storage.session import SessionRecord
from src.storage.session_writer import SessionWriter


class TraceVisionService:
    """
    Application service for running TraceVision on an
    authorized test video source.
    """

    def __init__(
        self,
        pipeline,
        session_writer: Optional[SessionWriter] = None,
    ):
        self.pipeline = pipeline
        self.session_writer = (
            session_writer
            if session_writer is not None
            else SessionWriter()
        )

        self.last_run = None

    def process_video(
        self,
        video_path: str | Path,
        frame_skip: int = 5,
    ) -> dict:

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        if frame_skip < 1:
            raise ValueError(
                "frame_skip must be >= 1."
            )

        # -------------------------------------------------
        # Create session
        # -------------------------------------------------

        session = SessionRecord(
            source=str(video_path),
            model_versions={
                "detector": "yolo11n.pt",
                "tracker": "bytetrack",
                "face": "buffalo_l",
                "person_reid": "osnet_x1_0",
            },
            configuration={
                "frame_skip": frame_skip,
                "person_similarity_threshold": 0.70,
                "face_similarity_threshold": 0.70,
                "person_top_k": 10,
                "face_top_k": 10,
            },
        )

        # Attach session to the pipeline
        self.pipeline.person_pipeline.session = session

        # Reset previous run
        self.pipeline.reset()

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            session.fail(
                error=f"Unable to open video: {video_path}"
            )

            self.session_writer.write(session)

            raise RuntimeError(
                f"Unable to open video: {video_path}"
            )

        frame_count = 0
        processed_frames = 0

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                if frame_count % frame_skip != 0:
                    continue

                self.pipeline.process_frame(
                    frame
                )

                processed_frames += 1

            # -------------------------------------------------
            # Finalize remaining tracks
            # -------------------------------------------------
            #
            # ByteTrack may still have active tracks when the
            # video ends. Since no additional frames arrive,
            # those tracks would otherwise never become
            # TERMINATED through the normal lifecycle.
            #
            # TraceVisionPipeline.finish() explicitly flushes
            # all remaining non-finalized tracks and performs:
            #
            #   Person Re-ID
            #        +
            #   Face Re-ID
            #        ↓
            #   Evidence Fusion
            #        ↓
            #   Final Track Result
            #        ↓
            #   Audit Record
            #
            self.pipeline.finish()

            # -------------------------------------------------
            # Finish session
            # -------------------------------------------------

            session.processed_frames = processed_frames

            session.finish()

            session_path = self.session_writer.write(
                session
            )

            result = {
                "video": str(video_path),
                "total_frames": frame_count,
                "processed_frames": processed_frames,
                "pipeline_frames": self.pipeline.frame_number,
                "finalized_tracks": session.finalized_tracks,
                "session_id": session.session_id,
                "session_path": str(session_path),
                "status": session.status,
            }

            self.last_run = result

            return result

        except Exception as exc:

            session.fail(
                error=str(exc)
            )

            self.session_writer.write(
                session
            )

            raise

        finally:

            cap.release()

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    def get_results(self) -> list[dict]:

        results = (
            self.pipeline
            .get_all_final_results()
        )

        return [
            result.to_dict()
            for result in results.values()
        ]

    def get_result(
        self,
        track_id: int,
    ) -> Optional[dict]:

        result = (
            self.pipeline
            .get_final_result(track_id)
        )

        if result is None:
            return None

        return result.to_dict()

