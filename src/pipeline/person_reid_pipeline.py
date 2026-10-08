
from typing import Dict, List, Optional

from src.tracking.base import BaseTracker
from src.tracking.track_manager import TrackManager
from src.recognition.person_crop import PersonCropExtractor
from src.recognition.quality import CropQualityAnalyzer
from src.tracking.memory_manager import TrackMemoryManager
from src.aggregation.track_embedding import TrackEmbeddingAggregator
from src.aggregation.track_finalizer import TrackFinalizer
from src.retrieval.retrieval_service import RetrievalService


class PersonReIDPipeline:
    """
    Person detection, tracking, Re-ID, memory, retrieval
    and track-level candidate aggregation pipeline.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        track_manager: TrackManager,
        crop_extractor: PersonCropExtractor,
        quality_analyzer: CropQualityAnalyzer,
        reid_extractor,
        memory_manager: TrackMemoryManager,
        aggregator: TrackEmbeddingAggregator,
        retrieval_service: RetrievalService,
        candidate_history=None,
        candidate_ranker=None,
        track_finalizer=None,
        session=None,
        audit_writer=None,
        auto_finalize: bool = True,
    ):
        self.tracker = tracker
        self.track_manager = track_manager
        self.crop_extractor = crop_extractor
        self.quality_analyzer = quality_analyzer
        self.reid_extractor = reid_extractor
        self.memory_manager = memory_manager
        self.aggregator = aggregator
        self.retrieval_service = retrieval_service

        self.candidate_history = candidate_history
        self.candidate_ranker = candidate_ranker

        self.track_finalizer = (
            track_finalizer
            if track_finalizer is not None
            else TrackFinalizer()
        )

        self.session = session
        self.audit_writer = audit_writer

        # Controls whether this pipeline finalizes tracks by itself.
        #
        # True:
        #     Normal standalone Person Re-ID pipeline.
        #
        # False:
        #     Used by the unified TraceVision pipeline so that
        #     Face Re-ID and evidence fusion can happen first.
        self.auto_finalize = auto_finalize

        self.frame_number = 0

        self.track_results: Dict[int, List] = {}
        self.ranked_results: Dict[int, List] = {}
        self.final_results: Dict[int, object] = {}

    def process_frame(self, frame):
        """
        Process one video frame.

        Returns:
            List of currently active tracks.
        """

        self.frame_number += 1

        if self.session is not None:
            self.session.processed_frames = self.frame_number

        # --------------------------------------------------
        # 1. Detection + tracking
        # --------------------------------------------------

        detections = self.tracker.update(frame)

        active_tracks = self.track_manager.update(
            detections=detections,
            frame_number=self.frame_number,
        )

        # --------------------------------------------------
        # 2. Person Re-ID processing
        # --------------------------------------------------

        for track in active_tracks:

            if not track.detections:
                continue

            detection = track.detections[-1]

            # ----------------------------------------------
            # Extract person crop
            # ----------------------------------------------

            person_crop = self.crop_extractor.extract(
                frame,
                detection,
            )

            if person_crop is None:
                continue

            # ----------------------------------------------
            # Quality assessment
            # ----------------------------------------------

            quality_result = self.quality_analyzer.analyze(
                person_crop.image
            )

            if not quality_result.accepted:
                continue

            # ----------------------------------------------
            # Extract Re-ID embedding
            # ----------------------------------------------

            embedding = self.reid_extractor.extract(
                person_crop
            )

            # ----------------------------------------------
            # Store embedding in track memory
            # ----------------------------------------------

            stored = self.memory_manager.add_person_embedding(
                track=track,
                embedding=embedding,
                frame_number=self.frame_number,
                quality=quality_result.score,
            )

            if not stored:
                continue

            # Need multiple observations before retrieval.
            if track.memory.person_memory_size < 3:
                continue

            # ----------------------------------------------
            # Track-level embedding aggregation
            # ----------------------------------------------

            track_embedding = self.aggregator.aggregate(
                track
            )

            if track_embedding is None:
                continue

            # ----------------------------------------------
            # Vector retrieval
            # ----------------------------------------------

            candidates = self.retrieval_service.search(
                track_embedding
            )

            self.track_results[track.track_id] = candidates

            # ----------------------------------------------
            # Candidate history
            # ----------------------------------------------

            if self.candidate_history is not None:

                self.candidate_history.update(
                    track_id=track.track_id,
                    candidates=candidates,
                )

                # ------------------------------------------
                # Candidate ranking
                # ------------------------------------------

                if self.candidate_ranker is not None:

                    evidence = (
                        self.candidate_history.get_candidates(
                            track.track_id
                        )
                    )

                    ranked = self.candidate_ranker.rank(
                        evidence
                    )

                    self.ranked_results[
                        track.track_id
                    ] = ranked

        # --------------------------------------------------
        # 3. Optional automatic finalization
        # --------------------------------------------------

        if self.auto_finalize:
            self._finalize_terminated_tracks()

        return active_tracks

    # ======================================================
    # Track finalization
    # ======================================================

    def _finalize_terminated_tracks(self):

        terminated_tracks = (
            self.track_manager
            .get_newly_terminated_tracks()
        )

        for track in terminated_tracks:

            # ----------------------------------------------
            # Ignore tracks that are too short
            # ----------------------------------------------

            if (
                track.observation_count
                < self.track_manager.minimum_track_length
            ):
                self.track_manager.mark_finalized(
                    track.track_id
                )
                continue

            # ----------------------------------------------
            # Get ranked person candidates
            # ----------------------------------------------

            ranked_candidates = (
                self.ranked_results.get(
                    track.track_id,
                    []
                )
            )

            # ----------------------------------------------
            # Create final result
            # ----------------------------------------------

            final_result = self.track_finalizer.finalize(
                track=track,
                person_candidates=ranked_candidates,
                face_candidates=[],
                fused_candidates=[],
            )

            self.final_results[
                track.track_id
            ] = final_result

            # ----------------------------------------------
            # Write audit record
            # ----------------------------------------------

            if (
                self.audit_writer is not None
                and self.session is not None
            ):

                audit_path = self.audit_writer.write(
                    final_result=final_result,
                    session_id=self.session.session_id,
                )

                self.session.add_audit_record(
                    str(audit_path)
                )

            # ----------------------------------------------
            # Update session statistics
            # ----------------------------------------------

            if self.session is not None:
                self.session.finalized_tracks += 1

            # ----------------------------------------------
            # Mark track finalized
            # ----------------------------------------------

            self.track_manager.mark_finalized(
                track.track_id
            )

    # ======================================================
    # Result accessors
    # ======================================================

    def get_track_candidates(self, track_id: int):

        return self.track_results.get(
            track_id,
            []
        )

    def get_ranked_candidates(self, track_id: int):

        return self.ranked_results.get(
            track_id,
            []
        )

    def get_final_result(self, track_id: int):

        return self.final_results.get(
            track_id
        )

    def get_all_final_results(self):

        return dict(
            self.final_results
        )

    # ======================================================
    # Session management
    # ======================================================

    def finish_session(self, session_writer):

        if self.session is None:
            raise RuntimeError(
                "No session is attached to the pipeline."
            )

        self.session.finish()

        return session_writer.write(
            self.session
        )

    def fail_session(
        self,
        session_writer,
        error: Optional[str] = None,
    ):

        if self.session is None:
            raise RuntimeError(
                "No session is attached to the pipeline."
            )

        self.session.fail(
            error=error
        )

        return session_writer.write(
            self.session
        )

    # ======================================================
    # Reset
    # ======================================================

    def reset(self):

        self.frame_number = 0

        self.track_results.clear()
        self.ranked_results.clear()
        self.final_results.clear()

        self.track_manager.reset()

        if self.candidate_history is not None:
            self.candidate_history.reset()

        self.tracker.reset()

