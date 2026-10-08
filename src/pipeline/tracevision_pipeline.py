
from typing import Dict, List, Optional

from src.aggregation.candidate_ranker import CandidateRanker
from src.aggregation.evidence_fusion import EvidenceFusion
from src.tracking.models import TrackState


class TraceVisionPipeline:
    """
    Unified TraceVision pipeline.

    Combines:

        Person Re-ID
              +
        Face Re-ID
              ↓
        Candidate Ranking
              ↓
        Evidence Fusion
              ↓
        Final Track Result
              ↓
        Audit Record

    The underlying PersonReIDPipeline is configured with
    auto_finalize=False so that tracks are finalized only
    after both person and face evidence have been considered.
    """

    def __init__(
        self,
        person_pipeline,
        face_pipeline,
        evidence_fusion: Optional[EvidenceFusion] = None,
        face_ranker: Optional[CandidateRanker] = None,
    ):
        self.person_pipeline = person_pipeline
        self.face_pipeline = face_pipeline

        self.evidence_fusion = (
            evidence_fusion
            if evidence_fusion is not None
            else EvidenceFusion(
                person_weight=0.40,
                face_weight=0.60,
            )
        )

        self.face_ranker = (
            face_ranker
            if face_ranker is not None
            else CandidateRanker(
                minimum_observations=3
            )
        )

        self.frame_number = 0

        self.fused_results: Dict[int, List] = {}

        # Prevent the person pipeline from finalizing tracks
        # before face evidence is available.
        self.person_pipeline.auto_finalize = False

    # ======================================================
    # Frame processing
    # ======================================================

    def process_frame(self, frame):

        self.frame_number += 1

        # --------------------------------------------------
        # 1. Person Re-ID pipeline
        # --------------------------------------------------

        active_tracks = (
            self.person_pipeline.process_frame(
                frame
            )
        )

        # --------------------------------------------------
        # 2. Face Re-ID pipeline
        # --------------------------------------------------

        self.face_pipeline.process_frame(
            frame=frame,
            active_tracks=active_tracks,
        )

        # --------------------------------------------------
        # 3. Finalize tracks that naturally terminated
        # --------------------------------------------------

        self._finalize_terminated_tracks()

        return active_tracks

    # ======================================================
    # Unified track finalization
    # ======================================================

    def _finalize_tracks(self, tracks):

        track_manager = self.person_pipeline.track_manager

        for track in tracks:

            # ------------------------------------------------
            # Ignore already finalized tracks
            # ------------------------------------------------

            if track.state == TrackState.FINALIZED:
                continue

            # ------------------------------------------------
            # A final result can only be produced for a
            # terminated track.
            # ------------------------------------------------

            if track.state != TrackState.TERMINATED:
                continue

            # ------------------------------------------------
            # Ignore tracks that are too short
            # ------------------------------------------------

            if (
                track.observation_count
                < track_manager.minimum_track_length
            ):

                track_manager.mark_finalized(
                    track.track_id
                )

                continue

            # ------------------------------------------------
            # Person candidate evidence
            # ------------------------------------------------

            person_candidates = (
                self.person_pipeline
                .get_ranked_candidates(
                    track.track_id
                )
            )

            # ------------------------------------------------
            # Face candidate evidence
            # ------------------------------------------------

            face_history = (
                self.face_pipeline
                .get_track_history(
                    track.track_id
                )
            )

            face_candidates = (
                self.face_ranker.rank(
                    face_history
                )
            )

            # ------------------------------------------------
            # Evidence fusion
            # ------------------------------------------------

            fused_candidates = (
                self.evidence_fusion.fuse(
                    person_candidates=person_candidates,
                    face_candidates=face_candidates,
                )
            )

            self.fused_results[
                track.track_id
            ] = fused_candidates

            # ------------------------------------------------
            # Build final result
            # ------------------------------------------------

            final_result = (
                self.person_pipeline
                .track_finalizer
                .finalize(
                    track=track,
                    person_candidates=person_candidates,
                    face_candidates=face_candidates,
                    fused_candidates=fused_candidates,
                )
            )

            # ------------------------------------------------
            # Store final result
            # ------------------------------------------------

            self.person_pipeline.final_results[
                track.track_id
            ] = final_result

            # ------------------------------------------------
            # Write audit record
            # ------------------------------------------------

            session = self.person_pipeline.session
            audit_writer = self.person_pipeline.audit_writer

            if (
                audit_writer is not None
                and session is not None
            ):

                audit_path = audit_writer.write(
                    final_result=final_result,
                    session_id=session.session_id,
                )

                session.add_audit_record(
                    str(audit_path)
                )

            # ------------------------------------------------
            # Update session statistics
            # ------------------------------------------------

            if session is not None:
                session.finalized_tracks += 1

            # ------------------------------------------------
            # Mark track finalized
            # ------------------------------------------------

            track_manager.mark_finalized(
                track.track_id
            )

    def _finalize_terminated_tracks(self):

        terminated_tracks = (
            self.person_pipeline
            .track_manager
            .get_newly_terminated_tracks()
        )

        self._finalize_tracks(
            terminated_tracks
        )

    # ======================================================
    # End-of-video flush
    # ======================================================

    def finish(self):
        """
        Finalize all remaining tracks at the end of a video.

        Tracks that are still ACTIVE, NEW, or LOST when the
        video ends cannot wait for additional frames. They are
        therefore converted to TERMINATED and passed through
        the normal person + face evidence fusion pipeline.
        """

        track_manager = self.person_pipeline.track_manager

        remaining_tracks = []

        # --------------------------------------------------
        # Collect every non-finalized track
        # --------------------------------------------------

        for track in track_manager.tracks.values():

            if track.state == TrackState.FINALIZED:
                continue

            remaining_tracks.append(track)

        # --------------------------------------------------
        # Force remaining tracks into TERMINATED state
        # --------------------------------------------------

        for track in remaining_tracks:

            if track.state != TrackState.TERMINATED:

                track.terminate()

        # --------------------------------------------------
        # Finalize all remaining tracks
        # --------------------------------------------------

        self._finalize_tracks(
            remaining_tracks
        )

        return self.get_all_final_results()

    # ======================================================
    # Result access
    # ======================================================

    def get_track_candidates(
        self,
        track_id: int,
    ):

        return (
            self.person_pipeline
            .get_track_candidates(
                track_id
            )
        )

    def get_person_ranked_candidates(
        self,
        track_id: int,
    ):

        return (
            self.person_pipeline
            .get_ranked_candidates(
                track_id
            )
        )

    def get_face_candidates(
        self,
        track_id: int,
    ):

        face_history = (
            self.face_pipeline
            .get_track_history(
                track_id
            )
        )

        return self.face_ranker.rank(
            face_history
        )

    def get_fused_candidates(
        self,
        track_id: int,
    ):

        return self.fused_results.get(
            track_id,
            []
        )

    def get_final_result(
        self,
        track_id: int,
    ):

        return (
            self.person_pipeline
            .get_final_result(
                track_id
            )
        )

    def get_all_final_results(self):

        return (
            self.person_pipeline
            .get_all_final_results()
        )

    # ======================================================
    # Session management
    # ======================================================

    @property
    def session(self):

        return self.person_pipeline.session

    def finish_session(
        self,
        session_writer,
    ):

        return (
            self.person_pipeline
            .finish_session(
                session_writer
            )
        )

    def fail_session(
        self,
        session_writer,
        error: Optional[str] = None,
    ):

        return (
            self.person_pipeline
            .fail_session(
                session_writer,
                error=error,
            )
        )

    # ======================================================
    # Reset
    # ======================================================

    def reset(self):

        self.frame_number = 0

        self.fused_results.clear()

        self.person_pipeline.reset()
        self.face_pipeline.reset()
