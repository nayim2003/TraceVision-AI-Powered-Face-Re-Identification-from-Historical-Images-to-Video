
from typing import Optional

from src.aggregation.track_result import (
    FinalTrackResult,
)


class TrackFinalizer:

    def __init__(
        self,
        model_version: str = "0.1.0",
    ):
        self.model_version = model_version

    def finalize(
        self,
        track,
        person_candidates=None,
        face_candidates=None,
        fused_candidates=None,
    ) -> FinalTrackResult:

        person_candidates = (
            person_candidates
            if person_candidates is not None
            else []
        )

        face_candidates = (
            face_candidates
            if face_candidates is not None
            else []
        )

        fused_candidates = (
            fused_candidates
            if fused_candidates is not None
            else []
        )

        # --------------------------------------------------
        # Final identity candidate
        #
        # Only a fused candidate can become the final
        # identity result.
        #
        # Person-only and face-only candidates remain
        # supporting evidence and are not treated as
        # final identity.
        # --------------------------------------------------

        best_candidate: Optional[object] = None

        if fused_candidates:
            best_candidate = fused_candidates[0]

        return FinalTrackResult(
            track_id=track.track_id,

            first_frame=track.first_frame,

            last_frame=track.last_frame,

            observation_count=(
                track.observation_count
            ),

            person_embedding_count=(
                track.memory.person_memory_size
            ),

            face_embedding_count=(
                track.face_memory.size
            ),

            state=track.state.value,

            person_candidates=(
                person_candidates
            ),

            face_candidates=(
                face_candidates
            ),

            fused_candidates=(
                fused_candidates
            ),

            best_candidate=best_candidate,

            model_version=self.model_version,
        )
