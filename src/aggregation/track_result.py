
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class FinalTrackResult:
    """
    Final result for one completed person track.

    A track is considered matched only when a fused
    person + face candidate exists.

    This result is a decision record, not a probability.
    """

    track_id: int
    first_frame: int
    last_frame: int
    observation_count: int

    person_embedding_count: int
    face_embedding_count: int

    state: str

    person_candidates: List = field(
        default_factory=list
    )

    face_candidates: List = field(
        default_factory=list
    )

    fused_candidates: List = field(
        default_factory=list
    )

    best_candidate: Optional[Any] = None

    model_version: str = "0.1.0"

    @property
    def duration_frames(self) -> int:
        if (
            self.first_frame < 0
            or self.last_frame < 0
        ):
            return 0

        return (
            self.last_frame
            - self.first_frame
            + 1
        )

    @property
    def has_candidates(self) -> bool:
        """
        Indicates whether any retrieval evidence exists.

        This does NOT mean that a final identity
        was established.
        """

        return bool(
            self.person_candidates
            or self.face_candidates
            or self.fused_candidates
        )

    @property
    def decision(self) -> str:
        """
        Final identity decision.

        A final match requires a fused candidate.
        """

        if self.best_candidate is not None:
            return "matched"

        return "unknown"

    @property
    def subject_id(self) -> Optional[str]:
        """
        Return the final matched subject ID.

        Returns None when no final identity
        candidate exists.
        """

        if self.best_candidate is None:
            return None

        return self.best_candidate.subject_id

    @property
    def fusion_score(self) -> Optional[float]:
        """
        Return the final fusion score.

        Returns None when no fused candidate exists.
        """

        if self.best_candidate is None:
            return None

        return float(
            self.best_candidate.fusion_score
        )

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "duration_frames": self.duration_frames,
            "observation_count": self.observation_count,
            "person_embedding_count": (
                self.person_embedding_count
            ),
            "face_embedding_count": (
                self.face_embedding_count
            ),
            "state": self.state,

            "decision": self.decision,

            "subject_id": self.subject_id,

            "fusion_score": self.fusion_score,

            "has_candidates": self.has_candidates,

            "best_candidate": (
                self.best_candidate.__dict__
                if self.best_candidate is not None
                else None
            ),

            "person_candidates": [
                candidate.__dict__
                for candidate in self.person_candidates
            ],

            "face_candidates": [
                candidate.__dict__
                for candidate in self.face_candidates
            ],

            "fused_candidates": [
                candidate.__dict__
                for candidate in self.fused_candidates
            ],

            "model_version": self.model_version,
        }
