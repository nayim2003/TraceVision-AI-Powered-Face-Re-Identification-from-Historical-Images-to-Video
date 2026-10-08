
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class FaceCandidateEvidence:
    """Temporal evidence for one face candidate."""

    subject_id: str

    similarities: List[float] = field(
        default_factory=list
    )

    ranks: List[int] = field(
        default_factory=list
    )

    observations: int = 0

    @property
    def mean_similarity(self) -> float:
        if not self.similarities:
            return 0.0

        return float(
            np.mean(self.similarities)
        )

    @property
    def max_similarity(self) -> float:
        if not self.similarities:
            return 0.0

        return float(
            np.max(self.similarities)
        )

    @property
    def latest_similarity(self) -> float:
        if not self.similarities:
            return 0.0

        return float(
            self.similarities[-1]
        )

    @property
    def mean_rank(self) -> float:
        if not self.ranks:
            return float("inf")

        return float(
            np.mean(self.ranks)
        )


class FaceCandidateHistory:
    """
    Maintains temporal candidate evidence
    for each tracked face.
    """

    def __init__(
        self,
        max_history_per_candidate: int = 20,
    ):
        self.max_history_per_candidate = (
            max_history_per_candidate
        )

        self.history: Dict[
            int,
            Dict[str, FaceCandidateEvidence]
        ] = {}

    def update(
        self,
        track_id: int,
        candidates,
    ) -> None:

        if track_id not in self.history:
            self.history[track_id] = {}

        track_history = self.history[
            track_id
        ]

        for candidate in candidates:

            subject_id = candidate.subject_id

            if subject_id not in track_history:
                track_history[subject_id] = (
                    FaceCandidateEvidence(
                        subject_id=subject_id
                    )
                )

            evidence = track_history[
                subject_id
            ]

            evidence.similarities.append(
                candidate.similarity
            )

            evidence.ranks.append(
                candidate.rank
            )

            evidence.observations += 1

            if (
                len(evidence.similarities)
                > self.max_history_per_candidate
            ):
                evidence.similarities = (
                    evidence.similarities[
                        -self.max_history_per_candidate:
                    ]
                )

                evidence.ranks = (
                    evidence.ranks[
                        -self.max_history_per_candidate:
                    ]
                )

    def get_candidates(
        self,
        track_id: int,
    ) -> List[FaceCandidateEvidence]:

        if track_id not in self.history:
            return []

        return list(
            self.history[track_id].values()
        )

    def clear_track(
        self,
        track_id: int,
    ) -> None:

        self.history.pop(
            track_id,
            None,
        )

    def reset(self) -> None:

        self.history.clear()
