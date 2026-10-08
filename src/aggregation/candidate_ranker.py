
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RankedCandidate:

    subject_id: str

    ranking_score: float

    mean_similarity: float
    latest_similarity: float
    consistency: float

    observation_count: int
    mean_rank: float


class CandidateRanker:

    def __init__(
        self,
        mean_weight: float = 0.40,
        latest_weight: float = 0.20,
        consistency_weight: float = 0.25,
        repetition_weight: float = 0.15,
        minimum_observations: int = 3,
    ):

        total = (
            mean_weight
            + latest_weight
            + consistency_weight
            + repetition_weight
        )

        if not np.isclose(total, 1.0):
            raise ValueError(
                "Ranking weights must sum to 1.0"
            )

        self.mean_weight = mean_weight
        self.latest_weight = latest_weight
        self.consistency_weight = (
            consistency_weight
        )
        self.repetition_weight = (
            repetition_weight
        )
        self.minimum_observations = (
            minimum_observations
        )

    def _consistency(
        self,
        similarities,
    ) -> float:

        if len(similarities) <= 1:
            return 1.0

        std = float(
            np.std(similarities)
        )

        # Convert variation into a bounded
        # consistency score.
        consistency = 1.0 / (
            1.0 + std
        )

        return float(consistency)

    def rank(
        self,
        candidates,
    ) -> List[RankedCandidate]:

        ranked = []

        for candidate in candidates:

            if (
                candidate.observations
                < self.minimum_observations
            ):
                continue

            mean_similarity = (
                candidate.mean_similarity
            )

            latest_similarity = (
                candidate.latest_similarity
            )

            consistency = self._consistency(
                candidate.similarities
            )

            repetition_score = min(
                candidate.observations
                / (self.minimum_observations * 2),
                1.0,
            )

            score = (
                self.mean_weight
                * mean_similarity
                +
                self.latest_weight
                * latest_similarity
                +
                self.consistency_weight
                * consistency
                +
                self.repetition_weight
                * repetition_score
            )

            ranked.append(
                RankedCandidate(
                    subject_id=candidate.subject_id,
                    ranking_score=float(score),
                    mean_similarity=float(
                        mean_similarity
                    ),
                    latest_similarity=float(
                        latest_similarity
                    ),
                    consistency=float(
                        consistency
                    ),
                    observation_count=(
                        candidate.observations
                    ),
                    mean_rank=float(
                        candidate.mean_rank
                    ),
                )
            )

        ranked.sort(
            key=lambda x: x.ranking_score,
            reverse=True,
        )

        return ranked
