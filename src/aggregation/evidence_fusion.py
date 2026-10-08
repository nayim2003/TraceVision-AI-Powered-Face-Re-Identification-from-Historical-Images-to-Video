
from typing import List, Dict

from src.aggregation.fusion_models import FusedCandidate


class EvidenceFusion:
    """
    Combines person Re-ID and face Re-ID evidence.

    A candidate is fused only when both person and face
    evidence are available.

    The output is a candidate ranking score,
    not an identity probability.
    """

    def __init__(
        self,
        person_weight: float = 0.40,
        face_weight: float = 0.60,
    ):
        total = (
            person_weight
            + face_weight
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Fusion weights must sum to 1.0"
            )

        self.person_weight = person_weight
        self.face_weight = face_weight

    def fuse(
        self,
        person_candidates,
        face_candidates,
    ) -> List[FusedCandidate]:

        person_map: Dict[str, object] = {
            candidate.subject_id: candidate
            for candidate in person_candidates
        }

        face_map: Dict[str, object] = {
            candidate.subject_id: candidate
            for candidate in face_candidates
        }

        fused = []

        # --------------------------------------------------
        # Only candidates supported by BOTH modalities
        # participate in evidence fusion.
        # --------------------------------------------------

        common_subject_ids = (
            set(person_map)
            & set(face_map)
        )

        for subject_id in common_subject_ids:

            person = person_map[subject_id]
            face = face_map[subject_id]

            person_score = (
                float(person.ranking_score)
            )

            face_score = (
                float(face.ranking_score)
            )

            fusion_score = (
                self.person_weight
                * person_score
                +
                self.face_weight
                * face_score
            )

            fused.append(
                FusedCandidate(
                    subject_id=subject_id,

                    fusion_score=float(
                        fusion_score
                    ),

                    person_score=(
                        person_score
                    ),

                    face_score=(
                        face_score
                    ),

                    person_observations=(
                        person.observation_count
                    ),

                    face_observations=(
                        face.observation_count
                    ),

                    person_rank=(
                        person.mean_rank
                    ),

                    face_rank=(
                        face.mean_rank
                    ),
                )
            )

        fused.sort(
            key=lambda x: x.fusion_score,
            reverse=True,
        )

        return fused
