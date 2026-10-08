
from dataclasses import dataclass
from typing import Optional


@dataclass
class FusedCandidate:
    subject_id: str

    fusion_score: float

    person_score: Optional[float] = None
    face_score: Optional[float] = None

    person_observations: int = 0
    face_observations: int = 0

    person_rank: Optional[float] = None
    face_rank: Optional[float] = None
