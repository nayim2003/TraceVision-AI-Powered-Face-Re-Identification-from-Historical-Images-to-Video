
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalCandidate:
    rank: int
    subject_id: str
    similarity: float
    reference_image: Optional[str] = None
    source: Optional[str] = None
