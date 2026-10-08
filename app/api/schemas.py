

from typing import Any, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class TrackSummary(BaseModel):
    track_id: int
    state: str
    observation_count: int
    person_embedding_count: int
    face_embedding_count: int
    has_candidates: bool

    # Final decision
    decision: Optional[str] = None
    subject_id: Optional[str] = None
    fusion_score: Optional[float] = None


class CandidateResponse(BaseModel):
    subject_id: str
    fusion_score: Optional[float] = None
    person_score: Optional[float] = None
    face_score: Optional[float] = None


class TrackResultResponse(BaseModel):
    track_id: int
    state: str
    observation_count: int
    person_embedding_count: int
    face_embedding_count: int
    has_candidates: bool

    # Optional final decision fields
    decision: Optional[str] = None
    subject_id: Optional[str] = None
    fusion_score: Optional[float] = None

    best_candidate: Optional[Any] = None

