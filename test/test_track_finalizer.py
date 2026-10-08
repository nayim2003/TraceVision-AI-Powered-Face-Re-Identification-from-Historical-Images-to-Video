
import pytest
from enum import Enum

from src.aggregation.track_finalizer import TrackFinalizer
from src.aggregation.fusion_models import FusedCandidate
from src.aggregation.track_result import FinalTrackResult


class MockTrackState(Enum):
    NEW = "new"
    ACTIVE = "active"
    TERMINATED = "terminated"


class MockTrackMemory:
    def __init__(self, person_memory_size=20):
        self.person_embeddings = [
            object() for _ in range(person_memory_size)
        ]

    @property
    def person_memory_size(self):
        return len(self.person_embeddings)


class MockFaceMemory:
    def __init__(self, size=20):
        self.size = size


class MockTrack:
    def __init__(
        self,
        track_id=1,
        first_frame=10,
        last_frame=109,
        observation_count=100,
        person_embedding_count=20,
        face_embedding_count=20,
        state=MockTrackState.TERMINATED,
    ):
        self.track_id = track_id
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.observation_count = observation_count
        self.memory = MockTrackMemory(person_embedding_count)
        self.face_memory = MockFaceMemory(face_embedding_count)
        self.state = state


def test_track_finalizer_returns_matched_result():
    finalizer = TrackFinalizer()
    track = MockTrack(track_id=5)

    candidate = FusedCandidate(
        subject_id="subject_001",
        fusion_score=0.82,
        person_score=0.75,
        face_score=0.87,
        person_observations=100,
        face_observations=80,
        person_rank=1.0,
        face_rank=1.0,
    )

    result = finalizer.finalize(
        track=track,
        person_candidates=[],
        face_candidates=[],
        fused_candidates=[candidate],
    )

    assert isinstance(result, FinalTrackResult)
    assert result.track_id == 5
    assert result.decision == "matched"
    assert result.subject_id == "subject_001"
    assert result.fusion_score == pytest.approx(0.82)


def test_track_finalizer_returns_unknown_without_fused_candidates():
    finalizer = TrackFinalizer()
    track = MockTrack(track_id=7)

    result = finalizer.finalize(
        track=track,
        person_candidates=[],
        face_candidates=[],
        fused_candidates=[],
    )

    assert isinstance(result, FinalTrackResult)
    assert result.track_id == 7
    assert result.decision == "unknown"
    assert result.subject_id is None
    assert result.fusion_score is None
    assert result.best_candidate is None


def test_track_finalizer_preserves_track_statistics():
    finalizer = TrackFinalizer()

    track = MockTrack(
        track_id=10,
        first_frame=25,
        last_frame=124,
        observation_count=100,
        person_embedding_count=20,
        face_embedding_count=18,
    )

    result = finalizer.finalize(
        track=track,
        person_candidates=[],
        face_candidates=[],
        fused_candidates=[],
    )

    assert result.track_id == 10
    assert result.first_frame == 25
    assert result.last_frame == 124
    assert result.duration_frames == 100
    assert result.observation_count == 100
    assert result.person_embedding_count == 20
    assert result.face_embedding_count == 18
