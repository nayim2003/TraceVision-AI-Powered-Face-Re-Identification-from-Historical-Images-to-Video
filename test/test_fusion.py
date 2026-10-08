
import pytest

from src.aggregation.evidence_fusion import EvidenceFusion


class MockCandidate:
    def __init__(
        self,
        subject_id,
        ranking_score,
        observation_count=5,
        mean_rank=1.0,
    ):
        self.subject_id = subject_id
        self.ranking_score = ranking_score
        self.observation_count = observation_count
        self.mean_rank = mean_rank


def test_fusion_requires_both_evidence_sources():

    fusion = EvidenceFusion(
        person_weight=0.40,
        face_weight=0.60,
    )

    person_candidates = [
        MockCandidate(
            subject_id="subject_001",
            ranking_score=0.80,
        )
    ]

    face_candidates = []

    result = fusion.fuse(
        person_candidates=person_candidates,
        face_candidates=face_candidates,
    )

    assert result == []


def test_fusion_combines_common_candidate():

    fusion = EvidenceFusion(
        person_weight=0.40,
        face_weight=0.60,
    )

    person_candidates = [
        MockCandidate(
            subject_id="subject_001",
            ranking_score=0.40,
            observation_count=10,
            mean_rank=2.0,
        )
    ]

    face_candidates = [
        MockCandidate(
            subject_id="subject_001",
            ranking_score=0.90,
            observation_count=8,
            mean_rank=1.0,
        )
    ]

    result = fusion.fuse(
        person_candidates=person_candidates,
        face_candidates=face_candidates,
    )

    assert len(result) == 1

    candidate = result[0]

    assert candidate.subject_id == "subject_001"

    expected_score = (
        0.40 * 0.40
        + 0.60 * 0.90
    )

    assert candidate.fusion_score == pytest.approx(
        expected_score
    )


def test_fusion_prefers_higher_combined_score():

    fusion = EvidenceFusion(
        person_weight=0.40,
        face_weight=0.60,
    )

    person_candidates = [
        MockCandidate(
            subject_id="subject_001",
            ranking_score=0.50,
        ),
        MockCandidate(
            subject_id="subject_002",
            ranking_score=0.80,
        ),
    ]

    face_candidates = [
        MockCandidate(
            subject_id="subject_001",
            ranking_score=0.90,
        ),
        MockCandidate(
            subject_id="subject_002",
            ranking_score=0.60,
        ),
    ]

    result = fusion.fuse(
        person_candidates=person_candidates,
        face_candidates=face_candidates,
    )

    assert len(result) == 2

    assert result[0].subject_id == "subject_001"

    assert (
        result[0].fusion_score
        > result[1].fusion_score
    )
