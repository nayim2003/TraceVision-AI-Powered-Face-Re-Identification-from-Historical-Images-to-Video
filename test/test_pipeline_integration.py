
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.aggregation.candidate_history import CandidateEvidence
from src.aggregation.candidate_ranker import (
    CandidateRanker,
    RankedCandidate,
)
from src.aggregation.evidence_fusion import EvidenceFusion
from src.pipeline.tracevision_pipeline import TraceVisionPipeline
from src.tracking.models import TrackState


class MockTrackManager:

    def __init__(self):
        self.minimum_track_length = 5
        self.tracks = {}
        self.finalized_track_ids = set()

    def get_newly_terminated_tracks(self):
        return [
            track
            for track in self.tracks.values()
            if (
                track.state == TrackState.TERMINATED
                and track.track_id
                not in self.finalized_track_ids
            )
        ]

    def mark_finalized(self, track_id):
        self.finalized_track_ids.add(track_id)


class MockTrack:

    def __init__(
        self,
        track_id=1,
        observation_count=10,
        state=TrackState.ACTIVE,
    ):
        self.track_id = track_id
        self.observation_count = observation_count
        self.state = state

    def terminate(self):
        self.state = TrackState.TERMINATED


class MockFinalizer:

    def finalize(
        self,
        track,
        person_candidates,
        face_candidates,
        fused_candidates,
    ):

        best_candidate = (
            fused_candidates[0]
            if fused_candidates
            else None
        )

        return SimpleNamespace(
            track_id=track.track_id,
            decision=(
                "matched"
                if best_candidate is not None
                else "unknown"
            ),
            subject_id=(
                best_candidate.subject_id
                if best_candidate is not None
                else None
            ),
            fusion_score=(
                best_candidate.fusion_score
                if best_candidate is not None
                else None
            ),
        )


class MockPersonPipeline:

    def __init__(self, track):

        self.track_manager = MockTrackManager()

        self.track = track

        self.track_manager.tracks[
            track.track_id
        ] = track

        self.auto_finalize = True

        self.session = None
        self.audit_writer = None

        self.track_finalizer = MockFinalizer()

        self.final_results = {}

        self.process_calls = 0

        self.person_candidates = {
            track.track_id: [
                RankedCandidate(
                    subject_id="subject_001",
                    ranking_score=0.80,
                    mean_similarity=0.80,
                    latest_similarity=0.80,
                    consistency=1.0,
                    observation_count=10,
                    mean_rank=1.0,
                )
            ]
        }

    def process_frame(self, frame):

        self.process_calls += 1

        return [self.track]

    def get_ranked_candidates(self, track_id):

        return self.person_candidates.get(
            track_id,
            [],
        )

    def get_track_candidates(self, track_id):

        return self.get_ranked_candidates(
            track_id
        )

    def get_final_result(self, track_id):

        return self.final_results.get(
            track_id
        )

    def get_all_final_results(self):

        return list(
            self.final_results.values()
        )

    def reset(self):

        self.process_calls = 0


class MockFacePipeline:

    def __init__(self):

        self.process_calls = 0

        self.face_history = {
            1: [
                CandidateEvidence(
                    subject_id="subject_001",
                    similarities=[
                        0.90,
                        0.88,
                        0.86,
                    ],
                    ranks=[
                        1,
                        1,
                        1,
                    ],
                    observations=3,
                )
            ]
        }

    def process_frame(
        self,
        frame,
        active_tracks,
    ):

        self.process_calls += 1

        assert active_tracks

    def get_track_history(self, track_id):

        return self.face_history.get(
            track_id,
            [],
        )

    def reset(self):

        self.process_calls = 0


def build_test_pipeline():

    track = MockTrack(
        track_id=1,
        observation_count=10,
        state=TrackState.ACTIVE,
    )

    person_pipeline = MockPersonPipeline(
        track=track
    )

    face_pipeline = MockFacePipeline()

    evidence_fusion = EvidenceFusion(
        person_weight=0.40,
        face_weight=0.60,
    )

    face_ranker = CandidateRanker(
        minimum_observations=3,
    )

    pipeline = TraceVisionPipeline(
        person_pipeline=person_pipeline,
        face_pipeline=face_pipeline,
        evidence_fusion=evidence_fusion,
        face_ranker=face_ranker,
    )

    return (
        pipeline,
        person_pipeline,
        face_pipeline,
        track,
    )


def test_tracevision_pipeline_processes_frame():

    (
        pipeline,
        person_pipeline,
        face_pipeline,
        track,
    ) = build_test_pipeline()

    frame = Mock()

    active_tracks = pipeline.process_frame(
        frame
    )

    assert active_tracks == [track]

    assert pipeline.frame_number == 1

    assert person_pipeline.process_calls == 1

    assert face_pipeline.process_calls == 1


def test_tracevision_pipeline_finishes_and_fuses_evidence():

    (
        pipeline,
        person_pipeline,
        face_pipeline,
        track,
    ) = build_test_pipeline()

    track.terminate()

    results = pipeline.finish()

    assert track.track_id in (
        person_pipeline.final_results
    )

    assert track.track_id in (
        pipeline.fused_results
    )

    assert len(results) == 1

    result = results[0]

    assert result.track_id == 1

    assert result.decision == "matched"

    assert result.subject_id == "subject_001"

    fused_candidate = (
        pipeline.fused_results[track.track_id][0]
    )

    assert result.fusion_score == pytest.approx(
        fused_candidate.fusion_score,
        abs=1e-6,
    )


def test_tracevision_pipeline_reset():

    (
        pipeline,
        person_pipeline,
        face_pipeline,
        _,
    ) = build_test_pipeline()

    pipeline.process_frame(
        Mock()
    )

    pipeline.reset()

    assert pipeline.frame_number == 0

    assert pipeline.fused_results == {}

    assert person_pipeline.process_calls == 0

    assert face_pipeline.process_calls == 0
