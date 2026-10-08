
from dataclasses import dataclass
from typing import Dict


@dataclass
class RunSummary:
    """Summary of one TraceVision video-processing run."""

    session_id: str
    status: str
    total_frames: int
    processed_frames: int
    finalized_tracks: int
    audit_records: int
    session_path: str

    @property
    def processing_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0

        return (
            self.processed_frames
            / self.total_frames
        )

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "finalized_tracks": self.finalized_tracks,
            "audit_records": self.audit_records,
            "processing_ratio": self.processing_ratio,
            "session_path": self.session_path,
        }

    @classmethod
    def from_runner_result(
        cls,
        result: Dict,
        audit_records: int,
    ):
        return cls(
            session_id=result["session_id"],
            status=result["status"],
            total_frames=result["total_frames"],
            processed_frames=result["processed_frames"],
            finalized_tracks=result["finalized_tracks"],
            audit_records=audit_records,
            session_path=result["session_path"],
        )
