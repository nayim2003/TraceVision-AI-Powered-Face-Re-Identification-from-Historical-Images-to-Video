
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid


@dataclass
class SessionRecord:
    """
    Metadata for one TraceVision processing session.
    """

    session_id: str = field(
        default_factory=lambda: uuid.uuid4().hex[:12]
    )

    source: Optional[str] = None

    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    ended_at: Optional[str] = None

    status: str = "running"

    error: Optional[str] = None

    processed_frames: int = 0

    finalized_tracks: int = 0

    model_versions: Dict = field(default_factory=dict)

    configuration: Dict = field(default_factory=dict)

    audit_records: List[str] = field(default_factory=list)

    def finish(self):
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.status = "completed"
        self.error = None

    def fail(self, error: Optional[str] = None):
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.status = "failed"
        self.error = error

    def add_audit_record(self, path: str):
        self.audit_records.append(str(path))

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "source": self.source,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "error": self.error,
            "processed_frames": self.processed_frames,
            "finalized_tracks": self.finalized_tracks,
            "model_versions": self.model_versions,
            "configuration": self.configuration,
            "audit_records": self.audit_records,
        }
