
from src.recognition.face_memory import FaceTrackMemory
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from src.detection.models import Detection
from src.tracking.memory import TrackMemory


class TrackState(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    LOST = "lost"
    TERMINATED = "terminated"
    FINALIZED = "finalized"


@dataclass
class Track:
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    memory: TrackMemory = field(default_factory=TrackMemory)
    face_memory: FaceTrackMemory = field(
        default_factory=FaceTrackMemory
    )
    first_frame: int = -1
    last_frame: int = -1
    missed_frames: int = 0
    state: TrackState = TrackState.NEW

    @property
    def age(self) -> int:
        if self.first_frame < 0 or self.last_frame < 0:
            return 0

        return self.last_frame - self.first_frame + 1

    @property
    def observation_count(self) -> int:
        return len(self.detections)

    def add_detection(
        self,
        detection: Detection,
        frame_number: int,
    ):
        detection.track_id = self.track_id

        self.detections.append(detection)

        if self.first_frame < 0:
            self.first_frame = frame_number

        self.last_frame = frame_number
        self.missed_frames = 0

        if self.observation_count == 1:
            self.state = TrackState.NEW
        else:
            self.state = TrackState.ACTIVE

    def mark_missed(self):
        self.missed_frames += 1

        if self.state not in (
            TrackState.TERMINATED,
            TrackState.FINALIZED,
        ):
            self.state = TrackState.LOST

    def terminate(self):
        if self.state != TrackState.FINALIZED:
            self.state = TrackState.TERMINATED

    def finalize(self):
        self.state = TrackState.FINALIZED
