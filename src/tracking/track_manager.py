
from typing import Dict, List, Optional

from src.detection.models import Detection
from src.tracking.models import Track, TrackState


class TrackManager:
    """
    Manages the lifecycle of person tracks.

    Responsibilities:
    - Create tracks for new tracker IDs.
    - Add detections to existing tracks.
    - Mark missing tracks as LOST.
    - Terminate tracks after max_missed_frames.
    - Expose active, lost, terminated, and completed tracks.
    - Provide runtime diagnostics for debugging.
    """

    def __init__(
        self,
        max_missed_frames: int = 30,
        minimum_track_length: int = 5,
    ):
        self.max_missed_frames = max_missed_frames
        self.minimum_track_length = minimum_track_length

        self.tracks: Dict[int, Track] = {}

        # Tracks terminated during the current update cycle.
        self._newly_terminated: List[Track] = []

    def update(
        self,
        detections: List[Detection],
        frame_number: int,
    ) -> List[Track]:
        """
        Update track states using detections from
        the current frame.

        Returns:
            List of currently active tracks.
        """

        # Reset per-frame termination list.
        self._newly_terminated.clear()

        observed_track_ids = set()

        # --------------------------------------------------
        # 1. Process Current Detections
        # --------------------------------------------------

        for detection in detections:

            if detection.track_id is None:
                continue

            track_id = detection.track_id

            observed_track_ids.add(track_id)

            # Create a new track when a new tracker ID appears.
            if track_id not in self.tracks:

                self.tracks[track_id] = Track(
                    track_id=track_id
                )

            track = self.tracks[track_id]

            # Do not update finalized tracks.
            if track.state == TrackState.FINALIZED:
                continue

            track.add_detection(
                detection=detection,
                frame_number=frame_number,
            )

        # --------------------------------------------------
        # 2. Mark Missing Tracks as LOST
        # --------------------------------------------------

        for track_id, track in self.tracks.items():

            if track.state in (
                TrackState.TERMINATED,
                TrackState.FINALIZED,
            ):
                continue

            if track_id not in observed_track_ids:

                track.mark_missed()

        # --------------------------------------------------
        # 3. Terminate Expired Tracks
        # --------------------------------------------------

        self._terminate_expired_tracks()

        # --------------------------------------------------
        # 4. Collect Active Tracks
        # --------------------------------------------------

        active_tracks = self.get_active_tracks()

        # --------------------------------------------------
        # 5. Runtime Diagnostic
        # --------------------------------------------------

        print(
            f"[TrackManager] "
            f"frame={frame_number} "
            f"detections={len(detections)} "
            f"observed={sorted(observed_track_ids)} "
            f"active={[track.track_id for track in active_tracks]} "
            f"lost={[track.track_id for track in self.get_lost_tracks()]} "
            f"terminated={[track.track_id for track in self.get_terminated_tracks()]} "
            f"newly_terminated={[track.track_id for track in self.get_newly_terminated_tracks()]}"
        )

        return active_tracks

    def _terminate_expired_tracks(self):
        """
        Terminate tracks that have been missing for
        max_missed_frames or longer.
        """

        for track in self.tracks.values():

            if track.state != TrackState.LOST:
                continue

            if track.missed_frames >= self.max_missed_frames:

                track.terminate()

                self._newly_terminated.append(track)

    def get_active_tracks(self) -> List[Track]:
        """
        Return tracks currently considered active.
        """

        return [
            track
            for track in self.tracks.values()
            if track.state in (
                TrackState.NEW,
                TrackState.ACTIVE,
            )
        ]

    def get_lost_tracks(self) -> List[Track]:
        """
        Return tracks currently in LOST state.
        """

        return [
            track
            for track in self.tracks.values()
            if track.state == TrackState.LOST
        ]

    def get_terminated_tracks(self) -> List[Track]:
        """
        Return tracks currently terminated but
        not finalized.
        """

        return [
            track
            for track in self.tracks.values()
            if track.state == TrackState.TERMINATED
        ]

    def get_newly_terminated_tracks(self) -> List[Track]:
        """
        Return tracks terminated during the
        current update cycle.
        """

        return list(self._newly_terminated)

    def get_track(
        self,
        track_id: int,
    ) -> Optional[Track]:
        """
        Return a specific track by ID.
        """

        return self.tracks.get(track_id)

    def get_completed_tracks(self) -> List[Track]:
        """
        Return tracks that satisfy the minimum
        observation requirement and are terminated
        or finalized.
        """

        return [
            track
            for track in self.tracks.values()
            if (
                track.observation_count
                >= self.minimum_track_length
                and track.state
                in (
                    TrackState.TERMINATED,
                    TrackState.FINALIZED,
                )
            )
        ]

    def mark_finalized(
        self,
        track_id: int,
    ):
        """
        Mark a terminated track as finalized.

        Returns:
            True if successful, otherwise False.
        """

        track = self.tracks.get(track_id)

        if track is None:
            return False

        if track.state != TrackState.TERMINATED:
            return False

        track.finalize()

        return True

    def reset(self):
        """
        Reset all track state.
        """

        self.tracks.clear()
        self._newly_terminated.clear()
