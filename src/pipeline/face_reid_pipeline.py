
from typing import Dict, List

import numpy as np

from src.recognition.base_face_detector import BaseFaceDetector
from src.recognition.face_crop import FaceCropExtractor
from src.recognition.face_quality import FaceQualityAnalyzer
from src.recognition.base_face_embedder import BaseFaceEmbedder
from src.recognition.face_memory_manager import FaceMemoryManager
from src.aggregation.face_embedding_aggregator import FaceEmbeddingAggregator
from src.retrieval.face_retrieval_service import FaceRetrievalService
from src.aggregation.face_candidate_history import FaceCandidateHistory
from src.recognition.face_track_associator import FaceTrackAssociator


class FaceReIDPipeline:
    """
    Face detection, face embedding, memory, retrieval,
    and candidate-history pipeline.

    The pipeline operates on existing person tracks
    produced by the person tracking pipeline.

    InsightFaceDetector already produces a normalized
    ArcFace embedding for each detected face. That
    upstream embedding is used directly.

    The crop extractor is retained for face-quality
    analysis and future fallback processing.
    """

    def __init__(
        self,
        face_detector: BaseFaceDetector,
        crop_extractor: FaceCropExtractor,
        quality_analyzer: FaceQualityAnalyzer,
        embedder: BaseFaceEmbedder,
        memory_manager: FaceMemoryManager,
        aggregator: FaceEmbeddingAggregator,
        retrieval_service: FaceRetrievalService,
        candidate_history: FaceCandidateHistory,
    ):
        self.face_detector = face_detector
        self.crop_extractor = crop_extractor
        self.quality_analyzer = quality_analyzer
        self.embedder = embedder
        self.memory_manager = memory_manager
        self.aggregator = aggregator
        self.retrieval_service = retrieval_service
        self.candidate_history = candidate_history

        # Face-to-person association
        self.associator = FaceTrackAssociator()

        # Runtime state
        self.frame_number = 0

        # Latest retrieval results per track
        self.track_results: Dict[int, List] = {}

        # Runtime statistics
        self.detected_faces = 0
        self.associated_faces = 0
        self.processed_faces = 0
        self.rejected_faces = 0
        self.failed_embeddings = 0

    def process_frame(self, frame, active_tracks):
        """
        Process one video frame using the currently active
        person tracks.

        Pipeline:

            Frame
              ↓
            Face Detection + Native ArcFace Embedding
              ↓
            Face ↔ Person Track Association
              ↓
            Face Crop
              ↓
            Quality Analysis
              ↓
            Native Face Embedding
              ↓
            Track Face Memory
              ↓
            Track-level Face Embedding
              ↓
            FAISS Retrieval
              ↓
            Candidate History
        """

        self.frame_number += 1

        # --------------------------------------------------
        # Reset per-frame statistics
        # --------------------------------------------------

        self.detected_faces = 0
        self.associated_faces = 0
        self.processed_faces = 0
        self.rejected_faces = 0
        self.failed_embeddings = 0

        # --------------------------------------------------
        # 1. Face Detection
        # --------------------------------------------------

        faces = self.face_detector.detect(frame)

        self.detected_faces = len(faces)

        if not faces:
            return self._build_frame_result()

        # --------------------------------------------------
        # 2. Collect Current Person Detections
        # --------------------------------------------------

        person_detections = []

        for track in active_tracks:

            if not track.detections:
                continue

            person_detections.append(
                track.detections[-1]
            )

        if not person_detections:
            return self._build_frame_result()

        # --------------------------------------------------
        # 3. Associate Faces with Person Tracks
        # --------------------------------------------------

        associated_faces = self.associator.associate(
            faces=faces,
            person_detections=person_detections,
        )

        self.associated_faces = sum(
            1
            for face in associated_faces
            if face.track_id is not None
        )

        if not associated_faces:
            return self._build_frame_result()

        # --------------------------------------------------
        # 4. Process Each Associated Face
        # --------------------------------------------------

        for face in associated_faces:

            # ----------------------------------------------
            # 4.1 Validate Track ID
            # ----------------------------------------------

            if face.track_id is None:
                continue

            track = next(
                (
                    current_track
                    for current_track in active_tracks
                    if current_track.track_id == face.track_id
                ),
                None,
            )

            if track is None:
                continue

            # ----------------------------------------------
            # 4.2 Extract Face Crop
            #
            # Used for quality analysis only.
            # We do NOT run InsightFace again on this crop.
            # ----------------------------------------------

            face_crop = self.crop_extractor.extract(
                frame,
                face,
            )

            if face_crop is None:
                self.rejected_faces += 1
                continue

            # ----------------------------------------------
            # 4.3 Face Quality Analysis
            # ----------------------------------------------

            quality = self.quality_analyzer.analyze(
                face_crop.image,
                confidence=face.confidence,
            )

            if not quality.accepted:
                self.rejected_faces += 1
                continue

            # ----------------------------------------------
            # 4.4 Use Native InsightFace Embedding
            # ----------------------------------------------

            embedding = getattr(
                face,
                "embedding",
                None,
            )

            if embedding is None:
                self.failed_embeddings += 1
                continue

            try:
                embedding = np.asarray(
                    embedding,
                    dtype=np.float32,
                ).reshape(-1)

            except (TypeError, ValueError):
                self.failed_embeddings += 1
                continue

            # ----------------------------------------------
            # Validate Embedding Dimension
            # ----------------------------------------------

            if embedding.shape[0] != 512:
                self.failed_embeddings += 1
                continue

            # ----------------------------------------------
            # L2 Normalize
            # ----------------------------------------------

            norm = np.linalg.norm(
                embedding
            )

            if norm == 0:
                self.failed_embeddings += 1
                continue

            embedding = (
                embedding / norm
            ).astype(np.float32)

            # ----------------------------------------------
            # 4.5 Store Face Embedding in Track Memory
            # ----------------------------------------------

            stored = self.memory_manager.add_embedding(
                memory=track.face_memory,
                embedding=embedding,
                frame_number=self.frame_number,
                quality=quality.score,
            )

            if not stored:
                self.rejected_faces += 1
                continue

            self.processed_faces += 1

            # ----------------------------------------------
            # 4.6 Minimum Memory Requirement
            # ----------------------------------------------

            if track.face_memory.size < 3:
                continue

            # ----------------------------------------------
            # 4.7 Track-level Face Embedding
            # ----------------------------------------------

            track_embedding = self.aggregator.aggregate(
                track.face_memory
            )

            if track_embedding is None:
                continue

            # ----------------------------------------------
            # 4.8 Face Retrieval
            # ----------------------------------------------

            candidates = self.retrieval_service.search(
                track_embedding
            )

            self.track_results[
                track.track_id
            ] = candidates

            # ----------------------------------------------
            # 4.9 Candidate History
            # ----------------------------------------------

            self.candidate_history.update(
                track_id=track.track_id,
                candidates=candidates,
            )

        # --------------------------------------------------
        # 5. Return Frame Result
        # --------------------------------------------------

        return self._build_frame_result()

    def _build_frame_result(self) -> dict:
        """
        Build a structured per-frame processing summary.
        """

        return {
            "frame_number": self.frame_number,
            "detected_faces": self.detected_faces,
            "associated_faces": self.associated_faces,
            "processed_faces": self.processed_faces,
            "rejected_faces": self.rejected_faces,
            "failed_embeddings": self.failed_embeddings,
        }

    def get_track_candidates(self, track_id: int):
        """
        Return the latest face retrieval candidates
        for a specific track.
        """

        return self.track_results.get(
            track_id,
            [],
        )

    def get_track_history(self, track_id: int):
        """
        Return candidate history for a specific track.
        """

        return self.candidate_history.get_candidates(
            track_id
        )

    def get_track_memory_size(self, track_id: int) -> int:
        """
        Return the number of stored face embeddings
        for a specific track.
        """

        return 0

    def reset(self):
        """
        Reset the complete face pipeline state.
        """

        self.frame_number = 0

        self.track_results.clear()

        self.detected_faces = 0
        self.associated_faces = 0
        self.processed_faces = 0
        self.rejected_faces = 0
        self.failed_embeddings = 0

        self.candidate_history.reset()
