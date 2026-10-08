
# src/pipeline/factory.py

from src.pipeline.tracevision_pipeline import TraceVisionPipeline
from src.pipeline.person_reid_pipeline import PersonReIDPipeline
from src.pipeline.face_reid_pipeline import FaceReIDPipeline

# ---------------------------------------------------------
# Person tracking
# ---------------------------------------------------------

from src.tracking.bytetrack_tracker import ByteTrackTracker
from src.tracking.track_manager import TrackManager

# ---------------------------------------------------------
# Person recognition
# ---------------------------------------------------------

from src.recognition.person_crop import PersonCropExtractor
from src.recognition.quality import CropQualityAnalyzer
from src.recognition.osnet_reid import OSNetReIDExtractor
from src.tracking.memory_manager import TrackMemoryManager

# ---------------------------------------------------------
# Person aggregation / ranking
# ---------------------------------------------------------

from src.aggregation.track_embedding import TrackEmbeddingAggregator
from src.aggregation.candidate_history import CandidateHistory
from src.aggregation.candidate_ranker import CandidateRanker

# ---------------------------------------------------------
# Face recognition
# ---------------------------------------------------------

from src.recognition.insightface_detector import InsightFaceDetector
from src.recognition.face_crop import FaceCropExtractor
from src.recognition.face_quality import FaceQualityAnalyzer
from src.recognition.arcface_embedder import ArcFaceEmbedder
from src.recognition.face_memory_manager import FaceMemoryManager

from src.aggregation.face_embedding_aggregator import (
    FaceEmbeddingAggregator
)

from src.aggregation.face_candidate_history import (
    FaceCandidateHistory
)

# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

from src.retrieval.faiss_index import FAISSRetriever
from src.retrieval.face_faiss_index import FaceFAISSIndex

from src.retrieval.retrieval_service import (
    RetrievalService
)

from src.retrieval.face_retrieval_service import (
    FaceRetrievalService
)

# ---------------------------------------------------------
# Gallery
# ---------------------------------------------------------

from src.storage.gallery import (
    GalleryIdentity,
    ReferenceGallery,
)

from src.storage.gallery_loader import (
    GalleryEmbeddingLoader
)

from src.retrieval.gallery_indexer import (
    ReferenceGalleryIndexer
)

# ---------------------------------------------------------
# Evidence fusion
# ---------------------------------------------------------

from src.aggregation.evidence_fusion import (
    EvidenceFusion
)


def build_tracevision_pipeline(
    *,
    session=None,
    audit_writer=None,
):
    """
    Build the complete TraceVision application pipeline.

    Architecture:

        Video
          ↓
        Person Detection
          ↓
        ByteTrack
          ↓
        Person Re-ID
          ↓
        Face Detection
          ↓
        Face Re-ID
          ↓
        Candidate Ranking
          ↓
        Evidence Fusion
          ↓
        Final Track Result
          ↓
        Audit Record

    The factory is intentionally self-contained so that
    the application does not depend on notebook variables.
    """

    # =====================================================
    # 1. PERSON TRACKING
    # =====================================================

    person_tracker = ByteTrackTracker(
        model_name="yolo11n.pt",
        confidence_threshold=0.50,
        tracker_config="bytetrack.yaml",
    )

    track_manager = TrackManager(
        max_missed_frames=30,
        minimum_track_length=5,
    )

    # =====================================================
    # 2. PERSON PROCESSING COMPONENTS
    # =====================================================

    person_crop_extractor = PersonCropExtractor(
        target_width=128,
        target_height=256,
        padding_ratio=0.05,
    )

    person_quality_analyzer = CropQualityAnalyzer(
        min_width=40,
        min_height=80,
        min_blur_score=50,
        minimum_quality=0.60,
    )

    person_reid_extractor = OSNetReIDExtractor(
        model_name="osnet_x1_0",
        device="auto",
    )

    person_memory_manager = TrackMemoryManager(
        max_person_embeddings=20,
        max_face_embeddings=20,
        minimum_quality=0.60,
    )

    person_aggregator = TrackEmbeddingAggregator(
        minimum_quality=0.60,
        max_embeddings=20,
    )

    person_candidate_history = CandidateHistory(
        max_history_per_candidate=20,
    )

    person_candidate_ranker = CandidateRanker(
        minimum_observations=3,
    )

    # =====================================================
    # 3. REFERENCE GALLERY
    # =====================================================

    gallery = ReferenceGallery()

    gallery.add(
        GalleryIdentity(
            subject_id="subject_001",
            label="Reference Subject 001",
            person_embedding_path=(
                "data/reference_gallery/"
                "test_embeddings/subject_001.npy"
            ),
            face_embedding_path=(
                "data/reference_gallery/"
                "test_face_embeddings/subject_001.npy"
            ),
            metadata={
                "source": "authorized_reference_dataset",
                "status": "active",
            },
        )
    )

    gallery.add(
        GalleryIdentity(
            subject_id="subject_002",
            label="Reference Subject 002",
            person_embedding_path=(
                "data/reference_gallery/"
                "test_embeddings/subject_002.npy"
            ),
            face_embedding_path=(
                "data/reference_gallery/"
                "test_face_embeddings/subject_002.npy"
            ),
            metadata={
                "source": "authorized_reference_dataset",
                "status": "active",
            },
        )
    )

    # =====================================================
    # 4. GALLERY LOADER
    # =====================================================

    gallery_loader = GalleryEmbeddingLoader(
        dimension=512,
    )

    # =====================================================
    # 5. PERSON FAISS INDEX
    # =====================================================

    gallery_person_index = FAISSRetriever(
        dimension=512,
    )

    gallery_indexer = ReferenceGalleryIndexer(
        gallery=gallery,
        loader=gallery_loader,
        retriever=gallery_person_index,
    )

    gallery_indexer.build_person_index()

    # =====================================================
    # 6. PERSON RETRIEVAL SERVICE
    # =====================================================

    gallery_person_retrieval_service = RetrievalService(
        retriever=gallery_person_index,
        top_k=10,
        similarity_threshold=0.70,
    )

    # =====================================================
    # 7. FACE FAISS INDEX
    # =====================================================

    gallery_face_index = FaceFAISSIndex(
        dimension=512,
    )

    gallery_indexer.build_face_index(
        face_retriever=gallery_face_index,
    )

    # =====================================================
    # 8. FACE RETRIEVAL SERVICE
    # =====================================================

    gallery_face_retrieval_service = FaceRetrievalService(
        index=gallery_face_index,
        top_k=10,
        similarity_threshold=0.70,
    )

    # =====================================================
    # 9. PERSON RE-ID PIPELINE
    # =====================================================

    person_pipeline = PersonReIDPipeline(
        tracker=person_tracker,
        track_manager=track_manager,
        crop_extractor=person_crop_extractor,
        quality_analyzer=person_quality_analyzer,
        reid_extractor=person_reid_extractor,
        memory_manager=person_memory_manager,
        aggregator=person_aggregator,
        retrieval_service=gallery_person_retrieval_service,
        candidate_history=person_candidate_history,
        candidate_ranker=person_candidate_ranker,
        auto_finalize=False,
        session=session,
        audit_writer=audit_writer,
    )

    # =====================================================
    # 10. FACE COMPONENTS
    # =====================================================

    face_detector = InsightFaceDetector(
        model_name="buffalo_l",
        det_size=(640, 640),
        confidence_threshold=0.50,
    )

    face_crop_extractor = FaceCropExtractor(
        target_size=(112, 112),
        padding_ratio=0.10,
    )

    face_quality_analyzer = FaceQualityAnalyzer(
        min_width=40,
        min_height=40,
        min_blur_score=50.0,
        minimum_quality=0.60,
    )

    face_embedder = ArcFaceEmbedder(
        model_name="buffalo_l",
        device="CPU",
    )

    face_memory_manager = FaceMemoryManager(
        max_embeddings=20,
        minimum_quality=0.60,
    )

    face_aggregator = FaceEmbeddingAggregator(
        minimum_quality=0.60,
        max_embeddings=20,
    )

    face_candidate_history = FaceCandidateHistory(
        max_history_per_candidate=20,
    )

    # =====================================================
    # 11. FACE RE-ID PIPELINE
    # =====================================================

    face_pipeline = FaceReIDPipeline(
        face_detector=face_detector,
        crop_extractor=face_crop_extractor,
        quality_analyzer=face_quality_analyzer,
        embedder=face_embedder,
        memory_manager=face_memory_manager,
        aggregator=face_aggregator,
        retrieval_service=gallery_face_retrieval_service,
        candidate_history=face_candidate_history,
    )

    # =====================================================
    # 12. EVIDENCE FUSION
    # =====================================================

    evidence_fusion = EvidenceFusion(
        person_weight=0.40,
        face_weight=0.60,
    )

    face_ranker = CandidateRanker(
        minimum_observations=3,
    )

    # =====================================================
    # 13. UNIFIED TRACEVISION PIPELINE
    # =====================================================

    tracevision_pipeline = TraceVisionPipeline(
        person_pipeline=person_pipeline,
        face_pipeline=face_pipeline,
        evidence_fusion=evidence_fusion,
        face_ranker=face_ranker,
    )

    # =====================================================
    # 14. RETURN COMPLETE PIPELINE
    # =====================================================

    return tracevision_pipeline

