
# tests/test_retrieval.py

import numpy as np

from src.retrieval.faiss_index import FAISSRetriever


def test_faiss_retriever_empty_search():
    retriever = FAISSRetriever(dimension=512)

    query = np.random.randn(512).astype(np.float32)

    results = retriever.search(
        query_embedding=query,
        top_k=5,
    )

    assert results == []


def test_faiss_retriever_add_and_search():
    retriever = FAISSRetriever(dimension=512)

    embedding = np.random.randn(512).astype(np.float32)

    retriever.add(
        embeddings=embedding,
        metadata=[
            {
                "subject_id": "subject_001",
                "modality": "person",
            }
        ],
    )

    results = retriever.search(
        query_embedding=embedding,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0][0]["subject_id"] == "subject_001"
    assert results[0][1] > 0.99


# tests/test_gallery.py

from src.storage.gallery import (
    GalleryIdentity,
    ReferenceGallery,
)


def test_gallery_add_and_get():

    gallery = ReferenceGallery()

    identity = GalleryIdentity(
        subject_id="subject_001",
        label="Test Subject",
    )

    gallery.add(identity)

    result = gallery.get("subject_001")

    assert result is not None
    assert result.subject_id == "subject_001"


def test_gallery_remove():

    gallery = ReferenceGallery()

    gallery.add(
        GalleryIdentity(
            subject_id="subject_001"
        )
    )

    assert gallery.remove("subject_001") is True
    assert gallery.get("subject_001") is None



# tests/test_final_result.py

from src.aggregation.track_result import FinalTrackResult


def test_final_track_result_duration():

    result = FinalTrackResult(
        track_id=1,
        first_frame=10,
        last_frame=30,
        observation_count=21,
        person_embedding_count=20,
        face_embedding_count=18,
        state="terminated",
    )

    assert result.duration_frames == 21


def test_final_track_result_serialization():

    result = FinalTrackResult(
        track_id=1,
        first_frame=1,
        last_frame=10,
        observation_count=10,
        person_embedding_count=10,
        face_embedding_count=10,
        state="terminated",
    )

    data = result.to_dict()

    assert data["track_id"] == 1
    assert data["observation_count"] == 10
    assert data["face_embedding_count"] == 10
    assert data["state"] == "terminated"


