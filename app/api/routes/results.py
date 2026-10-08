
# app/api/routes/results.py
import numpy as np
from fastapi import APIRouter

from app.api.schemas import (
    TrackResultResponse,
    TrackSummary,
)


router = APIRouter(
    prefix="/results",
    tags=["Results"],
)


# ==================================================
# Runtime TraceVision Pipeline
# ==================================================

tracevision_pipeline = None


def set_pipeline(pipeline):
    """
    Attach the active TraceVision pipeline
    to the results router.
    """

    global tracevision_pipeline

    tracevision_pipeline = pipeline


# ==================================================
# All Finalized Track Results
# ==================================================

@router.get("/tracks")
def get_tracks():
    if tracevision_pipeline is None:
        return []

    results = tracevision_pipeline.get_all_final_results()

    return [
        TrackSummary(
            track_id=result.track_id,
            state=result.state,
            observation_count=result.observation_count,
            person_embedding_count=result.person_embedding_count,
            face_embedding_count=result.face_embedding_count,
            has_candidates=result.has_candidates,
            decision=result.decision,
            subject_id=result.subject_id,
            fusion_score=result.fusion_score,
        )
        for result in results.values()
    ]


# ==================================================
# Single Track Result
# ==================================================

@router.get(
    "/tracks/{track_id}",
    response_model=TrackResultResponse,
)
def get_track_result(track_id: int):
    """
    Return the final result for a specific track.
    """

    if tracevision_pipeline is None:
        return {
            "track_id": track_id,
            "state": "unavailable",
            "observation_count": 0,
            "person_embedding_count": 0,
            "face_embedding_count": 0,
            "has_candidates": False,
            "best_candidate": None,
        }

    result = (
        tracevision_pipeline
        .get_final_result(track_id)
    )

    if result is None:
        return {
            "track_id": track_id,
            "state": "not_found",
            "observation_count": 0,
            "person_embedding_count": 0,
            "face_embedding_count": 0,
            "has_candidates": False,
            "best_candidate": None,
        }

    best_candidate = result.best_candidate

    if hasattr(best_candidate, "to_dict"):
        best_candidate = (
            best_candidate.to_dict()
        )

    elif hasattr(
        best_candidate,
        "__dict__",
    ):
        best_candidate = (
            best_candidate.__dict__
        )

    return TrackResultResponse(
        track_id=result.track_id,
        state=result.state,
        observation_count=result.observation_count,
        person_embedding_count=result.person_embedding_count,
        face_embedding_count=result.face_embedding_count,
        has_candidates=result.has_candidates,

        # Final decision
        decision=result.decision,
        subject_id=result.subject_id,
        fusion_score=result.fusion_score,

        best_candidate=best_candidate,
    )


# ==================================================
# Temporary Face History Debug
# ==================================================

@router.get(
    "/tracks/{track_id}/face-debug"
)
def get_face_debug(track_id: int):
    """
    Temporary debugging endpoint.

    Returns face candidate history and
    ranked face candidates for a track.
    """

    if tracevision_pipeline is None:
        return {
            "error": (
                "TraceVision pipeline "
                "is not initialized."
            )
        }

    face_history = (
        tracevision_pipeline
        .face_pipeline
        .get_track_history(track_id)
    )

    face_candidates = (
        tracevision_pipeline
        .get_face_candidates(track_id)
    )

    return {
        "track_id": track_id,
        "face_history_count": len(
            face_history
        ),
        "face_history": [
            (
                item.to_dict()
                if hasattr(item, "to_dict")
                else item.__dict__
            )
            for item in face_history
        ],
        "ranked_face_candidates": [
            (
                item.to_dict()
                if hasattr(item, "to_dict")
                else item.__dict__
            )
            for item in face_candidates
        ],
    }


# ==================================================
# Temporary Raw Face Retrieval Debug
# ==================================================

@router.get(
    "/tracks/{track_id}/face-retrieval-debug"
)
def get_face_retrieval_debug(track_id: int):
    """
    Temporary debugging endpoint.

    Runs face retrieval WITHOUT the configured
    similarity threshold.

    This allows us to inspect the actual ArcFace
    similarity scores returned by the FAISS index.
    """

    if tracevision_pipeline is None:
        return {
            "error": (
                "TraceVision pipeline "
                "is not initialized."
            )
        }

    # ----------------------------------------------
    # Access Person Track
    # ----------------------------------------------

    person_pipeline = (
        tracevision_pipeline
        .person_pipeline
    )

    face_pipeline = (
        tracevision_pipeline
        .face_pipeline
    )

    track = (
        person_pipeline
        .track_manager
        .get_track(track_id)
    )

    if track is None:
        return {
            "track_id": track_id,
            "error": "Track not found.",
        }

    # ----------------------------------------------
    # Face Memory
    # ----------------------------------------------

    face_memory = track.face_memory

    if face_memory.size == 0:
        return {
            "track_id": track_id,
            "face_memory_size": 0,
            "message": (
                "No face embeddings stored."
            ),
        }

    # ----------------------------------------------
    # Aggregate Face Embeddings
    # ----------------------------------------------

    track_embedding = (
        face_pipeline
        .aggregator
        .aggregate(face_memory)
    )

    if track_embedding is None:
        return {
            "track_id": track_id,
            "face_memory_size": (
                face_memory.size
            ),
            "error": (
                "Face embedding aggregation "
                "returned None."
            ),
        }

    # ----------------------------------------------
    # Raw Retrieval
    # ----------------------------------------------

    raw_candidates = (
        face_pipeline
        .retrieval_service
        .search_without_threshold(
            track_embedding
        )
    )

    # ----------------------------------------------
    # Serialize Candidates
    # ----------------------------------------------

    serialized_candidates = []

    for candidate in raw_candidates:

        if hasattr(
            candidate,
            "to_dict",
        ):
            serialized_candidates.append(
                candidate.to_dict()
            )

        elif hasattr(
            candidate,
            "__dict__",
        ):
            serialized_candidates.append(
                candidate.__dict__
            )

        else:
            serialized_candidates.append(
                candidate
            )

    # ----------------------------------------------
    # Return Debug Information
    # ----------------------------------------------

    return {
        "track_id": track_id,
        "face_memory_size": (
            face_memory.size
        ),
        "raw_candidate_count": (
            len(raw_candidates)
        ),
        "raw_candidates": (
            serialized_candidates
        ),
        "configured_similarity_threshold": (
            face_pipeline
            .retrieval_service
            .similarity_threshold
        ),
    }

@router.get(
    "/tracks/{track_id}/face-embedding-debug"
)
def get_face_embedding_debug(track_id: int):
    if tracevision_pipeline is None:
        return {
            "error": (
                "TraceVision pipeline "
                "is not initialized."
            )
        }

    person_pipeline = (
        tracevision_pipeline
        .person_pipeline
    )

    face_pipeline = (
        tracevision_pipeline
        .face_pipeline
    )

    track = (
        person_pipeline
        .track_manager
        .get_track(track_id)
    )

    if track is None:
        return {
            "track_id": track_id,
            "error": "Track not found.",
        }

    face_memory = track.face_memory

    if face_memory.size == 0:
        return {
            "track_id": track_id,
            "face_memory_size": 0,
            "error": "No face embeddings stored.",
        }

    track_embedding = (
        face_pipeline
        .aggregator
        .aggregate(face_memory)
    )

    if track_embedding is None:
        return {
            "track_id": track_id,
            "face_memory_size": face_memory.size,
            "error": (
                "Face embedding aggregation "
                "returned None."
            ),
        }

    track_embedding = np.asarray(
        track_embedding,
        dtype=np.float32,
    ).reshape(-1)

    norm = np.linalg.norm(track_embedding)

    if norm <= 0:
        return {
            "track_id": track_id,
            "face_memory_size": face_memory.size,
            "error": (
                "Aggregated embedding has "
                "zero norm."
            ),
        }

    track_embedding = track_embedding / norm

    face_index = (
        face_pipeline
        .retrieval_service
        .index
    )

    comparisons = []

    for index, metadata in enumerate(
        face_index.metadata
    ):
        gallery_embedding = (
            face_index.embeddings[index]
        )

        gallery_embedding = np.asarray(
            gallery_embedding,
            dtype=np.float32,
        ).reshape(-1)

        gallery_norm = np.linalg.norm(
            gallery_embedding
        )

        if gallery_norm <= 0:
            continue

        gallery_embedding = (
            gallery_embedding / gallery_norm
        )

        similarity = float(
            np.dot(
                track_embedding,
                gallery_embedding,
            )
        )

        comparisons.append(
            {
                "subject_id": str(
                    metadata.get(
                        "subject_id",
                        "unknown",
                    )
                ),
                "similarity": similarity,
            }
        )

    comparisons.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return {
        "track_id": track_id,
        "face_memory_size": face_memory.size,
        "embedding_dimension": int(
            track_embedding.shape[0]
        ),
        "embedding_norm": float(
            np.linalg.norm(track_embedding)
        ),
        "gallery_comparisons": comparisons,
    }

@router.get("/tracks/{track_id}/face-individual-debug")
def get_face_individual_debug(track_id: int):
    import numpy as np
    from pathlib import Path

    if tracevision_pipeline is None:
        return {
            "error": "TraceVision pipeline is not initialized."
        }

    person_pipeline = tracevision_pipeline.person_pipeline

    track = person_pipeline.track_manager.get_track(track_id)

    if track is None:
        return {
            "track_id": track_id,
            "error": "Track not found.",
        }

    face_memory = track.face_memory

    if face_memory.size == 0:
        return {
            "track_id": track_id,
            "face_memory_size": 0,
            "error": "No face embeddings stored.",
        }

    gallery_paths = {
        "subject_001": Path(
            "data/reference_gallery/test_face_embeddings/subject_001.npy"
        ),
        "subject_002": Path(
            "data/reference_gallery/test_face_embeddings/subject_002.npy"
        ),
    }

    gallery_embeddings = {}

    for subject_id, path in gallery_paths.items():
        if not path.exists():
            continue

        embedding = np.load(path).astype(np.float32).reshape(-1)

        norm = np.linalg.norm(embedding)

        if norm > 0:
            embedding = embedding / norm

        gallery_embeddings[subject_id] = embedding

    comparisons = []

    for index, observation in enumerate(face_memory.embeddings):

        query = np.asarray(
            observation.embedding,
            dtype=np.float32,
        ).reshape(-1)

        query_norm = np.linalg.norm(query)

        if query_norm <= 0:
            comparisons.append({
                "observation_index": index,
                "frame_number": observation.frame_number,
                "quality": observation.quality,
                "error": "Zero-norm embedding.",
            })
            continue

        query = query / query_norm

        result = {
            "observation_index": index,
            "frame_number": observation.frame_number,
            "quality": float(observation.quality),
            "embedding_dimension": int(query.shape[0]),
            "embedding_norm": float(np.linalg.norm(query)),
            "similarities": {},
        }

        for subject_id, gallery_embedding in gallery_embeddings.items():

            if query.shape != gallery_embedding.shape:
                result["similarities"][subject_id] = None
                continue

            similarity = float(
                np.dot(
                    query,
                    gallery_embedding,
                )
            )

            result["similarities"][subject_id] = similarity

        comparisons.append(result)

    return {
        "track_id": track_id,
        "face_memory_size": face_memory.size,
        "gallery_subjects": list(gallery_embeddings.keys()),
        "observations": comparisons,
    }

