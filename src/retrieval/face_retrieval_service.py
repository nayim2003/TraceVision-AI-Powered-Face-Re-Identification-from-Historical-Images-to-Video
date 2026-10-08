

from typing import List

from src.retrieval.face_faiss_index import FaceFAISSIndex
from src.retrieval.face_models import FaceRetrievalCandidate


class FaceRetrievalService:
    """
    Service layer for ArcFace vector retrieval.
    """

    def __init__(
        self,
        index: FaceFAISSIndex,
        top_k: int = 10,
        similarity_threshold: float = 0.70,
    ):
        self.index = index
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def search(
        self,
        query_embedding,
    ) -> List[FaceRetrievalCandidate]:

        results = self.index.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        candidates = []

        for result in results:

            similarity = float(
                result["similarity"]
            )

            if similarity < self.similarity_threshold:
                continue

            candidates.append(
                FaceRetrievalCandidate(
                    rank=int(result["rank"]),
                    subject_id=str(
                        result["subject_id"]
                    ),
                    similarity=similarity,
                    reference_image=result.get(
                        "reference_image"
                    ),
                    source=result.get(
                        "source"
                    ),
                )
            )

        return candidates

    def search_without_threshold(
        self,
        query_embedding,
    ) -> List[FaceRetrievalCandidate]:

        results = self.index.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        return [
            FaceRetrievalCandidate(
                rank=int(result["rank"]),
                subject_id=str(
                    result["subject_id"]
                ),
                similarity=float(
                    result["similarity"]
                ),
                reference_image=result.get(
                    "reference_image"
                ),
                source=result.get(
                    "source"
                ),
            )
            for result in results
        ]
