
from typing import List

from src.retrieval.models import RetrievalCandidate


class RetrievalService:

    def __init__(
        self,
        retriever,
        top_k: int = 5,
        similarity_threshold: float = 0.70,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def search(
        self,
        query_embedding,
    ) -> List[RetrievalCandidate]:

        results = self.retriever.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        candidates = []

        for rank, (metadata, score) in enumerate(
            results,
            start=1,
        ):

            candidates.append(
                RetrievalCandidate(
                    rank=rank,
                    subject_id=metadata["subject_id"],
                    similarity=score,
                    reference_image=metadata.get(
                        "reference_image"
                    ),
                    source=metadata.get(
                        "source"
                    ),
                )
            )

        return candidates

    def filter_by_threshold(
        self,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:

        return [
            candidate
            for candidate in candidates
            if candidate.similarity
            >= self.similarity_threshold
        ]
