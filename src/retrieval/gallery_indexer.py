
from typing import Optional

from src.storage.gallery import ReferenceGallery
from src.storage.gallery_loader import GalleryEmbeddingLoader
from src.retrieval.faiss_index import FAISSRetriever


class ReferenceGalleryIndexer:
    """
    Builds a FAISS index from an authorized reference gallery.
    """

    def __init__(
        self,
        gallery: ReferenceGallery,
        loader: GalleryEmbeddingLoader,
        retriever: FAISSRetriever,
    ):
        self.gallery = gallery
        self.loader = loader
        self.retriever = retriever

    def build_person_index(self) -> int:
        """
        Build the person Re-ID FAISS index.

        Returns:
            Number of indexed reference embeddings.
        """

        embeddings, metadata = (
            self.loader.load_person_embeddings(
                self.gallery.all()
            )
        )

        # Start from a clean index.
        self.retriever.reset()

        if embeddings.shape[0] == 0:
            return 0

        self.retriever.add(
            embeddings=embeddings,
            metadata=metadata,
        )

        return embeddings.shape[0]

    def build_face_index(
        self,
        face_retriever: Optional[FAISSRetriever] = None,
    ) -> int:
        """
        Build the face Re-ID FAISS index.

        Returns:
            Number of indexed reference embeddings.
        """

        if face_retriever is None:
            raise ValueError(
                "face_retriever is required."
            )

        embeddings, metadata = (
            self.loader.load_face_embeddings(
                self.gallery.all()
            )
        )

        face_retriever.reset()

        if embeddings.shape[0] == 0:
            return 0

        face_retriever.add(
            embeddings=embeddings,
            metadata=metadata,
        )

        return embeddings.shape[0]
