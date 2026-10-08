
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class ReferenceGalleryBuilder:

    def __init__(
        self,
        reid_extractor,
        quality_analyzer,
        supported_extensions=None,
    ):
        self.reid_extractor = reid_extractor
        self.quality_analyzer = quality_analyzer

        if supported_extensions is None:
            supported_extensions = {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }

        self.supported_extensions = supported_extensions

    def collect_images(
        self,
        gallery_dir: str,
    ) -> List[Tuple[Path, str]]:

        gallery_path = Path(gallery_dir)

        if not gallery_path.exists():
            raise FileNotFoundError(
                f"Gallery directory not found: {gallery_path}"
            )

        image_files = []

        for subject_dir in sorted(gallery_path.iterdir()):

            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            for image_path in sorted(subject_dir.iterdir()):

                if (
                    image_path.is_file()
                    and image_path.suffix.lower()
                    in self.supported_extensions
                ):
                    image_files.append(
                        (image_path, subject_id)
                    )

        return image_files

    def build(
        self,
        gallery_dir: str,
    ):

        image_files = self.collect_images(
            gallery_dir
        )

        embeddings = []
        metadata = []

        for image_path, subject_id in image_files:

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                continue

            quality = self.quality_analyzer.analyze(
                image
            )

            if not quality.accepted:
                continue

            # Gallery images are already reference crops.
            # Resize to the Re-ID input size.
            image = cv2.resize(
                image,
                (128, 256),
                interpolation=cv2.INTER_LINEAR,
            )

            class SimpleCrop:
                def __init__(self, image):
                    self.image = image

            crop = SimpleCrop(image)

            embedding = self.reid_extractor.extract(
                crop
            )

            embeddings.append(
                embedding
            )

            metadata.append(
                {
                    "subject_id": subject_id,
                    "reference_image": image_path.name,
                    "source": "authorized_reference_gallery",
                }
            )

        if not embeddings:
            raise RuntimeError(
                "No valid reference images were found."
            )

        embeddings = np.vstack(
            embeddings
        ).astype(np.float32)

        return embeddings, metadata
