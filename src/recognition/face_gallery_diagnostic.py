
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


INPUT_DIR = Path(
    "data/reference_gallery/images"
)

GALLERY_DIR = Path(
    "data/reference_gallery/test_face_embeddings"
)

MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def normalize(embedding):
    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    norm = np.linalg.norm(embedding)

    if norm <= 0:
        raise ValueError(
            "Zero-norm embedding."
        )

    return (
        embedding / norm
    ).astype(np.float32)


def select_best_face(faces):
    if not faces:
        return None

    return max(
        faces,
        key=lambda face: (
            float(face.bbox[2] - face.bbox[0])
            * float(face.bbox[3] - face.bbox[1])
        ),
    )


def main():

    print("Loading InsightFace...")

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    app.prepare(
        ctx_id=0,
        det_size=DET_SIZE,
    )

    subject_dirs = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_dir()
    )

    if not subject_dirs:
        raise RuntimeError(
            f"No subject directories found in {INPUT_DIR}"
        )

    for subject_dir in subject_dirs:

        subject_id = subject_dir.name

        gallery_path = (
            GALLERY_DIR
            / f"{subject_id}.npy"
        )

        if not gallery_path.exists():
            print()
            print(
                f"{subject_id}: gallery embedding not found."
            )
            continue

        gallery_embedding = normalize(
            np.load(gallery_path)
        )

        print()
        print("=" * 70)
        print(f"SUBJECT: {subject_id}")
        print("=" * 70)

        similarities = []

        image_paths = sorted(
            path
            for path in subject_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

        for image_path in image_paths:

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                print(
                    f"{image_path.name}: READ ERROR"
                )
                continue

            faces = app.get(image)

            if not faces:
                print(
                    f"{image_path.name}: NO FACE"
                )
                continue

            face = select_best_face(faces)

            embedding = getattr(
                face,
                "embedding",
                None,
            )

            if embedding is None:
                print(
                    f"{image_path.name}: NO EMBEDDING"
                )
                continue

            embedding = normalize(
                embedding
            )

            similarity = float(
                np.dot(
                    embedding,
                    gallery_embedding,
                )
            )

            similarities.append(
                similarity
            )

            print(
                f"{image_path.name:20s} "
                f"similarity={similarity:.6f}"
            )

        if similarities:

            print("-" * 70)

            print(
                f"min  = {min(similarities):.6f}"
            )

            print(
                f"mean = {np.mean(similarities):.6f}"
            )

            print(
                f"max  = {max(similarities):.6f}"
            )

        else:

            print(
                "No valid face embeddings found."
            )


if __name__ == "__main__":
    main()

