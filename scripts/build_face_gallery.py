
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ==================================================
# Configuration
# ==================================================

INPUT_DIR = Path(
    "data/reference_gallery/images"
)

OUTPUT_DIR = Path(
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


# ==================================================
# Helpers
# ==================================================

def normalize_embedding(embedding):
    """
    L2-normalize a face embedding.
    """

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    norm = np.linalg.norm(embedding)

    if norm <= 0:
        raise ValueError(
            "Embedding norm is zero."
        )

    return (
        embedding / norm
    ).astype(np.float32)


def select_best_face(faces):
    """
    Select the largest detected face.

    This assumes each reference image is
    intended to contain one primary subject.
    """

    if not faces:
        return None

    return max(
        faces,
        key=lambda face: (
            float(face.bbox[2] - face.bbox[0])
            * float(face.bbox[3] - face.bbox[1])
        ),
    )


# ==================================================
# Main
# ==================================================

def main():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Reference image directory not found: "
            f"{INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load the SAME InsightFace model used by TraceVision
    # --------------------------------------------------

    print(
        f"Loading InsightFace model: "
        f"{MODEL_NAME}"
    )

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

    # --------------------------------------------------
    # Process each subject
    # --------------------------------------------------

    subject_dirs = sorted(
        [
            path
            for path in INPUT_DIR.iterdir()
            if path.is_dir()
        ]
    )

    if not subject_dirs:
        raise RuntimeError(
            f"No subject directories found in "
            f"{INPUT_DIR}"
        )

    total_subjects = 0

    for subject_dir in subject_dirs:

        subject_id = subject_dir.name

        print()
        print("=" * 60)
        print(
            f"Processing: {subject_id}"
        )
        print("=" * 60)

        image_paths = sorted(
            [
                path
                for path in subject_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]
        )

        if not image_paths:
            print(
                "No supported images found."
            )
            continue

        embeddings = []

        for image_path in image_paths:

            print(
                f"  Image: {image_path.name}"
            )

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                print(
                    "    SKIPPED: unable to read image."
                )
                continue

            faces = app.get(image)

            if not faces:
                print(
                    "    SKIPPED: no face detected."
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
                    "    SKIPPED: no embedding."
                )
                continue

            try:
                embedding = normalize_embedding(
                    embedding
                )
            except ValueError:
                print(
                    "    SKIPPED: invalid embedding."
                )
                continue

            if embedding.shape[0] != 512:
                print(
                    "    SKIPPED: unexpected "
                    f"dimension {embedding.shape[0]}."
                )
                continue

            embeddings.append(
                embedding
            )

            print(
                "    OK: 512-D embedding"
            )

        # --------------------------------------------------
        # Validate subject
        # --------------------------------------------------

        if not embeddings:
            print(
                f"WARNING: No valid embeddings "
                f"for {subject_id}"
            )
            continue

        # --------------------------------------------------
        # Aggregate multiple reference images
        # --------------------------------------------------

        embedding_matrix = np.vstack(
            embeddings
        ).astype(np.float32)

        mean_embedding = np.mean(
            embedding_matrix,
            axis=0,
        )

        mean_embedding = normalize_embedding(
            mean_embedding
        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"{subject_id}.npy"
        )

        np.save(
            output_path,
            mean_embedding,
        )

        print(
            f"  Saved: {output_path}"
        )

        print(
            f"  Reference images used: "
            f"{len(embeddings)}"
        )

        print(
            f"  Embedding dimension: "
            f"{mean_embedding.shape[0]}"
        )

        print(
            f"  Embedding norm: "
            f"{np.linalg.norm(mean_embedding):.6f}"
        )

        total_subjects += 1

    print()
    print("=" * 60)
    print(
        f"Completed. Subjects processed: "
        f"{total_subjects}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()

