
from pathlib import Path
import sys

# --------------------------------------------------
# Add project root to Python import path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src.recognition.face_models import FaceDetection
from src.recognition.face_crop import FaceCropExtractor
from src.recognition.arcface_embedder import ArcFaceEmbedder


GALLERY_DIR = (
    PROJECT_ROOT
    / "data/reference_gallery/test_face_embeddings"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data/reference_gallery/images"
)

MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)


def normalize(embedding):
    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError("Zero-norm embedding.")

    return embedding / norm


def cosine_similarity(a, b):
    a = normalize(a)
    b = normalize(b)

    return float(np.dot(a, b))


def select_largest_face(faces):
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

    print("=" * 70)
    print("FACE EMBEDDING A/B DIAGNOSTIC")
    print("=" * 70)

    print()
    print(f"Project root: {PROJECT_ROOT}")

    print()
    print("Loading InsightFace...")

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=["CPUExecutionProvider"],
    )

    app.prepare(
        ctx_id=0,
        det_size=DET_SIZE,
    )

    crop_extractor = FaceCropExtractor(
        target_size=(112, 112),
        padding_ratio=0.10,
    )

    embedder = ArcFaceEmbedder(
        model_name="buffalo_l",
        device="CPU",
    )

    gallery = {}

    for subject_id in [
        "subject_001",
        "subject_002",
    ]:

        path = (
            GALLERY_DIR
            / f"{subject_id}.npy"
        )

        gallery[subject_id] = normalize(
            np.load(path)
        )

    print()
    print("Gallery loaded.")

    # --------------------------------------------------
    # Test each reference image
    # --------------------------------------------------

    for subject_id in [
        "subject_001",
        "subject_002",
    ]:

        subject_dir = (
            IMAGE_DIR
            / subject_id
        )

        image_paths = sorted(
            [
                p
                for p in subject_dir.iterdir()
                if p.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                }
            ]
        )

        print()
        print("=" * 70)
        print(f"SUBJECT: {subject_id}")
        print("=" * 70)

        for image_path in image_paths:

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                print(
                    f"{image_path.name:<20} "
                    "IMAGE LOAD FAILED"
                )
                continue

            faces = app.get(image)

            face = select_largest_face(
                faces
            )

            if face is None:
                print(
                    f"{image_path.name:<20} "
                    "NO FACE"
                )
                continue

            # --------------------------------------------------
            # PATH A
            # Full image -> InsightFace -> native embedding
            # --------------------------------------------------

            embedding_a = normalize(
                face.embedding
            )

            similarity_a = cosine_similarity(
                embedding_a,
                gallery[subject_id],
            )

            # --------------------------------------------------
            # PATH B
            # Face bbox -> crop -> 112x112 -> ArcFaceEmbedder
            # --------------------------------------------------

            detection = FaceDetection(
                bbox=(
                    float(face.bbox[0]),
                    float(face.bbox[1]),
                    float(face.bbox[2]),
                    float(face.bbox[3]),
                ),
                confidence=float(
                    face.det_score
                ),
            )

            face_crop = crop_extractor.extract(
                image,
                detection,
            )

            if face_crop is None:
                print(
                    f"{image_path.name:<20} "
                    "CROP FAILED"
                )
                continue

            embedding_b = embedder.extract(
                face_crop
            )

            similarity_b = cosine_similarity(
                embedding_b,
                gallery[subject_id],
            )

            embedding_similarity = cosine_similarity(
                embedding_a,
                embedding_b,
            )

            print(
                f"{image_path.name:<20} "
                f"A={similarity_a:.6f}  "
                f"B={similarity_b:.6f}  "
                f"A_vs_B={embedding_similarity:.6f}"
            )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
