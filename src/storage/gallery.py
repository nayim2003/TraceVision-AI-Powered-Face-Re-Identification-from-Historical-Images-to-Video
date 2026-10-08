
from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path


@dataclass
class GalleryIdentity:
    """
    Represents one authorized reference identity
    in the TraceVision gallery.
    """

    subject_id: str
    label: Optional[str] = None
    person_embedding_path: Optional[str] = None
    face_embedding_path: Optional[str] = None
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ReferenceGallery:
    """
    In-memory reference gallery with optional JSON persistence.
    """

    def __init__(self):
        self.identities: dict[str, GalleryIdentity] = {}

    def add(self, identity: GalleryIdentity) -> None:
        if not identity.subject_id:
            raise ValueError("subject_id cannot be empty.")

        self.identities[identity.subject_id] = identity

    def get(self, subject_id: str) -> Optional[GalleryIdentity]:
        return self.identities.get(subject_id)

    def remove(self, subject_id: str) -> bool:
        if subject_id not in self.identities:
            return False

        del self.identities[subject_id]
        return True

    def all(self) -> list[GalleryIdentity]:
        return list(self.identities.values())

    def __len__(self) -> int:
        return len(self.identities)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            identity.to_dict()
            for identity in self.identities.values()
        ]

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self, path: str | Path) -> None:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Gallery file not found: {path}"
            )

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.identities.clear()

        for item in data:
            identity = GalleryIdentity(**item)
            self.add(identity)

