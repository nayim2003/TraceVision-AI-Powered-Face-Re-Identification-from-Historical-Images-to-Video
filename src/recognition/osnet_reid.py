
from typing import Any

import numpy as np
import torch

from src.recognition.base_reid import BaseReIDExtractor


class OSNetReIDExtractor(BaseReIDExtractor):
    """
    OSNet-based person Re-ID feature extractor.
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: str = "auto",
    ):

        import torchreid

        self.model_name = model_name

        # --------------------------------
        # Device selection
        # --------------------------------

        if device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        # --------------------------------
        # Build OSNet
        # --------------------------------

        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,
            pretrained=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self._embedding_dimension = 512

    def extract(
        self,
        person_crop: Any,
    ) -> np.ndarray:

        # Import torchvision preprocessing
        from torchvision import transforms

        # --------------------------------
        # Convert crop to PIL
        # --------------------------------

        image = person_crop.image

        # OpenCV BGR → RGB
        image = image[:, :, ::-1]

        from PIL import Image

        image = Image.fromarray(image)

        # --------------------------------
        # Re-ID preprocessing
        # --------------------------------

        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        tensor = transform(image)

        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        # --------------------------------
        # Feature extraction
        # --------------------------------

        with torch.no_grad():

            features = self.model(
                tensor
            )

        # --------------------------------
        # Convert to numpy
        # --------------------------------

        embedding = features.squeeze(
            0
        ).cpu().numpy()

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        # --------------------------------
        # L2 normalization
        # --------------------------------

        norm = np.linalg.norm(
            embedding
        )

        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embedding_dimension(self) -> int:
        return self._embedding_dimension
