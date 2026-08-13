"""
CLIP encoder dùng chung cho Task 3.

Model phải khớp với vector CLIP BTC cung cấp:
ViT-B-32 + pretrained="openai".

Mọi vector đầu ra đều được L2-normalize để
inner product tương đương cosine similarity.
"""

import numpy as np
import open_clip
import torch
from PIL import Image


class ClipEncoder:
    """Mã hóa ảnh và văn bản bằng CLIP ViT-B/32."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()

    @property
    def dimension(self) -> int:
        """Số chiều vector CLIP."""
        return 512

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Mã hóa một ảnh PIL thành vector float32 chuẩn hóa."""

        image_input = self.preprocess(image).unsqueeze(0)

        with torch.no_grad():
            features = self.model.encode_image(image_input)

        features = features / features.norm(
            dim=-1,
            keepdim=True,
        )

        return (
            features[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def encode_image_path(self, image_path) -> np.ndarray:
        """Mã hóa ảnh từ đường dẫn."""

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.encode_image(image)

    def encode_text(self, text: str) -> np.ndarray:
        """Mã hóa một câu truy vấn thành vector float32 chuẩn hóa."""

        tokens = self.tokenizer([text])

        with torch.no_grad():
            features = self.model.encode_text(tokens)

        features = features / features.norm(
            dim=-1,
            keepdim=True,
        )

        return (
            features[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )