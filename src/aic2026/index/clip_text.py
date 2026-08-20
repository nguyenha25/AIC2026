"""Module mã hóa văn bản truy vấn bằng OpenCLIP."""

import torch
import open_clip
import numpy as np

_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_clip_text_encoder(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
    global _model, _tokenizer
    if _model is None:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=_device,
        )
        model.eval()
        _model = model
        _tokenizer = open_clip.get_tokenizer(model_name)
    return _model, _tokenizer

def encode_query_text(query: str, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> np.ndarray:
    """Chuyển text truy vấn thành vector normalized 512-d (float32)."""
    model, tokenizer = get_clip_text_encoder(model_name, pretrained)
    tokens = tokenizer([query]).to(_device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype(np.float32).flatten()