"""
Mã hoá câu hỏi dev bằng CLIP ViT-L/14 thành query embeddings.

Mục đích:
Đọc các câu hỏi từ tập dev và tạo vector embedding bằng
CLIP ViT-L/14 để dùng cho retrieval N-02.

Input:
D:/aic-data/dev/dev_questions.jsonl

Output:
D:/aic-data/dev/dev_query_embeddings_clip_l.npy

Mô hình:
CLIP ViT-L/14
pretrained = laion2b_s32b_b82k

Quy trình:
dev_questions.jsonl
↓
Lấy trường cau_hoi
↓
Tokenize bằng CLIP-L
↓
encode_text()
↓
L2 normalize
↓
Lưu thành NumPy float32

Shape mong đợi:
(76, 768)

Cách chạy:
python scripts/encode_dev_queries.py

Lưu ý:
- Script này tạo embedding cho CLIP-L/14.
- Vector được L2 normalize để tương thích với cosine similarity /
  FAISS IndexFlatIP.
- File .npy nằm trong D:/aic-data/ nên không đưa lên Git
  theo .gitignore.
- Cần chạy sau khi có dev_questions.jsonl.
"""

import json
from pathlib import Path

import numpy as np
import torch
import open_clip


DATA_ROOT = Path("D:/aic-data")

dev_jsonl_path = (
    DATA_ROOT
    / "dev"
    / "dev_questions.jsonl"
)

output_npy_path = (
    DATA_ROOT
    / "dev"
    / "dev_query_embeddings_clip_l.npy"
)


print("[ENCODE] Đang nạp mô hình CLIP ViT-L/14...")

# Nạp CLIP-L/14 với pretrained laion2b_s32b_b82k.
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="laion2b_s32b_b82k",
)

tokenizer = open_clip.get_tokenizer("ViT-L-14")

model.eval()


queries_text = []

with open(
    dev_jsonl_path,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        if line.strip():

            item = json.loads(line)

            # Lấy text câu hỏi.
            queries_text.append(
                item["cau_hoi"]
            )


print(
    f"[ENCODE] Đang encode "
    f"{len(queries_text)} câu hỏi dev..."
)

tokens = tokenizer(queries_text)


with torch.no_grad():

    query_features = model.encode_text(
        tokens
    )

    # Chuẩn hóa L2 để tương thích với
    # cosine similarity / FAISS IndexFlatIP.
    query_features /= query_features.norm(
        dim=-1,
        keepdim=True,
    )


# Chuyển sang NumPy float32.
embeddings = (
    query_features
    .cpu()
    .numpy()
    .astype(np.float32)
)


# ------------------------------------------------------------
# CHECK OUTPUT
# ------------------------------------------------------------

expected_shape = (
    len(queries_text),
    768,
)

if embeddings.shape != expected_shape:

    raise ValueError(
        "Shape embedding không đúng: "
        f"{embeddings.shape}, "
        f"mong đợi {expected_shape}."
    )


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

output_npy_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

np.save(
    output_npy_path,
    embeddings,
)


print(
    f"[ENCODE] Đã tạo thành công: "
    f"{output_npy_path}"
)

print(
    f"[ENCODE] Shape: {embeddings.shape}"
)
