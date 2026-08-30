"""
Mã hoá câu hỏi dev bằng CLIP ViT-B/32 thành query embeddings.

Mục đích:
Đọc các câu hỏi từ tập dev và tạo vector embedding bằng
CLIP ViT-B/32 để dùng cho retrieval N-02.

Input:
D:/aic-data/dev/dev_questions.jsonl

Output:
D:/aic-data/dev/dev_query_embeddings_clip_b.npy

Mô hình:
CLIP ViT-B/32
Sử dụng ClipEncoder của project:
aic2026.index.encode.clip_encoder

Quy trình:
dev_questions.jsonl
↓
Lấy trường cau_hoi
↓
Encode bằng ClipEncoder
↓
L2 normalize
↓
Lưu thành NumPy float32

Shape mong đợi:
(76, 512)

Cách chạy:
python scripts/encode_dev_queries_clip_b.py

Lưu ý:
- Script này tạo embedding cho CLIP-B/32.
- ClipEncoder phải sử dụng đúng cấu hình CLIP-B/32
  của project.
- Vector được chuẩn hoá để tương thích với FAISS
  IndexFlatIP / cosine similarity.
- Thứ tự embedding phải khớp với thứ tự câu hỏi trong
  dev_questions.jsonl.
- File .npy nằm trong D:/aic-data/ nên không đưa lên Git
  theo .gitignore.
- Cần chạy sau khi có dev_questions.jsonl.
- Sau khi sửa script phải chạy lại và kiểm tra Shape = (76, 512).
"""

import json

import numpy as np

from aic2026.index.encode.clip_encoder import ClipEncoder


DATA_ROOT = r"D:\aic-data"

DEV_JSONL_PATH = (
    DATA_ROOT + r"\dev\dev_questions.jsonl"
)

OUTPUT_NPY_PATH = (
    DATA_ROOT + r"\dev\dev_query_embeddings_clip_b.npy"
)


# ============================================================
# LOAD DEV QUESTIONS
# ============================================================

qs = [
    json.loads(x)
    for x in open(
        DEV_JSONL_PATH,
        encoding="utf-8",
    )
    if x.strip()
]


print(
    f"[ENCODE] Đang encode {len(qs)} câu hỏi dev bằng CLIP-B/32..."
)


# ============================================================
# ENCODE
# ============================================================

enc = ClipEncoder()

X = np.stack(
    [
        enc.encode_text(q["cau_hoi"])
        for q in qs
    ]
).astype(np.float32)


# ============================================================
# CHECK
# ============================================================

if X.shape != (76, 512):
    raise ValueError(
        f"Shape không đúng: {X.shape}, "
        "mong đợi (76, 512)."
    )


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_NPY_PATH,
    X,
)


print(
    f"[ENCODE] Đã tạo thành công: {OUTPUT_NPY_PATH}"
)

print(
    f"[ENCODE] Shape: {X.shape}"
)
