"""
Kho FAISS cho đặc trưng CLIP của toàn bộ keyframe.

Đọc các vector CLIP do BTC cung cấp trong raw/clip-features-32/,
chuyển float16 sang float32, chuẩn hóa độ dài vector và xây dựng
kho tra cứu nhanh.

Mỗi vector vẫn được đối chiếu với video_id và n thông qua
frame_map.parquet để các kết quả tìm kiếm có thể trả về đúng
ảnh tương ứng.

File index được ghi vào index/faiss/clip_b32.index.
"""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.aic2026.paths import (
    CLIP_FEATURES_DIR,
    FAISS_DIR,
    FRAME_MAP_PARQUET,
    clip_features_file,
    list_video_ids,
)
from src.aic2026.types import Hit


CLIP_DIMENSION = 512


def build_index() -> int:
    """
    Đọc toàn bộ đặc trưng CLIP và xây dựng FAISS index.

    Vector được đổi sang float32 và chuẩn hóa trước khi đưa vào
    IndexFlatIP. Với vector đã chuẩn hóa, inner product tương đương
    cosine similarity.
    """
    video_ids = list_video_ids(CLIP_FEATURES_DIR, ".npy")

    if not video_ids:
        raise FileNotFoundError(
            "Không tìm thấy tệp CLIP trong raw/clip-features-32/."
        )

    frame_map = pd.read_parquet(FRAME_MAP_PARQUET)

    vectors = []
    total_vectors = 0

    for video_id in video_ids:
        feature_path = clip_features_file(video_id)
        features = np.load(feature_path)

        if features.ndim != 2 or features.shape[1] != CLIP_DIMENSION:
            raise ValueError(
                f"{feature_path} có shape {features.shape}, "
                f"mong đợi (n, {CLIP_DIMENSION})."
            )

        features = features.astype(np.float32)

        faiss.normalize_L2(features)

        vectors.append(features)
        total_vectors += len(features)

        print(
            f"[OK] {video_id}: {len(features):,} vector"
        )

    all_vectors = np.vstack(vectors)

    if len(all_vectors) != len(frame_map):
        raise ValueError(
            f"Số vector ({len(all_vectors):,}) không khớp "
            f"số dòng frame_map ({len(frame_map):,})."
        )

    index = faiss.IndexFlatIP(CLIP_DIMENSION)
    index.add(all_vectors)

    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    index_path = FAISS_DIR / "clip_b32.index"
    faiss.write_index(index, str(index_path))

    print()
    print(f"Số video: {len(video_ids):,}")
    print(f"Số vector: {total_vectors:,}")
    print(f"Chiều vector: {CLIP_DIMENSION}")
    print(f"Đã lưu: {index_path}")

    return 0


def search_by_text(query: str, top_k: int = 100) -> list[Hit]:
    """
    Tìm kiếm bằng câu chữ.

    Hàm này cần ClipEncoder để biến câu chữ thành vector CLIP.
    Phần encoder sẽ được nối vào sau theo hợp đồng chung của nhóm.
    """
    raise NotImplementedError(
        "search_by_text() sẽ sử dụng ClipEncoder ở bước tích hợp."
    )