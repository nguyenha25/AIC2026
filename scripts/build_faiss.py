"""
Xây dựng kho FAISS cho toàn bộ CLIP features.

Đọc 873 tệp .npy từ raw/clip-features-32/,
chuyển float16 -> float32, chuẩn hóa vector,
sau đó xây IndexFlatIP để tìm kiếm cosine similarity.

Output:
    index/faiss/clip_b32.index
"""

from __future__ import annotations

import sys
import time

import faiss
import numpy as np
import pandas as pd

from aic2026.paths import (
    CLIP_FEATURES_DIR,
    FAISS_DIR,
    FRAME_MAP_PARQUET,
    clip_features_file,
    list_video_ids,
)

EXPECTED_VIDEOS = 873
EXPECTED_VECTORS = 177_321
CLIP_DIMENSION = 512


def main() -> int:
    started_at = time.time()

    print("=" * 60)
    print("XÂY DỰNG FAISS INDEX")
    print("=" * 60)

    video_ids = list_video_ids(CLIP_FEATURES_DIR, ".npy")

    print(f"Số tệp CLIP: {len(video_ids):,}")

    if len(video_ids) != EXPECTED_VIDEOS:
        print(
            f"[LỖI] Mong đợi {EXPECTED_VIDEOS:,} tệp, "
            f"nhưng tìm thấy {len(video_ids):,}.",
            file=sys.stderr,
        )
        return 1

    if not FRAME_MAP_PARQUET.exists():
        print(
            f"[LỖI] Không tìm thấy frame map: {FRAME_MAP_PARQUET}",
            file=sys.stderr,
        )
        return 1

    frame_map = pd.read_parquet(FRAME_MAP_PARQUET)

    if len(frame_map) != EXPECTED_VECTORS:
        print(
            f"[LỖI] frame_map có {len(frame_map):,} dòng, "
            f"cần {EXPECTED_VECTORS:,}.",
            file=sys.stderr,
        )
        return 1

    vectors = []
    total_vectors = 0

    for i, video_id in enumerate(video_ids, start=1):
        feature_path = clip_features_file(video_id)
        features = np.load(feature_path)

        if features.ndim != 2 or features.shape[1] != CLIP_DIMENSION:
            print(
                f"[LỖI] {video_id}: shape={features.shape}, "
                f"cần (?, {CLIP_DIMENSION})",
                file=sys.stderr,
            )
            return 1

        # FAISS cần float32.
        features = features.astype(np.float32, copy=False)

        # Chuẩn hóa để inner product = cosine similarity.
        faiss.normalize_L2(features)

        vectors.append(features)
        total_vectors += len(features)

        if i % 50 == 0 or i == len(video_ids):
            print(
                f"  {i}/{len(video_ids)} video — "
                f"{total_vectors:,} vector"
            )

    all_vectors = np.vstack(vectors)

    if all_vectors.shape != (EXPECTED_VECTORS, CLIP_DIMENSION):
        print(
            f"[LỖI] Shape cuối: {all_vectors.shape}, "
            f"cần ({EXPECTED_VECTORS}, {CLIP_DIMENSION}).",
            file=sys.stderr,
        )
        return 1

    print()
    print("Đang xây IndexFlatIP...")

    index = faiss.IndexFlatIP(CLIP_DIMENSION)
    index.add(all_vectors)

    if index.ntotal != EXPECTED_VECTORS:
        print(
            f"[LỖI] FAISS có {index.ntotal:,} vector, "
            f"cần {EXPECTED_VECTORS:,}.",
            file=sys.stderr,
        )
        return 1

    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    index_path = FAISS_DIR / "clip_b32.index"
    faiss.write_index(index, str(index_path))

    elapsed = time.time() - started_at

    print()
    print("=" * 60)
    print("ĐẠT — FAISS INDEX ĐÃ XÂY XONG")
    print("=" * 60)
    print(f"Số video       : {len(video_ids):,}")
    print(f"Số vector      : {index.ntotal:,}")
    print(f"Chiều vector   : {index.d}")
    print(f"Index type     : IndexFlatIP")
    print(f"Đã lưu         : {index_path}")
    print(f"Thời gian      : {elapsed:.1f} giây")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())