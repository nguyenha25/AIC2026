"""
Benchmark tốc độ tra cứu FAISS.

Dùng 10 vector CLIP có sẵn làm query để đo riêng tốc độ
của FAISS, chưa tính thời gian encode câu chữ.

Yêu cầu:
- index có 177.321 vector
- vector 512 chiều
- trung bình 10 truy vấn < 200 ms
"""

from __future__ import annotations

import time

import faiss
import numpy as np

from aic2026.paths import CLIP_FEATURES_DIR, FAISS_DIR, clip_features_file, list_video_ids

EXPECTED_VECTORS = 177_321
CLIP_DIMENSION = 512
NUM_QUERIES = 10
TOP_K = 100


def main() -> int:
    index_path = FAISS_DIR / "clip_b32.index"

    if not index_path.exists():
        print(f"[LỖI] Không tìm thấy FAISS index: {index_path}")
        return 1

    print("=" * 60)
    print("BENCHMARK FAISS")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load index
    # ---------------------------------------------------------
    index = faiss.read_index(str(index_path))

    print(f"Số vector : {index.ntotal:,}")
    print(f"Chiều     : {index.d}")

    if index.ntotal != EXPECTED_VECTORS:
        print(
            f"[LỖI] Index có {index.ntotal:,} vector, "
            f"cần {EXPECTED_VECTORS:,}."
        )
        return 1

    if index.d != CLIP_DIMENSION:
        print(
            f"[LỖI] Index có {index.d} chiều, "
            f"cần {CLIP_DIMENSION}."
        )
        return 1

    # ---------------------------------------------------------
    # 2. Lấy 10 vector query
    # ---------------------------------------------------------
    video_ids = list_video_ids(CLIP_FEATURES_DIR, ".npy")

    if not video_ids:
        print("[LỖI] Không tìm thấy CLIP features.")
        return 1

    queries = []

    # Lấy vector đầu tiên của 10 video đầu tiên.
    for video_id in video_ids[:NUM_QUERIES]:
        features = np.load(clip_features_file(video_id), mmap_mode="r")

        if features.ndim != 2 or features.shape[1] != CLIP_DIMENSION:
            print(f"[LỖI] {video_id}: shape={features.shape}")
            return 1

        query = np.asarray(features[0], dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        queries.append(query)

    queries = np.vstack(queries)

    # ---------------------------------------------------------
    # 3. Warm-up
    # ---------------------------------------------------------
    index.search(queries[:1], TOP_K)

    # ---------------------------------------------------------
    # 4. Benchmark 10 truy vấn
    # ---------------------------------------------------------
    times_ms = []

    for i, query in enumerate(queries, start=1):
        query = np.asarray(query, dtype=np.float32).reshape(1, CLIP_DIMENSION)
        
        started_at = time.perf_counter()

        distances, indices = index.search(query, TOP_K)

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        times_ms.append(elapsed_ms)

        print(
            f"Query {i:02d}: "
            f"{elapsed_ms:.3f} ms — "
            f"{len(indices[0])} kết quả"
        )

    average_ms = sum(times_ms) / len(times_ms)

    print()
    print("=" * 60)
    print("KẾT QUẢ")
    print("=" * 60)
    print(f"Số query       : {NUM_QUERIES}")
    print(f"Top-K          : {TOP_K}")
    print(f"Trung bình     : {average_ms:.3f} ms")
    print(f"Yêu cầu        : < 200 ms")

    if average_ms >= 200:
        print("[CHƯA ĐẠT] Tốc độ trung bình >= 200 ms.")
        return 1

    print("[ĐẠT] Tốc độ FAISS trung bình < 200 ms.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())