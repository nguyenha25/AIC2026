"""
Kiểm tra dữ liệu CLIP features trước khi xây kho FAISS.

Kiểm tra:
- Có đúng 873 tệp .npy.
- Mỗi tệp chứa ma trận vector 512 chiều.
- Tổng cộng đúng 177.321 vector.
- Không có giá trị NaN hoặc Inf.

Script chỉ đọc dữ liệu trong raw/ và không sửa dữ liệu gốc.
"""

from __future__ import annotations

import numpy as np

from aic2026.paths import (
    CLIP_FEATURES_DIR,
    clip_features_file,
    list_video_ids,
)

EXPECTED_FILES = 873
EXPECTED_VECTORS = 177_321
EXPECTED_DIM = 512


def main() -> int:
    """Kiểm tra toàn bộ CLIP features."""

    video_ids = list_video_ids(CLIP_FEATURES_DIR, ".npy")

    print("=" * 60)
    print("KIỂM TRA CLIP FEATURES")
    print("=" * 60)

    print(f"Số tệp: {len(video_ids):,}")

    if len(video_ids) != EXPECTED_FILES:
        print(
            f"[LỖI] Mong đợi {EXPECTED_FILES:,} tệp, "
            f"nhưng tìm thấy {len(video_ids):,}."
        )
        return 1

    total_vectors = 0
    errors: list[str] = []

    for video_id in video_ids:
        path = clip_features_file(video_id)
        features = np.load(path, mmap_mode="r")

        if features.ndim != 2:
            errors.append(
                f"{video_id}: shape={features.shape}, cần 2 chiều"
            )
            continue

        if features.shape[1] != EXPECTED_DIM:
            errors.append(
                f"{video_id}: shape={features.shape}, "
                f"cần chiều cuối = {EXPECTED_DIM}"
            )
            continue

        if not np.isfinite(features).all():
            errors.append(
                f"{video_id}: có giá trị NaN hoặc Inf"
            )
            continue

        total_vectors += features.shape[0]

    print(f"Tổng vector: {total_vectors:,}")

    if errors:
        print("\n[LỖI] Có dữ liệu không đúng:")
        for error in errors[:10]:
            print(f"  - {error}")

        if len(errors) > 10:
            print(f"  ... và {len(errors) - 10} lỗi khác")

        return 1

    if total_vectors != EXPECTED_VECTORS:
        print(
            f"[LỖI] Mong đợi {EXPECTED_VECTORS:,} vector, "
            f"nhưng tìm thấy {total_vectors:,}."
        )
        return 1

    print("[OK] 873 tệp đều có vector 512 chiều.")
    print(f"[OK] Tổng số vector: {total_vectors:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())