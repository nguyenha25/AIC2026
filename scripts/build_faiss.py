"""
Dựng kho tra cứu theo hình: index/faiss/clip_b32.index

Nạp 873 tệp trong raw/clip-features-32/ theo đúng thứ tự của
frame_map.parquet, ghi kèm bảng thứ tự clip_b32_ids.parquet.

Đặt tại scripts/build_faiss.py

Chạy:
    python scripts/build_faiss.py
    python scripts/build_faiss.py --no-strict   # dựng trên dữ liệu thiếu, chỉ để thử

Cần chạy scripts/build_frame_map.py trước.

Chạy xong phải thấy ĐẠT. Thấy CHƯA ĐẠT thì đừng báo xong Việc 2.
"""

from __future__ import annotations

import argparse
import sys
import time

# Import theo đúng kiểu cả nhóm dùng: 'aic2026.…', KHÔNG phải 'src.aic2026.…'.
from aic2026.index.faiss_index import (
    CLIP_DIMENSION,
    INDEX_IDS_PATH,
    INDEX_PATH,
    benchmark_search,
    build_index,
    write_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nạp đặc trưng CLIP vào FAISS theo thứ tự của frame_map."
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Vẫn ghi kho dù chưa đủ 177.321 vector. Chỉ dùng để thử.",
    )
    args = parser.parse_args()

    started_at = time.time()

    try:
        index, index_ids = build_index(strict=not args.no_strict)
    except (FileNotFoundError, ValueError) as error:
        print(f"\nDỪNG: {error}", file=sys.stderr)
        return 1

    elapsed_ms = benchmark_search(index)

    print()
    print(f"Số video     : {index_ids['video_id'].nunique():,}")
    print(f"Số vector    : {index.ntotal:,}")
    print(f"Chiều vector : {CLIP_DIMENSION}")
    print(f"Tra thử 10 câu: {elapsed_ms:.1f} ms/câu (cần dưới 200 ms)")

    if elapsed_ms >= 200:
        print(
            f"\nCHẬM: {elapsed_ms:.1f} ms vượt mốc 200 ms của Việc 2.",
            file=sys.stderr,
        )
        return 1

    print("\nĐẠT — kho đủ vector và tra đủ nhanh.")

    write_index(index, index_ids)

    print(f"Đã lưu: {INDEX_PATH}")
    print(f"Đã lưu: {INDEX_IDS_PATH}")
    print(f"Xong sau {time.time() - started_at:.1f} giây")
    print(
        "\nLƯU Ý: con số trên chỉ đo phần FAISS. Mốc 200 ms thật phải đo lại "
        "sau khi Việc 3 xong, vì mã hoá câu chữ cũng tốn thời gian."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
