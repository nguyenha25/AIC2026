"""
Xây dựng bảng frame map dùng chung cho toàn bộ dự án.

Đọc các bảng map-keyframes của BTC, kiểm tra dữ liệu,
sau đó ghi thành:

    index/frame_map.parquet

Chạy:

    python scripts/build_frame_map.py

Hoặc khi chỉ muốn thử trên dữ liệu chưa đủ:

    python scripts/build_frame_map.py --no-strict

Lưu ý:
    --no-strict chỉ dùng để thử.
    Không dùng Parquet tạo bằng --no-strict cho pipeline chính.
"""

from __future__ import annotations

import argparse
import sys
import time

from aic2026.frame_map import (
    EXPECTED_ROWS,
    EXPECTED_VIDEOS,
    check_frame_map,
    scan_map_keyframes,
    write_frame_map,
)
from aic2026.paths import FRAME_MAP_PARQUET


def main() -> int:
    """Đọc, kiểm tra và ghi frame map."""

    parser = argparse.ArgumentParser(
        description=(
            "Gộp các bảng map-keyframes thành "
            "index/frame_map.parquet."
        )
    )

    parser.add_argument(
        "--no-strict",
        action="store_true",
        help=(
            "Vẫn ghi Parquet dù dữ liệu chưa đạt "
            "đủ điều kiện. Chỉ dùng để thử."
        ),
    )

    args = parser.parse_args()

    started_at = time.time()

    try:
        (
            frame_map,
            worst_drift,
            worst_drift_at,
        ) = scan_map_keyframes()

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(
            f"\nDỪNG: {error}",
            file=sys.stderr,
        )
        return 1

    video_count = frame_map[
        "video_id"
    ].nunique()

    row_count = len(frame_map)

    empty_cells = int(
        frame_map.isna().sum().sum()
    )

    print()
    print(
        f"Số video       : "
        f"{video_count:,} "
        f"(cần {EXPECTED_VIDEOS:,})"
    )

    print(
        f"Số dòng        : "
        f"{row_count:,} "
        f"(cần {EXPECTED_ROWS:,})"
    )

    print(
        f"Ô trống        : "
        f"{empty_cells:,}"
    )

    print(
        f"Lệch lớn nhất  : "
        f"{worst_drift}"
        + (
            f" tại {worst_drift_at}"
            if worst_drift_at
            else ""
        )
    )

    problems = check_frame_map(
        frame_map,
        worst_drift,
        worst_drift_at,
    )

    if problems:
        message = (
            "\nCHƯA ĐẠT: "
            + "; ".join(problems)
        )

        if not args.no_strict:
            print(
                message,
                file=sys.stderr,
            )

            print(
                "Không ghi tệp. "
                "Kiểm tra lại dữ liệu "
                "raw/map-keyframes/.",
                file=sys.stderr,
            )

            return 1

        print(
            message
            + "\nTiếp tục ghi vì đang dùng --no-strict."
        )

    else:
        print(
            "\nĐẠT — bảng đối chiếu "
            "đủ và hợp lệ."
        )

    write_frame_map(frame_map)

    elapsed = time.time() - started_at

    print(
        f"Đã lưu: {FRAME_MAP_PARQUET}"
    )

    print(
        f"Xong sau {elapsed:.1f} giây"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())