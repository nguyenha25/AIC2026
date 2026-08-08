"""
Xây dựng bảng frame map dùng chung cho toàn bộ dự án.

Đọc các bảng map-keyframes của BTC thông qua load_frame_map()
và ghi thành một tệp Parquet tại index/frame_map.parquet.

Tệp Parquet này là dữ liệu trung gian dùng chung cho các thành phần
phía sau, giúp tra cứu video_id, n, pts_time, fps và frame_idx
mà không phải đọc lại toàn bộ các tệp CSV gốc.
"""

from src.aic2026.frame_map import load_frame_map
from src.aic2026.paths import FRAME_MAP_PARQUET


def main() -> int:
    """
    Đọc toàn bộ frame map và ghi ra index/frame_map.parquet.
    """
    frame_map = load_frame_map()

    FRAME_MAP_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_map.to_parquet(
        FRAME_MAP_PARQUET,
        index=False,
    )

    print(f"Số dòng: {len(frame_map):,}")
    print(f"Đã lưu: {FRAME_MAP_PARQUET}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())