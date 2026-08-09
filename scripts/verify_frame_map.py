"""
Kiểm tra frame_map.parquet sau khi xây dựng.

Kiểm tra ba yêu cầu của Task 1:

1. Có đúng 177.321 dòng.
2. Không có ô nào trống.
3. Lấy ngẫu nhiên 20 dòng và đối chiếu lại với CSV gốc
   trong raw/map-keyframes/.

Script chỉ đọc dữ liệu và không sửa các tệp dữ liệu gốc.
"""

import pandas as pd

from src.aic2026.paths import FRAME_MAP_PARQUET, map_keyframes_file

EXPECTED_ROWS = 177_321
SAMPLE_SIZE = 20


def main() -> int:
    """
    Thực hiện toàn bộ phép kiểm tra và in kết quả ra màn hình.
    """
    print("=" * 70)
    print("KIỂM TRA TASK 1 - frame_map.parquet")
    print("=" * 70)

    # 1. Đọc Parquet
    frame_map = pd.read_parquet(FRAME_MAP_PARQUET)

    print(f"\n[1] Số dòng: {len(frame_map):,}")

    rows_ok = len(frame_map) == EXPECTED_ROWS

    if rows_ok:
        print("    [OK] Đúng 177.321 dòng")
    else:
        print(
            f"    [LỖI] Mong đợi {EXPECTED_ROWS:,} dòng"
        )

    # 2. Kiểm tra ô trống
    null_count = int(frame_map.isna().sum().sum())

    print("\n[2] KIỂM TRA Ô TRỐNG")

    if null_count == 0:
        print("    [OK] Không có ô nào trống")
    else:
        print(f"    [LỖI] Có {null_count} ô trống")

    nulls_ok = null_count == 0

    # 3. Kiểm tra 20 dòng ngẫu nhiên
    sample = frame_map.sample(
        n=min(SAMPLE_SIZE, len(frame_map)),
        random_state=2026,
    )

    print("\n[3] KIỂM TRA 20 DÒNG NGẪU NHIÊN")
    print("-" * 70)

    sample_ok = True

    for _, row in sample.iterrows():
        video_id = row["video_id"]
        n = int(row["n"])

        csv_path = map_keyframes_file(video_id)
        source = pd.read_csv(csv_path)

        match = source[source["n"] == n]

        if len(match) != 1:
            print(
                f"[LỖI] {video_id} / n={n}: "
                f"CSV gốc có {len(match)} dòng"
            )
            sample_ok = False
            continue

        original = match.iloc[0]

        checks = {
            "n": int(row["n"]) == int(original["n"]),
            "pts_time": abs(
                float(row["pts_time"])
                - float(original["pts_time"])
            ) < 1e-6,
            "fps": abs(
                float(row["fps"])
                - float(original["fps"])
            ) < 1e-6,
            "frame_idx": (
                int(row["frame_idx"])
                == int(original["frame_idx"])
            ),
        }

        if all(checks.values()):
            print(
                f"[OK] {video_id} / n={n} "
                f"→ pts={row['pts_time']}, "
                f"fps={row['fps']}, "
                f"frame_idx={row['frame_idx']}"
            )
        else:
            print(
                f"[LỖI] {video_id} / n={n} "
                f"→ {checks}"
            )
            sample_ok = False

    print("\n" + "=" * 70)

    if rows_ok and nulls_ok and sample_ok:
        print("[ĐẠT] TASK 1 PASS")
        print("      - 177.321 dòng")
        print("      - Không có ô trống")
        print("      - 20 dòng ngẫu nhiên khớp CSV gốc")
        return 0

    print("[LỖI] TASK 1 CHƯA ĐẠT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())