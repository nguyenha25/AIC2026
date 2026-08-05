"""
TASK 5 + TASK 7 — nạp dữ liệu THẬT và kiểm phép tính.

Chạy:
    python -m scripts.peek_data                 # tự bốc một video
    python -m scripts.peek_data L21_V001        # chỉ định video
    python -m scripts.peek_data L21_V001 --rows 5

In ra bảng năm dòng: giây, số hình mỗi giây, vị trí GHI TRONG TỆP,
vị trí TỰ TÍNH ra, và chênh lệch. Chênh lệch phải bằng 0 hoặc 1.

MỖI NGƯỜI CHẠY MỘT VIDEO KHÁC NHAU rồi chụp màn hình gửi nhóm.
Bốn máy cộng lại là 20 dòng, cả 20 dòng phải lệch ≤ 1.

Không ai được sửa dòng nào trong tệp này để chạy được — nếu phải sửa
đường dẫn thì tức là paths.py hoặc .env có vấn đề, báo nhóm.
"""

import argparse

from src.aic2026.frame_map import FrameMap, available_video_ids

MAX_ALLOWED_DRIFT = 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id", nargs="?", help="vd: L21_V001")
    ap.add_argument("--rows", type=int, default=5, help="in bao nhiêu dòng")
    args = ap.parse_args()

    ids = available_video_ids()
    if not ids:
        print("Chưa có tệp map-keyframes nào trên máy. Làm Task 4 trước.")
        return 1

    video_id = args.video_id or ids[0]
    if video_id not in ids:
        print(f"Không thấy {video_id}. Vài video đang có: {ids[:5]}")
        return 1

    fm = FrameMap.load(video_id)

    print("=" * 74)
    print(f"VIDEO: {video_id}   |   tổng {len(fm)} tấm ảnh")
    print("=" * 74)
    print("Nhắc lại ý nghĩa bốn cột:")
    print("  n         số thứ tự tấm ảnh (0001.jpg -> n = 1)")
    print("  pts_time  tấm ảnh nằm ở giây thứ mấy")
    print("  fps       video quay bao nhiêu hình một giây")
    print("  frame_idx VỊ TRÍ KHUNG HÌNH TRONG VIDEO  <-- SỐ NÀY MỚI ĐEM NỘP")
    print()

    header = (
        f"{'n':>5} {'pts_time':>10} {'fps':>7} "
        f"{'frame_idx':>11} {'tự tính':>10} {'lệch':>6}"
    )
    print(header)
    print("-" * 74)

    rows = fm.rows[: args.rows]
    worst = 0
    for r in rows:
        worst = max(worst, r.drift)
        print(
            f"{r.n:>5} {r.pts_time:>10.3f} {r.fps:>7.2f} "
            f"{r.frame_idx:>11} {r.expected_frame_idx:>10} {r.drift:>6}"
        )
    print("-" * 74)

    full_worst = fm.max_drift()
    print(f"Lệch lớn nhất trong {args.rows} dòng in ra : {worst}")
    print(f"Lệch lớn nhất trên CẢ {len(fm)} dòng      : {full_worst}")
    print()

    # Minh hoạ cụ thể chỗ dễ sai nhất
    sample = rows[-1] if rows else None
    if sample:
        print("VÍ DỤ ĐỂ THẤY HAI SỐ KHÁC NHAU THẾ NÀO:")
        print(f"  Tấm ảnh {sample.image_name} là ảnh thứ {sample.n} của video.")
        print(f"  Nhưng phải nộp frame_id = {sample.frame_idx}.")
        print(f"  Nộp nhầm {sample.n} thì BTC chấm 0 điểm.")
        print()

    if full_worst > MAX_ALLOWED_DRIFT:
        print(f"[LỖI] Lệch {full_worst} > {MAX_ALLOWED_DRIFT}. "
              "Hoặc nhóm hiểu sai cột, hoặc dữ liệu có vấn đề. BÁO NHÓM.")
        return 1

    print("[ĐẠT] frame_idx khớp với pts_time × fps trên toàn bộ video.")
    print()
    print("Chụp màn hình phần này gửi nhóm chat. Nhớ ghi rõ tên video mình chạy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
