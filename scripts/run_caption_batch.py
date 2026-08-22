"""
Việc 10 — sinh caption cho keyframe rồi nạp vào bảng caption_fts.

CÁCH CHẠY:
    python -u -m scripts.run_caption_batch --video L23_V001 L23_V002
    python -u -m scripts.run_caption_batch --shard L23        # cả shard
    python -u -m scripts.run_caption_batch --chi-nap          # bỏ qua sinh
    python -u -m scripts.run_caption_batch --thu L23_V001 5   # xem thử 5 câu

ĐO THỬ TỐC ĐỘ TRƯỚC KHI CHẠY CẢ SHARD:
    python -u -m scripts.run_caption_batch --thu L23_V001 5
In ra số giây mỗi ảnh. Nhân với số keyframe của shard để biết mất bao lâu —
đừng khởi động một mẻ 30 giờ mà không biết trước.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# NẠP torch TRƯỚC MỌI THỨ KHÁC. ĐỪNG DỜI KHỐI NÀY XUỐNG DƯỚI.
#
# Trên Windows, nạp pandas (hoặc faiss, onnxruntime) vào tiến trình TRƯỚC torch
# thì tiến trình chết ngay với 0xC0000005: không traceback, không thông báo,
# màn hình chỉ dừng giữa chừng rồi về dấu nhắc — nhìn y như chạy xong. Mỗi thư
# viện mang một bản OpenMP riêng, bản nạp sau giẫm lên bản nạp trước.
#
# Script này chạm frame_map (pandas) trước khi chạm mô hình (torch), nên nếu
# không ghim thứ tự ở đây thì thứ tự nạp là ngẫu nhiên theo đường mã chạy.
# Cùng một cái bẫy đã ghi ở rank/search.py, hàm tim_ung_vien_clip().
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
except ImportError:
    pass          # máy chưa cài torch: để phần kiểm phụ thuộc báo lỗi tử tế

from aic2026.enrich.caption import (        # noqa: E402
    MO_HINH_MAC_DINH,
    BoSinhCaption,
    CaptionSearchIndex,
    sinh_cho_video,
)
from aic2026.paths import KEYFRAMES_DIR, list_video_ids   # noqa: E402


def chon_video(args) -> list[str]:
    co_anh = list_video_ids(KEYFRAMES_DIR)
    if args.video:
        thieu = [v for v in args.video if v not in co_anh]
        if thieu:
            print(f"CẢNH BÁO: chưa tải ảnh của {', '.join(thieu)} — sẽ bỏ qua")
        return [v for v in args.video if v in co_anh]
    if args.shard:
        return [v for v in co_anh if any(v.startswith(s) for s in args.shard)]
    return co_anh


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", nargs="*", default=None)
    p.add_argument("--shard", nargs="*", default=None, help="ví dụ L23 L27 L30")
    p.add_argument("--mo-hinh", default=MO_HINH_MAC_DINH)
    p.add_argument("--thiet-bi", default=None, help="cpu / cuda")
    p.add_argument("--lo", type=int, default=8)
    p.add_argument("--lam-lai", action="store_true")
    p.add_argument("--chi-nap", action="store_true", help="chỉ nạp JSONL vào FTS")
    p.add_argument("--thu", nargs=2, metavar=("VIDEO", "SO_ANH"), default=None)
    args = p.parse_args()

    if args.thu:
        from aic2026.kf_index import hang_sang_moc, so_hang
        from aic2026.paths import keyframe_image

        vid, so = args.thu[0], int(args.thu[1])
        bo_sinh = BoSinhCaption(args.mo_hinh, args.thiet_bi, kich_thuoc_lo=so)
        moc = [hang_sang_moc(vid, i) for i in range(min(so, so_hang(vid)))]
        duong_dan = [keyframe_image(vid, m.n) for m in moc]

        bat_dau = time.perf_counter()
        cap = bo_sinh.sinh(duong_dan)
        giay = time.perf_counter() - bat_dau

        for m, c in zip(moc, cap):
            print(f"  {m.video_id} n={m.n:<5} frame_idx={m.frame_idx:<7} {c}")
        print(f"\n{giay / max(len(moc), 1):.2f} giây/ảnh (đã tính cả lúc nạp mô hình)")
        print(f"Cả kho 177.321 ảnh ≈ {177321 * giay / max(len(moc), 1) / 3600:.1f} giờ")
        return 0

    video_ids = chon_video(args)
    if not video_ids and not args.chi_nap:
        print("Không có video nào để làm.")
        return 1

    if not args.chi_nap:
        print(f"Sinh caption cho {len(video_ids)} video — mô hình {args.mo_hinh}\n")
        bo_sinh = BoSinhCaption(args.mo_hinh, args.thiet_bi, args.lo)
        for thu_tu, vid in enumerate(video_ids, 1):
            bat_dau = time.perf_counter()
            kq = sinh_cho_video(vid, bo_sinh, lam_lai=args.lam_lai)
            if kq["bo_qua"]:
                print(f"  [{thu_tu}/{len(video_ids)}] {vid}: đã có, bỏ qua")
            else:
                print(
                    f"  [{thu_tu}/{len(video_ids)}] {vid}: {kq['so_dong']:,} caption "
                    f"({time.perf_counter() - bat_dau:.0f}s"
                    + (f", thiếu {kq['so_thieu_anh']} ảnh" if kq.get("so_thieu_anh") else "")
                    + ")"
                )

    print("\nNạp vào bảng caption_fts...")
    kho = CaptionSearchIndex()
    kq = kho.nap_tu_thu_muc(video_ids=video_ids or None)
    print(f"  {kq['so_tep']} tệp, {kq['so_dong_nap']:,} dòng")
    for k, v in kho.thong_ke().items():
        print(f"  {k:<26} {v:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
