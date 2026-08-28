"""
Việc 8 — dựng chỉ mục FAISS thứ hai từ vector đã mã hoá bằng ViT-L-14.

Chỉ mục BTC (ViT-B/32) GIỮ NGUYÊN. Đây là nhánh song song, tắt được bất cứ lúc
nào bằng cách đặt trọng số 0.

CÁCH CHẠY
    python -u -m scripts.build_faiss_l
    python -u -m scripts.build_faiss_l --kiem        # chỉ soát, không dựng
    python -u -m scripts.build_faiss_l --thu "a man holding an umbrella"

QUY TRÌNH ĐẦY ĐỦ
    1. Chạy notebook mã hoá trên Colab cho shard của mình
    2. Tải thư mục kết quả về  D:\\aic-data\\derived\\clip_l\\
    3. Gộp thư mục của cả bốn máy vào cùng chỗ (mỗi video một tệp .npy)
    4. Chạy script này
    5. Đo:  python -m scripts.do_trong_so_rrf --khong-caption \\
              --nguon-mo-rong marian --clip-l --chi-do-rieng
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.index.clip_l_index import (  # noqa: E402
    DAC_TRUNG_DIR,
    DUONG_DAN_INDEX,
    MO_HINH,
    SO_CHIEU,
    doc_dac_trung,
    dung_chi_muc,
    ghi_chi_muc,
    tep_dac_trung,
    video_da_ma_hoa,
)


def soat() -> int:
    """Soát vector đã có: đủ hàng chưa, đúng số chiều chưa, thiếu video nào."""
    from aic2026.frame_map import load_frame_map

    bang = load_frame_map()
    tat_ca = sorted(bang["video_id"].unique().tolist())
    da_co = video_da_ma_hoa()

    print(f"Mô hình      : {MO_HINH} ({SO_CHIEU} chiều)")
    print(f"Thư mục      : {DAC_TRUNG_DIR}")
    print(f"Đã mã hoá    : {len(da_co)}/{len(tat_ca)} video "
          f"({len(da_co) / max(len(tat_ca), 1):.0%})\n")

    if not da_co:
        print("Chưa có tệp .npy nào. Chạy notebook mã hoá trước.")
        return 1

    hong = []
    tong_hang = 0
    for vid in da_co:
        can = int((bang["video_id"] == vid).sum())
        try:
            v = doc_dac_trung(vid, so_hang_can=can)
            tong_hang += len(v)
        except Exception as loi:
            hong.append((vid, str(loi)[:90]))

    print(f"{tong_hang:,} vector hợp lệ")

    if hong:
        print(f"\n{len(hong)} tệp HỎNG — sẽ không vào chỉ mục:")
        for vid, ly_do in hong[:10]:
            print(f"  {vid}: {ly_do}")
        print("\nMã hoá lại các video đó rồi chạy lại.")

    thieu = [v for v in tat_ca if v not in set(da_co)]
    if thieu:
        shard = {}
        for v in thieu:
            shard[v.split("_")[0]] = shard.get(v.split("_")[0], 0) + 1
        print(f"\nCòn thiếu {len(thieu)} video, theo shard:")
        for s, n in sorted(shard.items(), key=lambda t: -t[1]):
            print(f"  {s}  {n} video")
        print(
            "\nThiếu không sao — nhánh clip_l chỉ vắng mặt ở các video đó.\n"
            "Đây là điểm khác căn bản với thay thế chỉ mục: hybrid dùng được\n"
            "ngay khi có phần nào."
        )

    return 1 if hong else 0


def main() -> int:
    p = argparse.ArgumentParser(description="Việc 8 — dựng chỉ mục clip_l")
    p.add_argument("--kiem", action="store_true", help="chỉ soát, không dựng")
    p.add_argument("--video", nargs="*", default=None)
    p.add_argument("--thu", default=None, help="tra thử một câu TIẾNG ANH")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    if args.thu:
        from aic2026.index.clip_l_index import tim

        for i, r in enumerate(tim(args.thu, args.top), 1):
            print(f"  {i:>3}. {r['video_id']}  n={r['n']:<5} "
                  f"frame_idx={r['frame_idx']:<8} cosine={r['score']:.4f}")
        return 0

    ma = soat()
    if args.kiem:
        return ma
    if ma:
        print("\nCó tệp hỏng — KHÔNG dựng chỉ mục. Sửa rồi chạy lại.")
        return 1

    print("\nDựng chỉ mục...")
    index, ids = dung_chi_muc(args.video)
    ghi_chi_muc(index, ids)

    print(f"  {index.ntotal:,} vector, {ids['video_id'].nunique()} video")
    print(f"  {DUONG_DAN_INDEX}  "
          f"({DUONG_DAN_INDEX.stat().st_size / 1e6:.0f} MB)")
    print(
        "\nĐo A/B:\n"
        "  python -m scripts.do_trong_so_rrf --khong-caption "
        "--nguon-mo-rong marian --chi-do-rieng\n"
        "  python -m scripts.do_trong_so_rrf --khong-caption "
        "--nguon-mo-rong marian --chi-do-rieng --clip-l\n"
        "\nMốc nền hiện tại: clip một mình KIS 0,3667."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
