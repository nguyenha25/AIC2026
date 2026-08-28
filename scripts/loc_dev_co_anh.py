"""
LỌC BỘ DEV XUỐNG NHỮNG CÂU CÓ ẢNH TRÊN MÁY NÀY.

VÌ SAO CẦN
----------
Việc 4 (rerank), 10 (caption) và 11 (Q&A) đều cần ẢNH GỐC. Máy chỉ tải một
phần kho thì phần lớn ứng viên không có ảnh, và các việc đó xử lý chúng theo
cách máy móc — rerank đẩy xuống cuối, caption bỏ qua, Q&A không đọc được.

Đo trên bộ dev đầy đủ trong tình huống đó là đo SHARD, không đo thuật toán.
Đã cắn thật: rerank làm nhánh CLIP tụt 0,3667 -> 0,2833 trên máy có ảnh cho
8/32 câu, và con số đó không nói gì về chất lượng rerank.

Script này lọc ra bộ con mà máy hiện tại đo được, rồi báo cỡ mẫu còn lại —
để biết ngay phép đo có đáng tin hay không.

CÁCH CHẠY
    python -u -m scripts.loc_dev_co_anh
    python -u -m scripts.do_trong_so_rrf --tep D:\\aic-data\\dev\\dev_co_anh.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.paths import DEV_QUERIES_PATH, KEYFRAMES_DIR, list_video_ids  # noqa: E402

# Dưới mức này thì mọi chênh lệch đều chìm trong nhiễu.
CO_MAU_TOI_THIEU = 15


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tep", default=str(DEV_QUERIES_PATH))
    p.add_argument("--ra", default=None)
    args = p.parse_args()

    nguon = Path(args.tep)
    dich = Path(args.ra) if args.ra else nguon.with_name("dev_co_anh.jsonl")

    co_anh = set(list_video_ids(KEYFRAMES_DIR))
    if not co_anh:
        print(f"Không thấy ảnh keyframe nào trong {KEYFRAMES_DIR}.")
        return 1

    cau = [
        json.loads(d) for d in nguon.open("r", encoding="utf-8") if d.strip()
    ]
    giu = [q for q in cau if str(q["video_id"]) in co_anh]

    print(f"Shard có ảnh: {sorted({v.split('_')[0] for v in co_anh})}")
    print(f"Bộ dev gốc  : {len(cau)} câu")
    print(f"Giữ lại     : {len(giu)} câu ({len(giu) / max(len(cau), 1):.0%})\n")

    dem_giu = Counter(q.get("loai_truy_van") for q in giu)
    dem_bo = Counter(
        q["video_id"].split("_")[0] for q in cau if str(q["video_id"]) not in co_anh
    )

    print("Còn lại theo dạng:")
    for dang, so in sorted(dem_giu.items()):
        print(f"  {dang:<16} {so}")

    print("\nBỏ đi, theo shard:")
    for shard, so in dem_bo.most_common():
        print(f"  {shard}  {so} câu")

    dich.parent.mkdir(parents=True, exist_ok=True)
    with dich.open("w", encoding="utf-8") as f:
        for q in giu:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    so_cham = sum(
        1 for q in giu if q.get("loai_truy_van") in ("mo_ta", "hoi_dap")
    )
    print(f"\nĐã ghi {dich}  ({so_cham} câu KIS/QA được chấm)")

    if so_cham < CO_MAU_TOI_THIEU:
        sai_so = (0.25 / max(so_cham, 1)) ** 0.5
        print(
            f"\n  CỠ MẪU QUÁ NHỎ: {so_cham} câu, sai số cỡ ±{sai_so:.0%}.\n"
            "  Mọi chênh lệch nhỏ hơn mức đó KHÔNG kết luận được gì.\n"
            "  Đo Việc 4 phải làm trên máy giữ shard chứa phần lớn bộ dev —\n"
            "  đếm cột 'Bỏ đi, theo shard' ở trên để biết là shard nào."
        )
        return 1

    print(
        f"\n  {so_cham} câu, sai số cỡ ±{(0.25 / so_cham) ** 0.5:.0%}. "
        "Dùng được, nhưng nhớ ghi cỡ mẫu vào sổ điểm."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
