"""
Việc 3 — dựng bảng objects_fts từ raw/objects/.

CÁCH CHẠY (từ gốc D:\\AIC2026):
    python -u -m scripts.build_objects_index
    python -u -m scripts.build_objects_index --video L23_V001 L23_V002
    python -u -m scripts.build_objects_index --liet-ke-nhan
    python -u -m scripts.build_objects_index --thu "bộ trống đỏ và đàn piano"

NGHIỆM THU (đúng dòng "Cách xác nhận đã xong" của checklist):
    --thu "trống và piano"  phải trả về danh sách keyframe hợp lý.
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

if hasattr(sys.stdout, "reconfigure"):        # Windows cp1252 nuốt tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.index.objects_index import (          # noqa: E402
    NGUONG_DIEM_MAC_DINH,
    BangNhan,
    ObjectSearchIndex,
    doc_mot_tep,
)
from aic2026.paths import OBJECTS_DIR, list_video_ids, resolve   # noqa: E402


def liet_ke_nhan(so_video: int = 30, nguong: float = NGUONG_DIEM_MAC_DINH) -> None:
    """In các nhãn hay gặp nhất — để biết cần thêm gì vào nhan_vat_the.yaml."""
    video_ids = list_video_ids(OBJECTS_DIR)[:so_video]
    if not video_ids:
        print(f"Không thấy video nào trong {OBJECTS_DIR}")
        return

    dem: Counter = Counter()
    for vid in video_ids:
        goc = resolve(OBJECTS_DIR, vid) or OBJECTS_DIR / vid
        for tep in sorted(goc.glob("*.json")):
            nhan, _ = doc_mot_tep(tep, nguong)
            dem.update(nhan)

    print(f"Nhãn hay gặp nhất trên {len(video_ids)} video (ngưỡng > {nguong}):\n")
    bang = BangNhan.nap()
    da_co = {n for ds in bang._map.values() for n in ds}
    for ten, so in dem.most_common(100):
        danh_dau = " " if ten in da_co else "  <-- CHƯA CÓ trong nhan_vat_the.yaml"
        print(f"  {so:>7,}  {ten}{danh_dau}")


def main() -> int:
    p = argparse.ArgumentParser(description="Dựng bảng tra ngược vật thể")
    p.add_argument("--video", nargs="*", default=None, help="chỉ nạp các video này")
    p.add_argument("--nguong", type=float, default=NGUONG_DIEM_MAC_DINH)
    p.add_argument("--khong-xoa-truoc", action="store_true")
    p.add_argument("--liet-ke-nhan", action="store_true")
    p.add_argument("--thu", default=None, help="tra thử một câu tiếng Việt")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--tan-suat", action="store_true",
                   help="in nhãn hiếm nhất và phổ biến nhất")
    p.add_argument("--giai-thich-loc", default=None, metavar="CÂU",
                   help="xem bộ lọc giữ lại bao nhiêu khung cho câu này")
    args = p.parse_args()

    if args.liet_ke_nhan:
        liet_ke_nhan(nguong=args.nguong)
        return 0

    kho = ObjectSearchIndex()

    if args.tan_suat:
        ts = kho.tan_suat_nhan()
        tong = kho.tong_so_khung()
        print(f"{len(ts)} nhãn khác nhau trên {tong:,} khung\n")
        xep = sorted(ts.items(), key=lambda t: -t[1])
        print("PHỔ BIẾN NHẤT — lọc theo mấy nhãn này không thu hẹp được gì:")
        for n, c in xep[:12]:
            print(f"  {c:>8,}  {c/tong:>6.1%}  {n}")
        print("\nHIẾM NHẤT (từ 200 khung trở lên) — đây mới là thứ đáng lọc:")
        for n, c in [x for x in xep if x[1] >= 200][-15:]:
            print(f"  {c:>8,}  {c/tong:>6.1%}  {n}")
        return 0

    if args.giai_thich_loc:
        gt = kho.giai_thich_loc(args.giai_thich_loc)
        print(f"Câu hỏi: {args.giai_thich_loc}\n")
        print(f"{'khái niệm':<24}{'số khung':>10}{'tỉ lệ':>9}  nhãn")
        for k in gt["khai_niem"]:
            print(f"  {k['khai_niem']:<22}{k['so_khung']:>10,}{k['ti_le']:>9.2%}  {k['nhan']}")
        print()
        if gt["co_loc"]:
            print(
                f"CÓ LỌC: giữ {gt['so_khung_giu_lai']:,}/{gt['tong_khung']:,} khung "
                f"({gt['ti_le_giu_lai']:.1%})"
            )
        else:
            print(
                "KHÔNG LỌC — hoặc không có khái niệm nào đủ hiếm, hoặc tập lọc\n"
                "ra quá hẹp. Giữ nguyên toàn bộ ứng viên, đây là hành vi an toàn."
            )
        return 0

    if args.thu:
        nhom = kho.bang_nhan.tim_nhom(args.thu)
        nhan = [x for _, ds in nhom for x in ds]
        print(f"Câu hỏi : {args.thu}")
        if nhom:
            print(f"Khái niệm ({len(nhom)}):")
            for ten, ds in nhom:
                print(f"   {ten!r} -> {ds}")
        else:
            print("Khái niệm : (không nhận ra vật thể nào)")
        if not nhan:
            print(
                "\nCâu này không nhắc vật thể nào có trong config/nhan_vat_the.yaml.\n"
                "Nhánh vật thể sẽ BỎ QUA câu này — đó là hành vi đúng, không phải lỗi."
            )
            return 0
        ket_qua = kho.tra_theo_nhan(nhan, top_k=args.top, nhom=nhom)

        from aic2026.index import objects_index as _oi

        if _oi.DA_LUI_VE_OR:
            print(
                f"\nCẢNH BÁO: KHÔNG khung nào có ĐỦ {len(nhom)} khái niệm trên.\n"
                "Kết quả dưới đây chỉ phủ MỘT PHẦN — xem cột 'phủ'. Bộ nhận dạng\n"
                "bỏ sót nhiều, nên đây là chuyện thường; nhưng nếu một khái niệm\n"
                "KHÔNG BAO GIỜ khớp thì nhiều khả năng rút nhầm nhãn."
            )

        print(f"\n{len(ket_qua)} kết quả đầu:\n")
        for i, r in enumerate(ket_qua, 1):
            print(
                f"  {i:>3}. {r['video_id']}  n={r['n']:<5} frame_idx={r['frame_idx']:<7}"
                f" phủ {r['so_nhan_khop']}/{len(nhom)}"
                f"  {'+'.join(r.get('khai_niem_khop', [])) or '-'}"
                f"  {json.dumps(r['dem'], ensure_ascii=False)}"
            )
        return 0

    print(f"Ngưỡng lọc điểm: > {args.nguong} (theo notebook baseline BTC)")
    print(f"Bảng ánh xạ    : {len(kho.bang_nhan)} từ tiếng Việt\n")

    ket_qua = kho.nap_tu_thu_muc(
        video_ids=args.video,
        nguong=args.nguong,
        xoa_truoc=not args.khong_xoa_truoc,
    )

    print("\nXong:")
    for khoa, gia_tri in ket_qua.items():
        print(f"  {khoa:<38} {gia_tri:,}")
    print()
    for khoa, gia_tri in kho.thong_ke().items():
        print(f"  {khoa:<38} {gia_tri:,}")

    if ket_qua["so_khung_khong_tra_duoc_frame_idx"]:
        print(
            f"\nCẢNH BÁO: {ket_qua['so_khung_khong_tra_duoc_frame_idx']:,} khung "
            "không tra được n -> frame_idx nên đã BỎ. Thường là do frame_map "
            "chưa dựng đủ video. Chạy: python -m scripts.build_frame_map"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
