"""
Chẩn đoán các mốc KHÔNG nằm trên lưới keyframe.

    python -m scripts.chan_doan_lech_luoi D:\\aic-data\\submissions\\p1_final\\submission

VÌ SAO CẦN
----------
Bộ soát bản 3 không còn coi mốc lệch lưới là lỗi — bài nộp được trỏ vào BẤT KỲ
khung nào trong video, và bản nộp BTC đã công nhận cũng có những mốc như vậy.
Nhưng "hợp lệ" không có nghĩa là "cố ý". Có hai khả năng, và chúng dẫn tới hai
hành động trái ngược:

  A. TINH CHỈNH TAY (PotPlayer). Người ngồi máy xem video, chọn đúng khoảnh
     khắc mình muốn thay vì chấp nhận keyframe gần nhất.
     -> Tốt. Đây chính là quy trình Việc 11. Không phải sửa gì.

  B. LỖI Ở ĐÂU ĐÓ. Sai số làm tròn, cộng trừ nhầm vài khung, hoặc đọc số khung
     của PotPlayer thay vì đọc số giây rồi mới tính.
     -> Xấu. Mỗi mốc lệch là một suất trong trăm suất có thể mất.

BẢN 2 — SỬA MỘT KẾT LUẬN SAI
Bản 1 chỉ nhìn ĐỘ LỆCH (biên độ và số giá trị khác nhau) rồi kết luận "tinh
chỉnh tay". Chạy trên bản nộp p1 thật, kết luận đó SAI, vì bản 1 bỏ qua hai
dấu vết mạnh hơn nhiều:

  · VỊ TRÍ DÒNG. 37/40 mốc lệch nằm đúng ở dòng 2 và dòng 3, ở cả 19 tệp.
    Người chỉnh tay sẽ chỉnh ứng viên SỐ 1, không chỉnh đúng dòng 2 và 3 của
    mọi tệp.
  · TÍNH ĐỐI XỨNG. Dòng 2 luôn nằm TRƯỚC dòng 1, dòng 3 luôn nằm SAU, biên độ
    gần bằng nhau (−828/+840, −407/+446, −377/+308).

Cộng thêm: |lệch so với dòng 1| tăng dần ở 91–95 trên 99 dòng. Đó là dấu vết
của phép LẤY MẪU TỪ TÂM RA NGOÀI trong một khoảng thời gian — dòng 1 là tâm,
dòng 2 và 3 là hai biên, các dòng sau lấp dần vào trong.

Nên bản 2 xét theo thứ tự: vị trí dòng trước, đối xứng sau, biên độ sau cùng.

MÃ THOÁT
    0   chạy xong
    1   không đọc được đầu vào
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.aic2026.frame_map import load_frame_map


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    goc = Path(sys.argv[1])
    if not goc.exists():
        print(f"Không thấy: {goc}")
        return 1

    cac_tep = sorted(goc.glob("*.csv")) if goc.is_dir() else [goc]
    if not cac_tep:
        print(f"Không có tệp .csv nào trong {goc}")
        return 1

    df = load_frame_map()
    theo_video: dict[str, list[int]] = {}
    for v, f in zip(df["video_id"], df["frame_idx"]):
        theo_video.setdefault(str(v), []).append(int(f))
    for v in theo_video:
        theo_video[v].sort()

    luoi = {(v, f) for v, fs in theo_video.items() for f in fs}

    print("=" * 74)
    print(f"CHẨN ĐOÁN MỐC LỆCH LƯỚI — {len(cac_tep)} tệp")
    print("=" * 74)

    tat_ca_lech: list[tuple[str, str, int, int, int]] = []
    tong_moc = 0

    for tep in cac_tep:
        loai = tep.stem.split("-")[-1].lower()
        with tep.open("r", encoding="utf-8-sig", newline="") as fh:
            dong = [d for d in csv.reader(fh) if d]

        cot = slice(1, 2) if loai in ("kis", "qa") else slice(1, None)

        for i, d in enumerate(dong, 1):
            vid = d[0]
            for c in d[cot]:
                c = c.strip()
                if not c.lstrip("-").isdigit():
                    continue
                f_idx = int(c)
                tong_moc += 1
                if (vid, f_idx) in luoi:
                    continue

                ds = theo_video.get(vid)
                if not ds:
                    continue

                gan = min(ds, key=lambda k: abs(k - f_idx))
                tat_ca_lech.append((tep.name, vid, i, f_idx, f_idx - gan))

    if not tat_ca_lech:
        print("Không có mốc nào lệch lưới. Mọi mốc đều là keyframe.")
        return 0

    print(f"{len(tat_ca_lech)}/{tong_moc} mốc lệch lưới ({len(tat_ca_lech) / tong_moc:.1%})")
    print()
    print(f"{'tệp':<26} {'video':<12} {'dòng':>5} {'frame':>8} {'lệch':>6}")
    print("-" * 74)
    for ten, vid, i, f_idx, lech in tat_ca_lech[:40]:
        print(f"{ten:<26} {vid:<12} {i:>5} {f_idx:>8} {lech:>+6}")
    if len(tat_ca_lech) > 40:
        print(f"… còn {len(tat_ca_lech) - 40} mốc nữa")

    lech = [x[4] for x in tat_ca_lech]
    dem = Counter(lech)

    print()
    print("-" * 74)
    print("PHÂN BỐ ĐỘ LỆCH (khung, âm = trước keyframe gần nhất)")
    for gt, sl in dem.most_common(12):
        print(f"   {gt:>+6} khung : {sl:>4} mốc  {'#' * min(sl, 40)}")

    bien_do = max(abs(x) for x in lech)
    duong = sum(1 for x in lech if x > 0)
    am = sum(1 for x in lech if x < 0)

    print()
    print("-" * 74)
    print(f"Biên độ lệch lớn nhất : {bien_do} khung")
    print(f"Lệch về sau / về trước: {duong} / {am}")
    print(f"Số giá trị lệch khác nhau: {len(dem)}")
    print()

    # Kết luận — dựa vào dấu vết, nói rõ mức tin cậy.
    # ---- Dấu vết 1 (mạnh nhất): VỊ TRÍ DÒNG ----
    vi_tri = Counter(x[2] for x in tat_ca_lech)
    dau_bang = sum(sl for d_, sl in vi_tri.items() if d_ <= 3)
    ty_le_dau = dau_bang / len(tat_ca_lech)

    print("-" * 74)
    print("VỊ TRÍ DÒNG của các mốc lệch")
    for d_, sl in vi_tri.most_common(8):
        print(f"   dòng {d_:>4} : {sl:>4} mốc  {'#' * min(sl, 40)}")
    print(f"   Nằm ở ba dòng đầu: {dau_bang}/{len(tat_ca_lech)} ({ty_le_dau:.0%})")

    print()
    print("=" * 74)

    if ty_le_dau >= 0.8:
        print("KẾT LUẬN: LỖI/ĐẶC TÍNH HỆ THỐNG, KHÔNG PHẢI TINH CHỈNH TAY.")
        print(f"  {ty_le_dau:.0%} số mốc lệch nằm trong ba dòng đầu của tệp.")
        print("  Người chỉnh tay sẽ chỉnh ứng viên SỐ 1, không chỉnh đúng dòng 2 và 3")
        print("  ở mọi tệp. Đây là dấu vết của chương trình, không phải của người.")
        print()
        print("  Nghi ngờ hàng đầu: bộ dựng tệp nộp nhận vào một KHOẢNG THỜI GIAN rồi")
        print("  lấy mẫu TỪ TÂM RA NGOÀI. Dòng 1 là tâm (một keyframe thật), dòng 2 và 3")
        print("  là hai BIÊN của khoảng — biên rơi vào giây bất kỳ nên không trùng lưới.")
        print()
        print("  Cách xác nhận: xem |frame − dòng1| có tăng dần theo số dòng không,")
        print("  và dòng 2 có luôn nằm TRƯỚC dòng 1 còn dòng 3 luôn nằm SAU không.")
        print("  Nếu đúng thì KHÔNG PHẢI LỖI — chỉ là cách bộ dựng hoạt động.")
    elif len(dem) <= 3 and bien_do <= 5:
        print("KẾT LUẬN: nghiêng về LỖI HỆ THỐNG.")
        print(f"  Chỉ {len(dem)} giá trị lệch khác nhau, biên độ tối đa {bien_do} khung.")
        print("  Nghi ngờ: dùng round() thay floor(), hoặc cộng/trừ nhầm một khung.")
    elif bien_do >= 10 and len(dem) >= 8 and ty_le_dau < 0.4:
        print("KẾT LUẬN: nghiêng về TINH CHỈNH TAY — không phải lỗi.")
        print(f"  {len(dem)} giá trị lệch khác nhau, biên độ tới {bien_do} khung,")
        print(f"  và chỉ {ty_le_dau:.0%} nằm ở ba dòng đầu — tức rải khắp tệp.")
        print("  Việc cần làm: ghi quy trình đó vào runbook.")
    else:
        print("KẾT LUẬN: CHƯA KẾT LUẬN ĐƯỢC — dấu vết nằm giữa các khả năng.")
        print("  Cách phân định rẻ nhất: hỏi thẳng người đã dựng bản nộp.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
