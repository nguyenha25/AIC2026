"""
Chẩn đoán vì sao TRAKE chấm 0.000 — tách bốn tầng nguyên nhân.

    python -m scripts.chan_doan_trake

Chỉ ĐỌC. Không sửa tệp nộp, không sửa mã của ai.

BỐN TẦNG, soát theo đúng thứ tự này. Tầng nào hỏng thì các tầng sau
vô nghĩa, nên script dừng báo ở tầng hỏng đầu tiên của mỗi câu.

    Tầng 1  SỐ MỐC     tệp nộp có đúng len(cac_giai_doan) mốc không.
                       Lệch một mốc là chấm 0 dù tìm đúng hoàn toàn.
    Tầng 2  TRẦN       frame_map có keyframe nào nằm trong khoảng
                       [frame_start, frame_end] của từng giai đoạn không.
                       Không có thì trần của câu đó là 0, hết cách.
    Tầng 3  VIDEO      có dòng nào trong 100 dòng đúng gt_video_id không,
                       và nằm hạng bao nhiêu.
    Tầng 4  KHOẢNG     với dòng đúng video, mốc thứ j có rơi vào
                       [s_j, e_j] không.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

from src.aic2026.paths import (
    DEV_QUERIES_PATH,
    FRAME_MAP_PARQUET,
    SUBMISSIONS_DIR,
)

LOAI_TRAKE = "chuoi_su_kien"


def doc_dev() -> list[dict]:
    with DEV_QUERIES_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(d) for d in f if d.strip()]


def doc_nop(duong_dan: Path) -> list[list[str]]:
    with duong_dan.open("r", encoding="utf-8", newline="") as f:
        return [d for d in csv.reader(f) if d]


def keyframe_trong_khoang(df: pd.DataFrame, video_id: str, s: int, e: int) -> list[int]:
    """frame_idx của mọi keyframe rơi vào [s, e] — dùng để đo trần."""
    m = (df["video_id"] == video_id) & (df["frame_idx"] >= s) & (df["frame_idx"] <= e)
    return sorted(int(x) for x in df.loc[m, "frame_idx"])


def soat_mot_cau(q: dict, df: pd.DataFrame) -> str:
    """Trả về nhãn nguyên nhân gọn để tổng kết ở cuối."""
    ma = str(q["id"]).zfill(2)
    video_gt = q["video_id"]
    giai_doan = q["cac_giai_doan"]
    khoang = [(int(g["frame_start"]), int(g["frame_end"])) for g in giai_doan]
    k_gt = len(khoang)

    print()
    print("-" * 72)
    print(f"CÂU {ma} — GT video {video_gt}, {k_gt} giai đoạn")

    tep = SUBMISSIONS_DIR / f"query-{ma}-trake.csv"
    if not tep.exists():
        print(f"  Không thấy tệp nộp: {tep.name}")
        return "thiếu tệp"

    dong = doc_nop(tep)

    # --- tầng 1: số mốc ----------------------------------------------------
    so_moc = sorted({len(d) - 1 for d in dong})
    print(f"  [1] Số mốc tệp nộp : {so_moc}   (GT cần {k_gt})")

    if so_moc != [k_gt]:
        print(f"      LỆCH — chấm 0 ngay bất kể tìm đúng hay sai.")
        return "lệch số mốc"

    # --- tầng 2: trần ------------------------------------------------------
    print("  [2] Trần dữ liệu   :")
    tran_hong = False
    for j, (s, e) in enumerate(khoang, start=1):
        co = keyframe_trong_khoang(df, video_gt, s, e)
        if co:
            print(f"      mốc {j} [{s}–{e}] : {len(co)} keyframe {co[:4]}")
        else:
            print(f"      mốc {j} [{s}–{e}] : KHÔNG CÓ KEYFRAME  <-- trần 0")
            tran_hong = True

    if tran_hong:
        print("      Trần của câu này là 0.000. Không phải lỗi, là giới hạn Task 2.")
        return "trần 0"

    # --- tầng 3: video -----------------------------------------------------
    hang_dung = [i for i, d in enumerate(dong, start=1) if d[0] == video_gt]
    print(f"  [3] Dòng đúng video: {len(hang_dung)}/{len(dong)}", end="")
    if hang_dung:
        print(f"   (hạng đầu tiên: {hang_dung[0]})")
    else:
        print()
        print("      Không dòng nào đúng video. Hỏng ở TẦNG TRA CỨU,")
        print("      chưa tới lượt căn thời gian. So với KIS cùng shard để")
        print("      biết là lỗi truy vấn hay lỗi nhánh TRAKE.")
        return "sai video"

    # --- tầng 4: khoảng ----------------------------------------------------
    print("  [4] Căn khoảng     :")
    tot_nhat = -1
    for i in hang_dung[:5]:
        d = dong[i - 1]
        moc = [int(x) for x in d[1:]]
        dung = 0
        chi_tiet = []
        for j, (f, (s, e)) in enumerate(zip(moc, khoang), start=1):
            ok = s <= f <= e
            dung += ok
            chi_tiet.append(f"m{j}={f}{'✓' if ok else f'✗[{s}–{e}]'}")
        tot_nhat = max(tot_nhat, dung)
        print(f"      hạng {i:>3}: {dung}/{k_gt} đúng   " + "  ".join(chi_tiet))

    if len(hang_dung) > 5:
        print(f"      … còn {len(hang_dung) - 5} dòng đúng video nữa")

    if tot_nhat == 0:
        print("      Đúng video nhưng KHÔNG mốc nào vào khoảng.")
        print("      Nghi ngờ: mốc nộp lấy từ keyframe gần nhất chứ chưa")
        print("      căn theo thứ tự thời gian của các giai đoạn.")
        return "sai khoảng hoàn toàn"

    return f"đúng {tot_nhat}/{k_gt} mốc"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 72)
    print("CHẨN ĐOÁN TRAKE")
    print("=" * 72)
    print(f"Bộ dev   : {DEV_QUERIES_PATH}")
    print(f"Tệp nộp  : {SUBMISSIONS_DIR}")

    df = pd.read_parquet(FRAME_MAP_PARQUET, columns=["video_id", "frame_idx"])
    cau = [q for q in doc_dev() if q.get("loai_truy_van") == LOAI_TRAKE]
    print(f"Số câu   : {len(cau)} câu chuỗi sự kiện")

    tong_ket: list[tuple[str, str]] = []
    for q in cau:
        nhan = soat_mot_cau(q, df)
        tong_ket.append((str(q["id"]).zfill(2), nhan))

    print()
    print("=" * 72)
    print("TỔNG KẾT — nguyên nhân của từng câu")
    print("=" * 72)
    for ma, nhan in tong_ket:
        print(f"  câu {ma} : {nhan}")

    print()
    print("Câu mang nhãn 'trần 0' là giới hạn dữ liệu, ghi vào Nhật ký rồi bỏ qua.")
    print("Mọi nhãn khác là lỗi sửa được — ưu tiên nhãn xuất hiện nhiều nhất.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
