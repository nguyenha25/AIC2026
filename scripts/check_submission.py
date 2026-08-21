"""
Soát một tệp nộp đã ghi ra đĩa — cổng thoát số 5 và định dạng.

    python -m scripts.check_submission D:\\aic-data\\submissions\\query-01-kis.csv
    python -m scripts.check_submission <tệp> --loai trake

Khác với phép soát nằm trong run_search.py ở chỗ: tệp này đọc tệp CSV ĐÃ GHI
chứ không đọc kết quả trong RAM. Dùng được cho mọi tệp nộp, kể cả tệp do người
khác sinh ra hoặc tệp nộp giả của Task 8.

Giây lấy từ index/frame_map.parquet bằng cách tra ngược frame_idx. Không đoán
theo fps 25 — fps mỗi video một khác, đoán sai là báo nhầm số cặp vi phạm.

LOẠI TRUY VẤN — suy từ đuôi tên tệp (-kis / -qa / -trake), ghi đè bằng --loai.
Bản trước không phân biệt loại: nó coi MỌI cột số sau video_id là frame_idx.
Hậu quả đo được trên bộ 32 câu là hai lỗi báo nhầm, nay đã sửa:

    QA    cột đáp án khi là số (ví dụ "5") bị đọc thành frame_idx rồi báo
          "không có trong frame_map". Nay cột đáp án soát riêng, chỉ kiểm
          rỗng hay không.

    TRAKE k mốc trên CÙNG MỘT DÒNG bị đem so với nhau bằng luật 10 giây.
          Nhưng k mốc đó là MỘT đáp án — các khoảnh khắc của một cú nhảy cao
          cách nhau vài giây là đúng bản chất bài toán, không phải trùng lặp.
          Nay luật 10 giây chỉ áp GIỮA CÁC DÒNG. Xem docs/decisions/
          002-luat-loc-trung-cho-trake.md.

MÃ THOÁT
    0   đạt
    1   trượt cổng 5, hoặc sai định dạng, hoặc có frame_idx không tra được
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from src.aic2026.frame_map import load_frame_map
from src.aic2026.rank.config import cua_so_giay
from src.aic2026.rank.dedupe import tim_cap_qua_gan

KIS = "kis"
QA = "qa"
TRAKE = "trake"

SO_COT_MONG_DOI = {KIS: 2, QA: 3}  # TRAKE thay đổi theo câu, xét riêng


class DongNop:
    """Một mốc trong tệp nộp, đủ ba thuộc tính tim_cap_qua_gan() cần."""

    def __init__(self, video_id: str, frame_idx: int, pts_time: float, so_dong: int):
        self.video_id = video_id
        self.frame_idx = frame_idx
        self.pts_time = pts_time
        self.score = 0.0
        self.so_dong = so_dong

    def __repr__(self) -> str:
        return f"dòng {self.so_dong}: {self.video_id} frame {self.frame_idx}"


def bang_tra_nguoc() -> dict[tuple[str, int], float]:
    """(video_id, frame_idx) -> pts_time, dựng từ frame_map.parquet."""
    df = load_frame_map()
    return {
        (str(v), int(f)): float(t)
        for v, f, t in zip(df["video_id"], df["frame_idx"], df["pts_time"])
    }


def doc_tep_nop(duong_dan: Path) -> list[list[str]]:
    with duong_dan.open("r", encoding="utf-8", newline="") as f:
        return [d for d in csv.reader(f) if d]


def doan_loai(duong_dan: Path) -> str | None:
    """Suy loại truy vấn từ đuôi tên tệp: query-07-trake.csv -> 'trake'."""
    duoi = duong_dan.stem.rsplit("-", 1)[-1].lower()
    return duoi if duoi in (KIS, QA, TRAKE) else None


def la_so(o: str) -> bool:
    return o.strip().lstrip("-").isdigit()


def in_cap_qua_gan(vi_pham, moc, nhan=""):
    for i, j, video_id, khoang_cach in vi_pham[:10]:
        a, b = moc[i], moc[j]
        print(
            f"    {nhan}{video_id}: dòng {a.so_dong} frame {a.frame_idx}"
            f" (giây {a.pts_time:.2f}) và dòng {b.so_dong} frame {b.frame_idx}"
            f" (giây {b.pts_time:.2f}) — cách {khoang_cach:.2f} giây"
        )
    if len(vi_pham) > 10:
        print(f"    … còn {len(vi_pham) - 10} cặp nữa")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    tham_so = [a for a in sys.argv[1:]]

    if not tham_so:
        print("Thiếu đường dẫn tệp nộp.")
        print(r"  python -m scripts.check_submission D:\aic-data\submissions\query-01-kis.csv")
        return 1

    loai_ep = None
    if "--loai" in tham_so:
        i = tham_so.index("--loai")
        if i + 1 >= len(tham_so):
            print("--loai thiếu giá trị (kis / qa / trake).")
            return 1
        loai_ep = tham_so[i + 1].lower()
        del tham_so[i : i + 2]

    duong_dan = Path(tham_so[0])

    if not duong_dan.exists():
        print(f"Không thấy tệp: {duong_dan}")
        return 1

    loai = loai_ep or doan_loai(duong_dan)

    if loai not in (KIS, QA, TRAKE):
        print(f"Không suy được loại truy vấn từ tên tệp: {duong_dan.name}")
        print("Đặt tên theo mẫu query-NN-kis.csv / -qa.csv / -trake.csv,")
        print("hoặc chỉ rõ bằng --loai kis|qa|trake.")
        return 1

    dong = doc_tep_nop(duong_dan)
    cua_so = cua_so_giay()

    print("=" * 72)
    print(f"SOÁT TỆP NỘP: {duong_dan.name}   (loại: {loai})")
    print("=" * 72)

    loi: list[str] = []

    # --- định dạng ----------------------------------------------------------
    print(f"Số dòng            : {len(dong)}")

    if len(dong) > 100:
        loi.append(f"{len(dong)} dòng, BTC cho tối đa 100")
    elif len(dong) < 100:
        print(f"                     (bỏ phí {100 - len(dong)} suất)")

    so_cot = sorted({len(d) for d in dong})
    print(f"Số cột             : {so_cot}")

    if len(so_cot) > 1:
        loi.append(f"số cột không đều: {so_cot}")

    so_moc_moi_dong = 0

    if loai in SO_COT_MONG_DOI:
        can = SO_COT_MONG_DOI[loai]
        if so_cot != [can]:
            loi.append(f"{loai} cần đúng {can} cột, tệp có {so_cot}")
        so_moc_moi_dong = 1
    else:  # TRAKE
        if len(so_cot) == 1:
            so_moc_moi_dong = so_cot[0] - 1
            print(f"Số mốc mỗi dòng    : {so_moc_moi_dong}")
            if so_moc_moi_dong < 2:
                loi.append(f"trake chỉ có {so_moc_moi_dong} mốc mỗi dòng, cần từ 2")

    raw = duong_dan.read_bytes()

    if raw.startswith(b"\xef\xbb\xbf"):
        loi.append("tệp có BOM ở đầu")

    if b"\r\n" in raw:
        loi.append("xuống dòng kiểu CRLF, cần \\n")

    khoa = [tuple(d) for d in dong]
    trung = len(khoa) - len(set(khoa))

    print(f"Dòng trùng nhau    : {trung}")

    if trung:
        loi.append(f"{trung} dòng trùng — mỗi dòng trùng phí một suất")

    # --- cột đáp án của QA --------------------------------------------------
    if loai == QA:
        rong = [i for i, d in enumerate(dong, start=1) if len(d) < 3 or not d[2].strip()]
        print(f"Đáp án rỗng        : {len(rong)}")
        if rong:
            loi.append(f"{len(rong)} dòng thiếu đáp án — QA không có đáp án là mất điểm")

    # --- tra giây -----------------------------------------------------------
    # Chỉ các cột MỐC mới đem tra. QA: cột 1. KIS: cột 1. TRAKE: cột 1..k.
    tra = bang_tra_nguoc()
    moc_theo_cot: list[list[DongNop]] = [[] for _ in range(max(so_moc_moi_dong, 1))]
    khong_tra_duoc: list[str] = []
    khong_phai_so: list[str] = []
    tong_moc = 0

    for i, d in enumerate(dong, start=1):
        video_id = d[0]
        cot_moc = d[1 : 1 + so_moc_moi_dong]

        for j, o in enumerate(cot_moc):
            if not la_so(o):
                khong_phai_so.append(f"dòng {i}: cột mốc {j + 1} không phải số: {o!r}")
                continue

            tong_moc += 1
            frame_idx = int(o)
            pts_time = tra.get((video_id, frame_idx))

            if pts_time is None:
                khong_tra_duoc.append(f"dòng {i}: {video_id} frame {frame_idx}")
                continue

            moc_theo_cot[j].append(DongNop(video_id, frame_idx, pts_time, i))

    tra_duoc = sum(len(x) for x in moc_theo_cot)
    print(f"Mốc tra được giây  : {tra_duoc} / {tong_moc}")

    if khong_phai_so:
        loi.append(f"{len(khong_phai_so)} cột mốc không phải số")
        for x in khong_phai_so[:5]:
            print(f"    {x}")

    if khong_tra_duoc:
        loi.append(
            f"{len(khong_tra_duoc)} mốc không có trong frame_map — "
            "nhiều khả năng đang ghi n thay vì frame_idx"
        )
        for x in khong_tra_duoc[:5]:
            print(f"    {x}")

    # --- thứ tự thời gian trong một dòng TRAKE ------------------------------
    if loai == TRAKE and so_moc_moi_dong >= 2:
        nghich = 0
        for d in dong:
            so = [int(o) for o in d[1 : 1 + so_moc_moi_dong] if la_so(o)]
            if any(b <= a for a, b in zip(so, so[1:])):
                nghich += 1
        print(f"Dòng sai thứ tự    : {nghich}")
        if nghich:
            print("    (các giai đoạn phải tăng dần theo thời gian — xem lại nhánh TRAKE)")
            loi.append(f"{nghich} dòng có mốc không tăng dần")

    # --- cổng thoát số 5 ----------------------------------------------------
    print()
    print(f"CỔNG THOÁT SỐ 5 (cửa sổ {cua_so:.1f} giây)")

    tong_vi_pham = 0

    if loai == TRAKE:
        # Luật 10 giây áp GIỮA CÁC DÒNG, so từng cột sự kiện với chính nó.
        # Hai chuỗi ứng viên mà khoảnh khắc thứ j cách nhau dưới 10 giây thì
        # thực chất là cùng một đáp án — dòng sau phí một suất.
        print("  (trake: so giữa các dòng, từng cột sự kiện một)")
        for j, moc in enumerate(moc_theo_cot, start=1):
            vi_pham = tim_cap_qua_gan(moc, cua_so_giay=cua_so)
            if vi_pham:
                print(f"  cột sự kiện {j}: {len(vi_pham)} cặp quá gần")
                in_cap_qua_gan(vi_pham, moc)
                tong_vi_pham += len(vi_pham)
    else:
        moc = moc_theo_cot[0]
        vi_pham = tim_cap_qua_gan(moc, cua_so_giay=cua_so)
        if vi_pham:
            in_cap_qua_gan(vi_pham, moc)
            tong_vi_pham = len(vi_pham)

    if tong_vi_pham:
        print(f"  TRƯỢT — {tong_vi_pham} cặp cùng video cách nhau dưới {cua_so:.1f} giây")
        loi.append(f"trượt cổng 5: {tong_vi_pham} cặp quá gần")
    else:
        print("  ĐẠT")

    # --- kết luận -----------------------------------------------------------
    print()

    if loi:
        print("KHÔNG ĐẠT:")
        for x in loi:
            print(f"  - {x}")
        return 1

    print("ĐẠT — tệp này nộp được.")
    return 0


if __name__ == "__main__":
    sys.exit(main())