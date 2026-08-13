"""
Soát một tệp nộp đã ghi ra đĩa — cổng thoát số 5 và định dạng.

    python -m scripts.check_submission D:\\aic-data\\submissions\\query-demo-kis.csv

Khác với phép soát nằm trong run_search.py ở chỗ: tệp này đọc tệp CSV ĐÃ GHI
chứ không đọc kết quả trong RAM. Dùng được cho mọi tệp nộp, kể cả tệp do người
khác sinh ra hoặc tệp nộp giả của Task 8.

Giây lấy từ index/frame_map.parquet bằng cách tra ngược frame_idx. Không đoán
theo fps 25 — fps mỗi video một khác, đoán sai là báo nhầm số cặp vi phạm.

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


def main() -> int:
    if len(sys.argv) < 2:
        print("Thiếu đường dẫn tệp nộp.")
        print(r"  python -m scripts.check_submission D:\aic-data\submissions\query-demo-kis.csv")
        return 1

    duong_dan = Path(sys.argv[1])

    if not duong_dan.exists():
        print(f"Không thấy tệp: {duong_dan}")
        return 1

    dong = doc_tep_nop(duong_dan)
    cua_so = cua_so_giay()

    print("=" * 72)
    print(f"SOÁT TỆP NỘP: {duong_dan.name}")
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

    # --- tra giây -----------------------------------------------------------
    tra = bang_tra_nguoc()
    moc: list[DongNop] = []
    khong_tra_duoc: list[str] = []

    for i, d in enumerate(dong, start=1):
        video_id = d[0]

        # KIS/QA: một mốc. TRAKE: nhiều mốc, soát từng mốc một.
        cot_so = [c for c in d[1:] if c.strip().lstrip("-").isdigit()]

        for c in cot_so:
            frame_idx = int(c)
            pts_time = tra.get((video_id, frame_idx))

            if pts_time is None:
                khong_tra_duoc.append(f"dòng {i}: {video_id} frame {frame_idx}")
                continue

            moc.append(DongNop(video_id, frame_idx, pts_time, i))

    print(f"Mốc tra được giây  : {len(moc)}")

    if khong_tra_duoc:
        loi.append(
            f"{len(khong_tra_duoc)} mốc không có trong frame_map — "
            "nhiều khả năng đang ghi n thay vì frame_idx"
        )
        for x in khong_tra_duoc[:5]:
            print(f"    {x}")

    # --- cổng thoát số 5 ----------------------------------------------------
    vi_pham = tim_cap_qua_gan(moc, cua_so_giay=cua_so)

    print()
    print(f"CỔNG THOÁT SỐ 5 (cửa sổ {cua_so:.1f} giây)")

    if vi_pham:
        print(f"  TRƯỢT — {len(vi_pham)} cặp cùng video cách nhau dưới {cua_so:.1f} giây")
        for i, j, video_id, khoang_cach in vi_pham[:10]:
            a, b = moc[i], moc[j]
            print(
                f"    {video_id}: frame {a.frame_idx} (giây {a.pts_time:.2f})"
                f" và frame {b.frame_idx} (giây {b.pts_time:.2f})"
                f" — cách {khoang_cach:.2f} giây"
            )
        if len(vi_pham) > 10:
            print(f"    … còn {len(vi_pham) - 10} cặp nữa")
        loi.append(f"trượt cổng 5: {len(vi_pham)} cặp quá gần")
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