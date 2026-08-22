"""
Việc 3 — KHO TRA NGƯỢC VẬT THỂ (bảng objects_fts).

Vì sao có tệp này
-----------------
BTC phát sẵn raw/objects/<video>/<n>.json — kết quả Faster R-CNN cho TỪNG
keyframe. Nhóm chưa dùng lần nào. Nhiều câu vòng p1 tả vật thể rất cụ thể
("ba nón đỏ", "bộ trống và piano", "ba ổ bánh mì"), lọc theo vật thể thu hẹp
177.321 keyframe xuống vài trăm TRƯỚC khi tính CLIP.

Ba quyết định cần biết trước khi sửa
------------------------------------
1. NHÂN BẢN NHÃN THEO SỐ LƯỢNG. Một khung có 3 người thì cột `nhan` ghi
   "Person Person Person". BM25 tính tần suất từ, nên khung có 3 nón xếp trên
   khung có 1 nón mà không cần thêm mã lọc số lượng. Số đếm thật vẫn lưu
   nguyên ở cột `dem_json` để lọc cứng khi cần.

2. NGƯỠNG 0.4 LÀ CỦA BTC, KHÔNG PHẢI CỦA NHÓM. Notebook baseline BTC lọc
   detection_scores > 0.4. Giữ đúng con số đó để kết quả so được với baseline.

3. TOẠ ĐỘ HỘP LÀ [ymin, xmin, ymax, xmax] ĐÃ CHUẨN HOÁ — khác thứ tự thông
   thường (x trước y). Tệp này chỉ dùng hộp để tính vị trí thô (trái/phải/
   trên/dưới), nhưng ai đọc detection_boxes ở chỗ khác phải nhớ điều này.

Bảng nằm chung tệp index/fts/text.sqlite với ocr_fts và asr_fts, nhưng là
BẢNG RIÊNG — cùng lý do đã ghi ở fts_index.py: khác đơn vị dữ liệu, khác
trọng số RRF, và nạp lại nhánh này không được xoá nhánh kia.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

BANG_OBJECT = "objects_fts"

# Ngưỡng của notebook baseline BTC. Đổi số này là mất khả năng so với baseline.
NGUONG_DIEM_MAC_DINH = 0.4

_SO_TRONG_TEN = re.compile(r"(\d+)")


# ---------------------------------------------------------------------------
# Bảng ánh xạ tiếng Việt -> nhãn tiếng Anh
# ---------------------------------------------------------------------------

def chuan_hoa(s: str) -> str:
    """Hạ chữ thường bằng PYTHON, GIỮ NGUYÊN DẤU.

    Hai điểm, cả hai đều đã cắn nhóm một lần:

    1. Hạ chữ thường phải làm bằng Python. Bài học vòng p1: SQLite chỉ đổi
       hoa/thường cho chữ ASCII nên 'Đèo' không khớp 'đèo'. Python thì đúng.

    2. TUYỆT ĐỐI KHÔNG bỏ dấu ở đây. Bỏ dấu làm nhập nhằng hàng loạt cặp từ
       khác hẳn nghĩa:
           trống (Drum)  == trong (giới từ)
           tím  (purple) == tìm  (tìm kiếm)
           đèn  (Lamp)   == đen  (màu đen)
           cầm  (holding)== cam  (quả cam)
       Bộ test đã bắt đúng chuyện này: câu "một cảnh không có vật thể nào
       trong bảng" từng khớp nhầm nhãn Drum vì chữ "trong".

       Bảng ocr_fts vẫn dùng remove_diacritics 2 — đúng, vì OCR hay mất dấu.
       Nhưng ĐỀ THI thì gõ đủ dấu, nên tra bảng nhãn phải so có dấu.
    """
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


class BangNhan:
    """Từ tiếng Việt trong đề -> danh sách nhãn tiếng Anh của Faster R-CNN.

    `bo_qua` là danh sách cụm PHẢI GẠCH ĐI TRƯỚC khi tra nhãn. Không có nó thì
    hàng loạt cụm ghép bị hiểu sai vì chứa một từ có trong bảng:

        "cảnh quay TRONG NHÀ"  -> khớp "nhà" -> House      (sai: bối cảnh, không phải vật thể)
        "một NHÀ BÁO"          -> khớp "nhà" -> House      (sai: nghề nghiệp)
        "KÉO DÀI ba tiếng"     -> khớp "kéo" -> Scissors   (sai: động từ)
        "KÍNH THƯA quý vị"     -> khớp "kính" -> Glasses   (sai: kính ngữ)
        "CHÍNH SÁCH mới"       -> khớp "sách" -> Book      (sai)
        "CỬA HÀNG tiện lợi"    -> khớp "cửa" -> Door       (sai)

    Ranh giới từ KHÔNG cứu được những trường hợp này, vì "nhà" trong "trong
    nhà" đúng là một từ trọn vẹn. Chỉ có cách gạch cụm đi trước.
    """

    def __init__(self, anh_xa: dict[str, list[str]], bo_qua: list[str] | None = None):
        self._bo_qua = [
            re.compile(rf"(?<!\w){re.escape(chuan_hoa(str(c)))}(?!\w)")
            for c in (bo_qua or [])
            if str(c).strip()
        ]
        self._map: dict[str, list[str]] = {}
        for tu_viet, nhan_anh in anh_xa.items():
            khoa = chuan_hoa(str(tu_viet))
            ds = [nhan_anh] if isinstance(nhan_anh, str) else list(nhan_anh)
            self._map.setdefault(khoa, []).extend(str(x) for x in ds)

        # cụm dài khớp trước cụm ngắn: "bánh mì" phải thắng "bánh"
        self._khoa_theo_do_dai = sorted(
            self._map, key=lambda k: (len(k.split()), len(k)), reverse=True
        )
        self._mau = {
            k: re.compile(rf"(?<!\w){re.escape(k)}(?!\w)") for k in self._map
        }

    @classmethod
    def nap(cls, duong_dan: Path | None = None) -> "BangNhan":
        from ..paths import CONFIG_DIR

        duong_dan = duong_dan or CONFIG_DIR / "nhan_vat_the.yaml"
        if not duong_dan.exists():
            return cls({})
        with duong_dan.open("r", encoding="utf-8") as f:
            noi_dung = yaml.safe_load(f) or {}
        return cls(noi_dung.get("anh_xa", {}), noi_dung.get("bo_qua", []))

    def tim_nhan(self, cau_hoi: str) -> list[str]:
        """Rút mọi nhãn tiếng Anh mà câu hỏi tiếng Việt có nhắc tới.

        Trả về danh sách đã khử trùng, giữ thứ tự xuất hiện. Rỗng nghĩa là
        câu hỏi không nhắc vật thể nào có trong bảng — phía gọi nên BỎ QUA
        nhánh vật thể, đừng tra bừa.
        """
        van_ban = chuan_hoa(cau_hoi)

        # Gạch các cụm gây hiểu nhầm TRƯỚC, không tra nhãn trên chúng.
        for mau in self._bo_qua:
            van_ban = mau.sub(" | ", van_ban)

        ra: list[str] = []

        for khoa in self._khoa_theo_do_dai:
            if not khoa:
                continue
            # So theo RANH GIỚI TỪ, không phải chuỗi con: không có ranh giới
            # thì "cam" trong "camera" cũng khớp thành quả cam.
            if not self._mau[khoa].search(van_ban):
                continue

            for nhan in self._map[khoa]:
                if nhan not in ra:
                    ra.append(nhan)

            # ĂN LUÔN ĐOẠN ĐÃ KHỚP. Nếu không, "đàn piano" khớp xong thì "đàn"
            # cũng khớp tiếp trên cùng đoạn chữ đó và kéo theo Guitar,
            # Musical instrument. Với truy vấn AND, mỗi nhãn thừa là một điều
            # kiện không khung nào thoả -> tra_theo_nhan phải lùi về OR và
            # chất lượng xếp hạng tụt mà không có dấu hiệu gì.
            van_ban = self._mau[khoa].sub(" | ", van_ban)

        return ra

    def __len__(self) -> int:
        return len(self._map)


# ---------------------------------------------------------------------------
# Đọc một tệp objects
# ---------------------------------------------------------------------------

def doc_mot_tep(
    duong_dan: Path,
    nguong: float = NGUONG_DIEM_MAC_DINH,
) -> tuple[list[str], dict[str, int]]:
    """Đọc objects/<video>/<n>.json -> (danh sách nhãn đã lọc, số đếm mỗi nhãn).

    Tệp hỏng hoặc thiếu khoá trả về ([], {}) chứ không ném lỗi: 177 nghìn tệp
    thì một tệp hỏng không được làm dừng cả mẻ nạp.
    """
    try:
        with duong_dan.open("r", encoding="utf-8") as f:
            du_lieu = json.load(f)
    except (OSError, json.JSONDecodeError):
        return [], {}

    nhan = du_lieu.get("detection_class_entities") or []
    diem = du_lieu.get("detection_scores") or []

    giu: list[str] = []
    for i, ten in enumerate(nhan):
        try:
            d = float(diem[i]) if i < len(diem) else 0.0
        except (TypeError, ValueError):
            d = 0.0
        if d > nguong:
            giu.append(str(ten).strip())

    return giu, dict(Counter(giu))


def _n_tu_ten_tep(duong_dan: Path) -> int | None:
    """0047.json -> 47. Chịu được cả '47.json' lẫn '0047.json'."""
    khop = _SO_TRONG_TEN.search(duong_dan.stem)
    return int(khop.group(1)) if khop else None


# ---------------------------------------------------------------------------
# Kho tra cứu
# ---------------------------------------------------------------------------

class ObjectSearchIndex:
    """Kho FTS5 cho nhãn vật thể. Dùng chung tệp sqlite với ocr_fts/asr_fts."""

    def __init__(self, db_path: Path | None = None, bang_nhan: BangNhan | None = None):
        if db_path is None:
            from ..paths import FTS_DIR

            db_path = FTS_DIR / "text.sqlite"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.bang_nhan = bang_nhan if bang_nhan is not None else BangNhan.nap()
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {BANG_OBJECT} USING fts5(
                    video_id UNINDEXED,
                    n UNINDEXED,
                    frame_idx UNINDEXED,
                    pts_time UNINDEXED,
                    dem_json UNINDEXED,
                    nhan,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                """
            )
            conn.commit()

    # -- nạp ---------------------------------------------------------------

    def nap_tu_thu_muc(
        self,
        objects_dir: Path | None = None,
        video_ids: Iterable[str] | None = None,
        nguong: float = NGUONG_DIEM_MAC_DINH,
        xoa_truoc: bool = True,
        in_tien_do: bool = True,
    ) -> dict[str, int]:
        """Quét raw/objects/ và nạp vào bảng objects_fts.

        Mỗi keyframe một dòng. Keyframe KHÔNG có vật thể nào qua ngưỡng thì
        bỏ hẳn, không nạp dòng rỗng — dòng rỗng chỉ làm phình bảng chứ không
        bao giờ khớp truy vấn nào.

        video_ids=None nghĩa là nạp tất cả video tìm thấy trên máy. Máy chỉ
        tải một phần kho thì truyền danh sách shard của mình vào.
        """
        from ..frame_map import lookup
        from ..paths import OBJECTS_DIR, list_video_ids, resolve

        objects_dir = objects_dir or OBJECTS_DIR
        danh_sach = list(video_ids) if video_ids else list_video_ids(objects_dir)

        if not danh_sach:
            raise FileNotFoundError(
                f"Không thấy video nào trong {objects_dir}. "
                "Đã giải nén objects vào raw/objects/ chưa?"
            )

        so_tep = so_dong_nap = so_bo_qua = so_khong_tra_duoc = 0

        with self._conn() as conn:
            cur = conn.cursor()
            if xoa_truoc:
                if video_ids:
                    cur.executemany(
                        f"DELETE FROM {BANG_OBJECT} WHERE video_id = ?",
                        [(v,) for v in danh_sach],
                    )
                else:
                    cur.execute(f"DELETE FROM {BANG_OBJECT}")

            for thu_tu, video_id in enumerate(danh_sach, start=1):
                goc = resolve(objects_dir, video_id) or objects_dir / video_id
                if not goc.is_dir():
                    continue

                lo: list[tuple] = []
                for tep in sorted(goc.glob("*.json")):
                    so_tep += 1
                    n = _n_tu_ten_tep(tep)
                    if n is None:
                        continue

                    nhan, dem = doc_mot_tep(tep, nguong)
                    if not nhan:
                        so_bo_qua += 1
                        continue

                    try:
                        frame_idx, pts_time = lookup(video_id, n)
                    except Exception:
                        # Không tra được n -> frame_idx thì KHÔNG nạp. Nạp vào
                        # với frame_idx đoán bừa là gieo đúng loại lỗi âm thầm
                        # mà cả dự án đang tránh.
                        so_khong_tra_duoc += 1
                        continue

                    lo.append(
                        (
                            video_id,
                            n,
                            frame_idx,
                            pts_time,
                            json.dumps(dem, ensure_ascii=False),
                            " ".join(nhan),   # đã nhân bản theo số lượng
                        )
                    )

                if lo:
                    cur.executemany(
                        f"INSERT INTO {BANG_OBJECT} "
                        f"(video_id, n, frame_idx, pts_time, dem_json, nhan) "
                        f"VALUES (?, ?, ?, ?, ?, ?)",
                        lo,
                    )
                    so_dong_nap += len(lo)

                if in_tien_do and (thu_tu % 25 == 0 or thu_tu == len(danh_sach)):
                    print(
                        f"  {thu_tu}/{len(danh_sach)} video — "
                        f"{so_dong_nap:,} dòng đã nạp",
                        flush=True,
                    )

            conn.commit()

        return {
            "so_video": len(danh_sach),
            "so_tep_doc": so_tep,
            "so_dong_nap": so_dong_nap,
            "so_khung_khong_co_vat_the": so_bo_qua,
            "so_khung_khong_tra_duoc_frame_idx": so_khong_tra_duoc,
        }

    # -- tra cứu -----------------------------------------------------------

    def tra_theo_nhan(
        self,
        nhan: list[str],
        top_k: int = 500,
        bat_buoc_du: bool = True,
    ) -> list[dict[str, Any]]:
        """Tra thẳng bằng danh sách nhãn TIẾNG ANH.

        bat_buoc_du=True: nối AND, khung phải có ĐỦ mọi nhãn. Câu "bộ trống
        và piano" cần cả hai thì AND đúng hơn OR rất nhiều.
        Không ra kết quả nào thì tự hạ xuống OR — thà xếp hạng kém còn hơn
        trả rỗng.
        """
        if not nhan:
            return []

        boc = ['"' + t.replace('"', '""') + '"' for t in nhan]
        bieu_thuc = (" AND " if bat_buoc_du else " OR ").join(boc)

        with self._conn() as conn:
            cur = conn.cursor()
            ra = self._chay(cur, bieu_thuc, top_k)
            if not ra and bat_buoc_du and len(boc) > 1:
                ra = self._chay(cur, " OR ".join(boc), top_k)
        return ra

    def _chay(self, cur, bieu_thuc: str, top_k: int) -> list[dict[str, Any]]:
        cur.execute(
            f"""
            SELECT video_id, n, frame_idx, pts_time, dem_json, nhan,
                   bm25({BANG_OBJECT}) AS diem
            FROM {BANG_OBJECT}
            WHERE {BANG_OBJECT} MATCH ?
            ORDER BY diem ASC
            LIMIT ?
            """,
            (bieu_thuc, top_k),
        )
        return [
            {
                "video_id": r[0],
                "n": int(r[1]),
                "frame_idx": int(r[2]),
                "pts_time": float(r[3]),
                "dem": json.loads(r[4]) if r[4] else {},
                "nhan": r[5],
                "score": float(abs(r[6])),
                "source": "object",
            }
            for r in cur.fetchall()
        ]

    def tra_bang_cau_viet(self, cau_hoi: str, top_k: int = 500) -> list[dict[str, Any]]:
        """Câu hỏi TIẾNG VIỆT -> nhãn tiếng Anh -> tra bảng.

        Câu không nhắc vật thể nào trong bảng ánh xạ trả về [] — đó là câu
        trả lời ĐÚNG, không phải lỗi. Nhánh vật thể chỉ nên đóng góp khi
        câu hỏi thật sự nói về vật thể.
        """
        nhan = self.bang_nhan.tim_nhan(cau_hoi)
        return self.tra_theo_nhan(nhan, top_k=top_k)

    # -- soát lại ----------------------------------------------------------

    def thong_ke(self) -> dict[str, int]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT video_id) FROM {BANG_OBJECT}"
            )
            so_dong, so_video = cur.fetchone()
        return {"so_ban_ghi_object": int(so_dong), "so_video_object": int(so_video)}
