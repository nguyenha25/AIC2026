"""
Việc 10 — SINH MÔ TẢ TỰ ĐỘNG CHO KEYFRAME, LÀM NHÁNH TÌM BẰNG CHỮ THỨ BA.

Vì sao cần
----------
OCR chỉ đọc được chữ CÓ TRÊN HÌNH. ASR chỉ nghe được LỜI NÓI. Câu thuần thị
giác ("người đàn ông cầm ô đứng dưới mưa") thì cả hai nhánh chữ đều câm, và
mọi thứ dồn hết vào một nhánh CLIP. Đó đúng là nhóm câu p1 trượt nhiều nhất.

Caption biến nội dung nhìn thấy thành CHỮ, nên nhánh BM25 cũng góp được tiếng
cho câu thuần thị giác. Nó KHÔNG thay thế CLIP — nó là nguồn thứ hai, độc lập,
sai theo kiểu khác. Đó mới là thứ RRF cần.

BA ĐIỀU PHẢI BIẾT
-----------------
1. CAPTION LÀ TIẾNG ANH. Mô hình BLIP sinh tiếng Anh. Nên `tra_cuu()` tự đổi
   câu hỏi tiếng Việt sang cụm tiếng Anh bằng Việc 5 TRƯỚC khi tra. Đưa thẳng
   câu tiếng Việt vào đây là bảo đảm 0 kết quả.

2. TẤT ĐỊNH. Sinh bằng beam search (num_beams=3, do_sample=False), KHÔNG lấy
   mẫu ngẫu nhiên. Chạy lại trên cùng một ảnh phải ra đúng một câu — nếu không
   thì mỗi lần dựng lại kho là một kho khác và không so điểm được với gì.

3. CHẠY LẠI ĐƯỢC. Mỗi video một tệp JSONL trong derived/captions/. Đứt giữa
   chừng thì chạy lại chỉ làm tiếp video còn thiếu. 177.321 ảnh trên CPU là
   nhiều ngày — không có cơ chế này thì không ai chạy nổi.

CHI PHÍ THẬT — ĐỌC TRƯỚC KHI BẮT ĐẦU
------------------------------------
BLIP-base trên CPU khoảng 0,4-1,0 giây một ảnh. 177.321 ảnh ≈ 20-50 giờ.
Trên GPU rời (theo lô 32 ảnh) khoảng 1-2 giờ. Chạy CPU thì làm theo shard,
và làm SAU khi Việc 4 với Việc 5 đã đo xong — đúng thứ tự checklist ghi.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

BANG_CAPTION = "caption_fts"

MO_HINH_MAC_DINH = "Salesforce/blip-image-captioning-base"
SO_BEAM = 3
DAI_TOI_DA = 40


# ---------------------------------------------------------------------------
# Sinh caption
# ---------------------------------------------------------------------------

class BoSinhCaption:
    """Nạp mô hình một lần, sinh caption theo lô."""

    def __init__(
        self,
        ten_mo_hinh: str = MO_HINH_MAC_DINH,
        thiet_bi: str | None = None,
        kich_thuoc_lo: int = 8,
    ):
        self.ten_mo_hinh = ten_mo_hinh
        self._thiet_bi_yeu_cau = thiet_bi
        self.kich_thuoc_lo = kich_thuoc_lo
        self._processor = None
        self._model = None
        self._thiet_bi = None

    def _nap(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self._thiet_bi = self._thiet_bi_yeu_cau or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._processor = BlipProcessor.from_pretrained(self.ten_mo_hinh)
        self._model = BlipForConditionalGeneration.from_pretrained(self.ten_mo_hinh)
        self._model.to(self._thiet_bi)
        self._model.eval()

    def sinh(self, duong_dan_anh: list[Path]) -> list[str]:
        """Sinh caption cho một lô ảnh. Ảnh hỏng trả về chuỗi rỗng."""
        import torch
        from PIL import Image

        self._nap()

        anh: list[Any] = []
        chi_so_hop_le: list[int] = []
        for i, p in enumerate(duong_dan_anh):
            try:
                with Image.open(p) as im:
                    anh.append(im.convert("RGB").copy())
                chi_so_hop_le.append(i)
            except Exception:
                continue

        ra = [""] * len(duong_dan_anh)
        if not anh:
            return ra

        dau_vao = self._processor(images=anh, return_tensors="pt").to(self._thiet_bi)
        with torch.no_grad():
            token = self._model.generate(
                **dau_vao,
                max_new_tokens=DAI_TOI_DA,
                num_beams=SO_BEAM,
                do_sample=False,          # BẮT BUỘC: không lấy mẫu -> tất định
            )
        van_ban = self._processor.batch_decode(token, skip_special_tokens=True)

        for vi_tri, chu in zip(chi_so_hop_le, van_ban):
            ra[vi_tri] = chu.strip()
        return ra


def sinh_cho_video(
    video_id: str,
    bo_sinh: BoSinhCaption,
    lam_lai: bool = False,
    in_tien_do: bool = True,
) -> dict[str, int]:
    """Sinh caption cho toàn bộ keyframe của MỘT video -> derived/captions/<v>.jsonl.

    Ghi qua tệp .partial rồi mới đổi tên: đứt điện giữa chừng không để lại
    tệp JSONL cụt mà lần sau tưởng là đã xong.
    """
    try:
        from ..kf_index import hang_sang_moc, so_hang
    except ImportError as loi:
        raise ImportError(
            "Việc 10 cần src/aic2026/kf_index.py của Việc 3c (gói 02 — Thi). "
            "Trộn gói 02 trước rồi chạy lại."
        ) from loi

    from ..paths import captions_file, keyframe_image

    dich = captions_file(video_id)

    if dich.exists() and not lam_lai:
        so_dong = sum(1 for _ in dich.open("r", encoding="utf-8"))
        if so_dong >= so_hang(video_id):
            return {"video_id": video_id, "bo_qua": 1, "so_dong": so_dong}

    tong = so_hang(video_id)
    dich.parent.mkdir(parents=True, exist_ok=True)
    tam = dich.with_suffix(".partial")

    so_ghi = so_thieu_anh = 0

    with tam.open("w", encoding="utf-8") as f:
        for dau_lo in range(0, tong, bo_sinh.kich_thuoc_lo):
            moc = [
                hang_sang_moc(video_id, i)
                for i in range(dau_lo, min(dau_lo + bo_sinh.kich_thuoc_lo, tong))
            ]
            duong_dan = [keyframe_image(video_id, m.n) for m in moc]

            co_anh = [p.exists() for p in duong_dan]
            so_thieu_anh += co_anh.count(False)

            caption = bo_sinh.sinh(duong_dan)

            for m, p, c, co in zip(moc, duong_dan, caption, co_anh):
                if not co or not c:
                    continue
                f.write(
                    json.dumps(
                        {
                            "video_id": m.video_id,
                            "n": m.n,
                            "frame_idx": m.frame_idx,
                            "pts_time": m.pts_time,
                            "caption": c,
                            "engine": bo_sinh.ten_mo_hinh,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                so_ghi += 1

            if in_tien_do:
                print(
                    f"    {min(dau_lo + bo_sinh.kich_thuoc_lo, tong)}/{tong}",
                    end="\r",
                    flush=True,
                )

    tam.replace(dich)
    return {
        "video_id": video_id,
        "bo_qua": 0,
        "so_dong": so_ghi,
        "so_thieu_anh": so_thieu_anh,
    }


# ---------------------------------------------------------------------------
# Kho tra cứu
# ---------------------------------------------------------------------------

class CaptionSearchIndex:
    """Bảng caption_fts trong cùng tệp index/fts/text.sqlite."""

    def __init__(self, db_path: Path | None = None, tu_dich_cau_hoi: bool = True):
        if db_path is None:
            from ..paths import FTS_DIR

            db_path = FTS_DIR / "text.sqlite"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.tu_dich_cau_hoi = tu_dich_cau_hoi
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {BANG_CAPTION} USING fts5(
                    video_id UNINDEXED,
                    n UNINDEXED,
                    frame_idx UNINDEXED,
                    pts_time UNINDEXED,
                    caption,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                """
            )
            conn.commit()

    def nap_tu_thu_muc(
        self,
        captions_dir: Path | None = None,
        video_ids: Iterable[str] | None = None,
        xoa_truoc: bool = True,
    ) -> dict[str, int]:
        from ..paths import CAPTIONS_DIR

        captions_dir = captions_dir or CAPTIONS_DIR
        tep = sorted(captions_dir.glob("*.jsonl"))
        if video_ids:
            giu = set(video_ids)
            tep = [t for t in tep if t.stem in giu]

        so_dong = 0
        with self._conn() as conn:
            cur = conn.cursor()
            if xoa_truoc:
                if video_ids:
                    cur.executemany(
                        f"DELETE FROM {BANG_CAPTION} WHERE video_id = ?",
                        [(v,) for v in video_ids],
                    )
                else:
                    cur.execute(f"DELETE FROM {BANG_CAPTION}")

            for t in tep:
                lo = []
                with t.open("r", encoding="utf-8") as f:
                    for dong in f:
                        dong = dong.strip()
                        if not dong:
                            continue
                        try:
                            d = json.loads(dong)
                        except json.JSONDecodeError:
                            continue
                        if not d.get("caption"):
                            continue
                        lo.append(
                            (
                                d["video_id"],
                                int(d["n"]),
                                int(d["frame_idx"]),
                                float(d["pts_time"]),
                                d["caption"],
                            )
                        )
                if lo:
                    cur.executemany(
                        f"INSERT INTO {BANG_CAPTION} "
                        f"(video_id, n, frame_idx, pts_time, caption) "
                        f"VALUES (?, ?, ?, ?, ?)",
                        lo,
                    )
                    so_dong += len(lo)
            conn.commit()

        return {"so_tep": len(tep), "so_dong_nap": so_dong}

    def tra_cuu(self, cau_hoi: str, top_k: int = 500) -> list[dict[str, Any]]:
        """Câu hỏi TIẾNG VIỆT -> cụm tiếng Anh -> BM25 trên caption.

        Caption trong bảng là tiếng Anh, nên phải đổi câu hỏi. Đây là chỗ
        DUY NHẤT trong các nhánh chữ cần bản dịch — ba nhánh kia (ocr, asr,
        object) đọc dữ liệu tiếng Việt hoặc nhãn cố định.
        """
        cau_tra = cau_hoi
        if self.tu_dich_cau_hoi:
            try:
                from ..query_expand import mo_rong

                cau_tra = mo_rong(cau_hoi).cum_chinh
            except Exception:
                cau_tra = cau_hoi

        tu = [t for t in cau_tra.split() if t]
        if not tu:
            return []

        boc = ['"' + t.replace('"', '""') + '"' for t in tu]

        with self._conn() as conn:
            cur = conn.cursor()
            ra = self._chay(cur, " AND ".join(boc), top_k)
            if not ra and len(boc) > 1:
                ra = self._chay(cur, " OR ".join(boc), top_k)
        return ra

    def _chay(self, cur, bieu_thuc: str, top_k: int) -> list[dict[str, Any]]:
        cur.execute(
            f"""
            SELECT video_id, n, frame_idx, pts_time, caption,
                   bm25({BANG_CAPTION}) AS diem
            FROM {BANG_CAPTION}
            WHERE {BANG_CAPTION} MATCH ?
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
                "text": r[4],
                "score": float(abs(r[5])),
                "source": "caption",
            }
            for r in cur.fetchall()
        ]

    def thong_ke(self) -> dict[str, int]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT video_id) FROM {BANG_CAPTION}"
            )
            so_dong, so_video = cur.fetchone()
        return {"so_ban_ghi_caption": int(so_dong), "so_video_caption": int(so_video)}
