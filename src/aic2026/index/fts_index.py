"""
Task 9 -- Kho tra cuu chu (SQLite FTS5).

SUA SO VOI BAN CU: search_text() truoc day boc CA CAU query thanh MOT cum tu
chinh xac ("tu1 tu2 tu3"), doi hoi cac tu phai nam LIEN KE dung thu tu trong
van ban OCR. Vi van ban OCR rat nhieu -- lan (xem file mau L22_V002.jsonl),
kieu boc nay lam giam recall rat nang (2 trong 4 cau test acceptance cu ra 0
ket qua). Ban moi:
  1) Tach query thanh tung tu, moi tu boc quote rieng (van an toan voi
     ky tu dac biet nhu ban cu), noi bang AND -> khong doi hoi lien ke,
     chi doi hoi CA 4-5 tu deu xuat hien dau do trong text.
  2) Neu AND khong ra ket qua nao, tu dong thu lai bang OR de khong bo sot
     hoan toan (vi OCR nhieu, co the mat 1-2 tu trong cum).
Van dung tham so hoa (?, ?) nhu ban cu -- khong co thay doi ve an toan
SQL injection, no da an toan tu truoc.

---------------------------------------------------------------------------
THEM O VIEC 15 (nhanh nhan/asr-vao-fts) -- BANG THU HAI: asr_fts
---------------------------------------------------------------------------
Loi noi da chuyen thanh chu (derived/asr/) nam trong CUNG MOT tep
index/fts/text.sqlite, nhung o BANG RIENG. Ba ly do, khong phai lam cho dep:

  1. HAI DON VI KHAC NHAU. Mot dong OCR la MOT TAM ANH (co n, co frame_idx).
     Mot dong ASR la MOT DOAN LOI NOI (co start/end, trai dai qua nhieu tam
     anh). Nhet chung mot bang thi phai bo trong mot nua so cot cua moi dong.

  2. VIEC 10 GOP THEO NGUON. settings.yaml dat trong so rieng cho tung nhanh
     (ocr 0.6, asr 0.6) va RRF xep hang TRONG TUNG NGUON roi moi gop. Chung
     bang thi khong con tach duoc hai bang xep hang de gop.

  3. NAP LAI KHONG DAM NHAU. build_index_from_jsonl_dir() xoa sach ocr_fts
     truoc khi nap. Neu ASR nam chung bang thi moi lan Nghi chay lai OCR se
     xoa mat toan bo ASR ma khong bao gi ca.

Tokenizer giu nguyen 'unicode61 remove_diacritics 2' -- ASR ra tieng Viet co
dau, mac dinh remove_diacritics 1 lam hong chu hai dau (e, a, u).
"""
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

_TOKEN_RE = re.compile(r"\S+", re.UNICODE)

# Ten bang -- de o mot cho, khong rai chuoi khap file
BANG_OCR = "ocr_fts"
BANG_ASR = "asr_fts"


def _quote_token(tok: str) -> str:
    return '"' + tok.replace('"', '""') + '"'


def _tokenize(query: str) -> List[str]:
    return _TOKEN_RE.findall(query.strip())


def _build_and_query(tokens: List[str]) -> str:
    return " AND ".join(_quote_token(t) for t in tokens)


def _build_or_query(tokens: List[str]) -> str:
    return " OR ".join(_quote_token(t) for t in tokens)


class TextSearchIndex:
    def __init__(self, db_path: Path = Path("index/fts/text.sqlite")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {BANG_OCR} USING fts5(
                    video_id UNINDEXED,
                    frame_idx UNINDEXED,
                    n UNINDEXED,
                    text,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                """
            )
            # Ten cot la start_sec/end_sec chu KHONG phai start/end: "end" la
            # tu khoa SQL, dat ten do thi moi cau lenh deu phai boc dau nhay
            # nguoc, quen mot cho la loi cu phap kho tim. Khoa trong JSONL van
            # la "start"/"end" dung theo docs/schema -- doi ten chi o trong DB.
            cursor.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {BANG_ASR} USING fts5(
                    video_id UNINDEXED,
                    start_sec UNINDEXED,
                    end_sec UNINDEXED,
                    frame_idx_start UNINDEXED,
                    frame_idx_end UNINDEXED,
                    lang UNINDEXED,
                    text,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                """
            )
            conn.commit()

    # -----------------------------------------------------------------
    # NAP DU LIEU
    # -----------------------------------------------------------------

    def build_index_from_jsonl_dir(self, ocr_dir: Path):
        """Doc cac tep .jsonl trong derived/ocr/ va nap cac frame co text vao FTS index.

        CHI dung toi bang ocr_fts. Bang asr_fts khong bi anh huong.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {BANG_OCR}")

            jsonl_files = list(ocr_dir.glob("*.jsonl"))
            for jfile in jsonl_files:
                if jfile.name.startswith("_"):
                    continue          # _nhat_ky.jsonl, _errors.log ... khong phai du lieu
                with open(jfile, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("text") and data["text"].strip():
                            cursor.execute(
                                f"INSERT INTO {BANG_OCR} (video_id, frame_idx, n, text) "
                                "VALUES (?, ?, ?, ?)",
                                (data["video_id"], data["frame_idx"], data["n"], data["text"]),
                            )
            conn.commit()

    def build_asr_index_from_jsonl_dir(self, asr_dir: Path) -> Dict[str, int]:
        """Nap derived/asr/*.jsonl vao bang asr_fts. CHI dung toi bang asr_fts.

        Bo qua dong co text rong -- do la dong danh dau "da chay roi ma khong
        co loi noi", giu trong tep de nguoi doc phan biet voi "chua chay",
        nhung khong co gi de tra cuu.

        Tra ve thong ke {so_tep, so_dong_doc, so_dong_nap} de goi xong con
        biet co that su nap duoc gi khong.

        CA LUOT NAP NAM TRONG MOT GIAO DICH. Gap dong hong giua chung thi
        ValueError bay ra, `with conn` cua sqlite3 rollback, va bang tro ve
        dung nhu truoc khi goi -- KHONG bi xoa sach roi bo do.
        """
        so_tep = 0
        so_dong_doc = 0
        so_dong_nap = 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {BANG_ASR}")

            for jfile in sorted(asr_dir.glob("*.jsonl")):
                if jfile.name.startswith("_"):
                    continue
                so_tep += 1
                with open(jfile, "r", encoding="utf-8") as f:
                    for so_dong, line in enumerate(f, start=1):
                        if not line.strip():
                            continue
                        so_dong_doc += 1
                        try:
                            data = json.loads(line)
                        except ValueError as e:
                            raise ValueError(
                                f"{jfile.name} dong {so_dong}: JSON hong -- {e}"
                            ) from e

                        text = (data.get("text") or "").strip()
                        if not text:
                            continue

                        thieu = [
                            k for k in ("video_id", "start", "end",
                                        "frame_idx_start", "frame_idx_end")
                            if data.get(k) is None
                        ]
                        if thieu:
                            raise ValueError(
                                f"{jfile.name} dong {so_dong}: thieu khoa {thieu}. "
                                "Xem docs/schema/README.md muc 3."
                            )

                        cursor.execute(
                            f"INSERT INTO {BANG_ASR} (video_id, start_sec, end_sec, "
                            "frame_idx_start, frame_idx_end, lang, text) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                data["video_id"],
                                float(data["start"]),
                                float(data["end"]),
                                int(data["frame_idx_start"]),
                                int(data["frame_idx_end"]),
                                data.get("lang", "vi"),
                                text,
                            ),
                        )
                        so_dong_nap += 1
            conn.commit()

        return {"so_tep": so_tep, "so_dong_doc": so_dong_doc, "so_dong_nap": so_dong_nap}

    # -----------------------------------------------------------------
    # TRA CUU
    # -----------------------------------------------------------------

    def _run_match(self, match_expr: str, top_k: int, cursor) -> List[Dict[str, Any]]:
        sql = f"""
            SELECT video_id, frame_idx, text, bm25({BANG_OCR}) as score, n
            FROM {BANG_OCR}
            WHERE {BANG_OCR} MATCH ?
            ORDER BY score ASC
            LIMIT ?
        """
        cursor.execute(sql, (match_expr, top_k))
        rows = cursor.fetchall()
        return [
            {
                "video_id": row[0],
                "frame_idx": int(row[1]),
                "text": row[2],
                "score": float(abs(row[3])),
                "n": int(row[4]),
                "source": "ocr",
            }
            for row in rows
        ]

    def _run_match_asr(self, match_expr: str, top_k: int, cursor) -> List[Dict[str, Any]]:
        sql = f"""
            SELECT video_id, start_sec, end_sec, frame_idx_start, frame_idx_end,
                   lang, text, bm25({BANG_ASR}) as score
            FROM {BANG_ASR}
            WHERE {BANG_ASR} MATCH ?
            ORDER BY score ASC
            LIMIT ?
        """
        cursor.execute(sql, (match_expr, top_k))
        rows = cursor.fetchall()
        return [
            {
                "video_id": row[0],
                "start": float(row[1]),
                "end": float(row[2]),
                "frame_idx_start": int(row[3]),
                "frame_idx_end": int(row[4]),
                "lang": row[5],
                "text": row[6],
                "score": float(abs(row[7])),
                "source": "asr",
            }
            for row in rows
        ]

    def search_text(
        self, query: str, top_k: int = 100, fallback_to_or: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Tim kiem Full-Text Search TREN CHU DOC DUOC TREN ANH (OCR).
        Dau ra uu tien cap dinh danh giao tiep chuan: video_id + frame_idx.

        - Thu truoc: AND giua cac tu (khong doi hoi lien ke, don gian hon
          phrase-match nen chiu duoc OCR nhieu tot hon).
        - Neu 0 ket qua va fallback_to_or=True: thu lai bang OR de khong
          bo sot hoan toan khi 1-2 tu bi OCR sai/thieu.
        """
        tokens = _tokenize(query)
        if not tokens:
            return []

        with self._get_connection() as conn:
            cursor = conn.cursor()

            results = self._run_match(_build_and_query(tokens), top_k, cursor)
            if results or not fallback_to_or or len(tokens) == 1:
                return results

            return self._run_match(_build_or_query(tokens), top_k, cursor)

    def search_asr(
        self, query: str, top_k: int = 100, fallback_to_or: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Tim kiem tren LOI NOI (ASR). Cung chien luoc AND -> OR nhu ben OCR.

        Dau ra la DOAN LOI NOI, chua phai tam anh: moi ket qua co start/end va
        frame_idx_start/frame_idx_end. Viec 10 doi chieu khoang do voi bang
        frame_map de lay ra cac tam anh nam trong khoang, roi moi dua vao RRF
        cung nhanh clip va nhanh ocr. KHONG nop thang frame_idx_start -- so do
        la vi tri khung hinh bat dau CAU NOI, khong chac la tam anh co trong
        kho tra cuu.
        """
        tokens = _tokenize(query)
        if not tokens:
            return []

        with self._get_connection() as conn:
            cursor = conn.cursor()

            results = self._run_match_asr(_build_and_query(tokens), top_k, cursor)
            if results or not fallback_to_or or len(tokens) == 1:
                return results

            return self._run_match_asr(_build_or_query(tokens), top_k, cursor)

    # -----------------------------------------------------------------
    # KIEM TRA
    # -----------------------------------------------------------------

    def thong_ke(self) -> Dict[str, int]:
        """So ban ghi va so video cua tung bang. Dung de soat sau khi nap."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            ket_qua = {}
            for nhan, bang in (("ocr", BANG_OCR), ("asr", BANG_ASR)):
                cursor.execute(f"SELECT COUNT(*), COUNT(DISTINCT video_id) FROM {bang}")
                so_dong, so_video = cursor.fetchone()
                ket_qua[f"so_ban_ghi_{nhan}"] = int(so_dong)
                ket_qua[f"so_video_{nhan}"] = int(so_video)
            return ket_qua