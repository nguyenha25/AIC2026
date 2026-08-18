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
"""
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

_TOKEN_RE = re.compile(r"\S+", re.UNICODE)


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
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(
                    video_id UNINDEXED,
                    frame_idx UNINDEXED,
                    n UNINDEXED,
                    text,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                """
            )
            conn.commit()

    def build_index_from_jsonl_dir(self, ocr_dir: Path):
        """Doc cac tep .jsonl trong derived/ocr/ va nap cac frame co text vao FTS index."""
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ocr_fts")

            jsonl_files = list(ocr_dir.glob("*.jsonl"))
            for jfile in jsonl_files:
                with open(jfile, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("text") and data["text"].strip():
                            cursor.execute(
                                "INSERT INTO ocr_fts (video_id, frame_idx, n, text) VALUES (?, ?, ?, ?)",
                                (data["video_id"], data["frame_idx"], data["n"], data["text"]),
                            )
            conn.commit()

    def _run_match(self, match_expr: str, top_k: int, cursor) -> List[Dict[str, Any]]:
        sql = """
            SELECT video_id, frame_idx, text, bm25(ocr_fts) as score, n
            FROM ocr_fts
            WHERE ocr_fts MATCH ?
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
            }
            for row in rows
        ]

    def search_text(
        self, query: str, top_k: int = 100, fallback_to_or: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Tim kiem Full-Text Search.
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