"""
Đọc tệp nộp (submissions/query-<id>-<dạng>.csv) để chấm điểm — Task 12.
 
Định dạng ĐÃ CHỐT ở docs/schema/README.md mục 6 và
src/aic2026/submit/formatter.py — không đoán lại ở đây, chỉ ĐỌC.
 
Dùng lại submission_filename() từ submit/formatter.py để tính đường dẫn tệp,
KHÔNG tự đặt quy ước tên mới — sửa quy ước tên chỉ sửa một chỗ ở formatter.py.
"""
 
from __future__ import annotations
 
import csv
from pathlib import Path
 
from ..submit import KIS, QA, TRAKE, submission_filename
 
 
def read_submission(path: Path) -> list[dict]:
    """
    Đọc một tệp submission thô, giữ ĐÚNG THỨ TỰ DÒNG (thứ tự = thứ hạng).
 
    Không biết trước dạng truy vấn ở tệp này — trả về list các list ô,
    dùng parse_kis/parse_qa/parse_trake bên dưới để ép kiểu theo dạng.
    """
    if not path.exists():
        raise FileNotFoundError(f"Không thấy tệp nộp: {path}")
 
    with path.open("r", encoding="utf-8", newline="") as f:
        return [row for row in csv.reader(f) if row]
 
 
def parse_kis(rows: list[list[str]]) -> list[dict]:
    """<video_id>,<frame_id> -> [{"video_id":..., "frame_id":...}, ...]"""
    return [{"video_id": r[0], "frame_id": int(r[1])} for r in rows]
 
 
def parse_qa(rows: list[list[str]]) -> list[dict]:
    """<video_id>,<frame_id>,<answer> -> thêm khoá "answer"."""
    return [{"video_id": r[0], "frame_id": int(r[1]), "answer": r[2]} for r in rows]
 
 
def parse_trake(rows: list[list[str]]) -> list[dict]:
    """<video_id>,<frame_id_1>,...,<frame_id_n> -> khoá "frame_ids" (list)."""
    return [{"video_id": r[0], "frame_ids": [int(x) for x in r[1:]]} for r in rows]
 
 
_PARSERS = {KIS: parse_kis, QA: parse_qa, TRAKE: parse_trake}
 
 
def load_submission(submissions_dir: Path, query_id: str, task: str) -> list[dict]:
    """
    Đọc + parse tệp nộp của một truy vấn, theo đúng tên tệp mà nhánh Nộp
    (submit/formatter.py) đã dùng để ghi.
 
    Trả về [] nếu tệp không tồn tại (câu đó chưa được Task 4 chạy) — KHÔNG
    raise, để run_scoring.py tự quyết định coi đây là 0 điểm hay bỏ qua.
    """
    if task not in _PARSERS:
        raise ValueError(f"Dạng truy vấn lạ: {task}")
 
    path = submissions_dir / submission_filename(query_id, task)
    if not path.exists():
        return []
 
    rows = read_submission(path)
    return _PARSERS[task](rows)