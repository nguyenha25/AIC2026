"""
Chấm điểm theo đúng công thức BTC — Task 12
(Thong_tin_vong_So_tuyen_AIC2026.pdf, mục 2).
 
GT (ground truth) đọc từ dev_question.jsonl (schema THẬT — không phải mẫu ở docs/schema/README.md, xem run_scoring.py phần
build_gt() để biết cách ánh xạ):
 
    kis/qa : {"gt_video_id":..., "gt_frame_range":[s,e]}
    qa     : thêm {"gt_answer": "..."}
    trake  : {"gt_video_id":..., "gt_events":[[s1,e1],[s2,e2],...]}
             (mỗi khoảnh khắc có ĐÚNG khoảng [s,e] thật, lấy từ
             cac_giai_doan[j].frame_start/frame_end trong file gốc)
 
Submission đọc bằng io_submissions.load_submission(), trả về:
 
    kis/qa : {"video_id":..., "frame_id":..., "answer": ... | None}
    trake  : {"video_id":..., "frame_ids": [...]}
 
So khớp ngữ nghĩa Q&A (answer_match.is_semantic_match) chạy HOÀN TOÀN LOCAL
bằng mô hình embedding câu (sentence-transformers), KHÔNG gọi API trả phí,
KHÔNG cần cache — mô hình chạy tức thì trên máy sau lần tải đầu tiên.
"""
 
from __future__ import annotations
 
from ..submit import KIS, QA, TRAKE
from .answer_match import is_semantic_match
 
K_THRESHOLDS = (1, 5, 20, 50, 100)
 
 
# ---------------------------------------------------------------------------
# R-Score cho từng dạng truy vấn (mục 2.1)
# ---------------------------------------------------------------------------
 
 
def r_score_kis(gt: dict, submitted: dict) -> float:
    """KIS (mục 2.1.1): đúng nếu video khớp VÀ frame_id trong [s, e]."""
    if submitted["video_id"] != gt["gt_video_id"]:
        return 0.0
    s, e = gt["gt_frame_range"]
    return 1.0 if s <= submitted["frame_id"] <= e else 0.0
 
 
def r_score_qa(gt: dict, submitted: dict) -> float:
    """
    Q&A (mục 2.1.2): đúng nếu video khớp, frame trong [s, e], VÀ answer
    khớp ngữ nghĩa.
 
    Thứ tự kiểm tra: video -> frame range -> answer. Chỉ so khớp ngữ nghĩa
    (chạy mô hình embedding) ở bước CUỐI, khi hai điều kiện trước đã đúng —
    tránh tính toán lãng phí cho các dòng chắc chắn sai video/frame (đa số
    trong 100 dòng).
    """
    if submitted["video_id"] != gt["gt_video_id"]:
        return 0.0
    s, e = gt["gt_frame_range"]
    if not (s <= submitted["frame_id"] <= e):
        return 0.0
    if not is_semantic_match(gt["gt_answer"], submitted.get("answer") or ""):
        return 0.0
    return 1.0
 
 
def r_score_trake(gt: dict, submitted: dict) -> float:
    """
    TRAKE (mục 2.1.3): sai video -> 0 điểm NGAY, không tính trung bình.
    Đúng video -> tỉ lệ khoảnh khắc khớp / tổng N khoảnh khắc.
 
    gt["gt_events"] là list [s_j, e_j] — khoảng khung hình ĐÚNG thật của
    từng khoảnh khắc. Một khung hình nộp submitted["frame_ids"][j] được
    coi là khớp khoảnh khắc thứ j nếu s_j <= frame_ids[j] <= e_j — đúng
    nguyên văn mục 2.1.3.
    """
    if submitted["video_id"] != gt["gt_video_id"]:
        return 0.0
 
    events = gt["gt_events"]
    submitted_ids = submitted["frame_ids"]
    n = len(events)
    if n == 0:
        return 0.0
 
    khop = 0
    for j, (s, e) in enumerate(events):
        if j < len(submitted_ids) and s <= submitted_ids[j] <= e:
            khop += 1
 
    return khop / n
 
 
_R_SCORE_FNS = {KIS: r_score_kis, QA: r_score_qa, TRAKE: r_score_trake}
 
 
def compute_r_score(task: str, gt: dict, submitted: dict) -> float:
    """Hàm dispatch theo task ("kis"/"qa"/"trake" — dùng hằng số KIS/QA/TRAKE)."""
    if task not in _R_SCORE_FNS:
        raise ValueError(f"Dạng truy vấn lạ: {task}")
    return _R_SCORE_FNS[task](gt, submitted)
 
 
# ---------------------------------------------------------------------------
# R@k và Final Score (mục 2.2)
# ---------------------------------------------------------------------------
 
 
def compute_r_at_k(gt: dict, submissions: list[dict], task: str, k: int) -> float:
    """R@k = R-Score cao nhất trong k dòng đầu tiên (đã xếp hạng)."""
    top_k = submissions[:k]
    if not top_k:
        return 0.0
    return max(compute_r_score(task, gt, s) for s in top_k)
 
 
def compute_final_score(gt: dict, submissions: list[dict], task: str) -> dict:
    """Final Score = trung bình cộng 5 giá trị R@k, k thuộc {1,5,20,50,100}."""
    ket_qua = {f"R@{k}": compute_r_at_k(gt, submissions, task, k) for k in K_THRESHOLDS}
    ket_qua["final_score"] = sum(ket_qua.values()) / len(K_THRESHOLDS)
    return ket_qua