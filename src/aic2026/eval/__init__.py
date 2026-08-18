"""Nhánh eval — Giai đoạn 1, Task 12: tự chấm điểm theo công thức BTC."""
 
from .scorer import (
    K_THRESHOLDS,
    compute_final_score,
    compute_r_at_k,
    compute_r_score,
    r_score_kis,
    r_score_qa,
    r_score_trake,
)
from .io_submissions import load_submission
from .answer_match import is_semantic_match, normalize_text
 
__all__ = [
    "K_THRESHOLDS",
    "compute_final_score",
    "compute_r_at_k",
    "compute_r_score",
    "r_score_kis",
    "r_score_qa",
    "r_score_trake",
    "load_submission",
    "is_semantic_match",
    "normalize_text",
]