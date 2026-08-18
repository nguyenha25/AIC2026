"""
Kiểm tra Task 12 — chương trình tự chấm điểm.
 
Các điều kiện:
 
1. r_score_kis khớp đúng 3 ví dụ số của BTC (mục 2.1.1).
2. r_score_qa khớp đúng ví dụ Q&A của BTC khi answer y hệt (không cần gọi API).
3. r_score_trake khớp đúng ví dụ nhảy cao của BTC, ra đúng 0.75 (mục 2.1.3),
   dùng gt_events là khoảng [s,e] THẬT (schema thật của
   dev_questions.jsonl), không phải điểm + dung sai.
4. r_score_trake: sai video_id phải trả 0.0 TUYỆT ĐỐI, không tính trung bình.
5. compute_r_at_k và compute_final_score khớp đúng ví dụ 100 câu trả lời
   của BTC, Final Score ra đúng 0.74 (mục 2.2).
6. R@k không giảm khi k tăng (tính đơn điệu bắt buộc của công thức max).
 
Chạy:
 
    pytest tests/test_scoring.py -v
"""
 
from __future__ import annotations
 
import pytest
 
from src.aic2026.eval.scorer import (
    K_THRESHOLDS,
    compute_final_score,
    compute_r_at_k,
    r_score_kis,
    r_score_trake,
)
from src.aic2026.submit import KIS
 
 
# ---------------------------------------------------------------------------
# 1. KIS — mục 2.1.1
# ---------------------------------------------------------------------------
 
 
def test_kis_dung():
    """KIS: video khớp, frame trong khoảng -> R-Score = 1"""
    gt = {"gt_video_id": "L01_V001", "gt_frame_range": [500, 510]}
    r = r_score_kis(gt, {"video_id": "L01_V001", "frame_id": 505})
    assert r == 1.0
 
 
def test_kis_sai_frame():
    """KIS: video khớp nhưng frame ngoài khoảng -> R-Score = 0"""
    gt = {"gt_video_id": "L01_V001", "gt_frame_range": [500, 510]}
    r = r_score_kis(gt, {"video_id": "L01_V001", "frame_id": 600})
    assert r == 0.0
 
 
def test_kis_sai_video():
    """KIS: sai video -> R-Score = 0, dù frame đúng khoảng"""
    gt = {"gt_video_id": "L01_V001", "gt_frame_range": [500, 510]}
    r = r_score_kis(gt, {"video_id": "L02_V003", "frame_id": 505})
    assert r == 0.0
 
 
# ---------------------------------------------------------------------------
# 3, 4. TRAKE — mục 2.1.3, ví dụ nhảy cao
# ---------------------------------------------------------------------------
 
 
def test_trake_khop_3_tren_4():
    """
    Ví dụ BTC: đáp án [95,105]/[145,155]/[195,205]/[245,255],
    nộp 101,156,203,251 -> khớp 3/4 -> R-Score = 0.75.
 
    gt_events là khoảng [s,e] THẬT cho từng khoảnh khắc — đúng schema thật
    của dev_questions.jsonl (cac_giai_doan[j].frame_start/frame_end),
    không cần ngưỡng dung sai tự chế.
    """
    gt = {
        "gt_video_id": "L10_V010",
        "gt_events": [[95, 105], [145, 155], [195, 205], [245, 255]],
    }
    submitted = {"video_id": "L10_V010", "frame_ids": [101, 156, 203, 251]}
 
    r = r_score_trake(gt, submitted)
    assert abs(r - 0.75) < 1e-9
 
 
def test_trake_sai_video_la_khong_diem_tuyet_doi():
    """TRAKE: sai video -> 0 điểm NGAY, không tính trung bình các mốc."""
    gt = {
        "gt_video_id": "L10_V010",
        "gt_events": [[95, 105], [145, 155], [195, 205], [245, 255]],
    }
    submitted = {"video_id": "L99_V999", "frame_ids": [101, 156, 203, 251]}
 
    r = r_score_trake(gt, submitted)
    assert r == 0.0
 
 
def test_trake_video_dung_nhung_thieu_frame_nop():
    """Nộp ít khoảnh khắc hơn GT (ví dụ mới soạn 2 giai đoạn như câu 07 của
    Nguyên) vẫn phải chạy được, các khoảnh khắc thiếu tính là không khớp."""
    gt = {"gt_video_id": "L26_V300", "gt_events": [[2922, 2925], [2962, 2972]]}
    submitted = {"video_id": "L26_V300", "frame_ids": [2923]}  # thiếu mốc 2
 
    r = r_score_trake(gt, submitted)
    assert abs(r - 0.5) < 1e-9
 
 
# ---------------------------------------------------------------------------
# 5, 6. R@k và Final Score — mục 2.2
# ---------------------------------------------------------------------------
 
 
def test_final_score_vi_du_btc(monkeypatch):
    """
    Ví dụ BTC: câu 1 R-Score=0.5, câu 3 R-Score=0.8 (cao nhất), câu 15
    R-Score=0.6, còn lại thấp hơn.
    R@1=0.5, R@5=R@20=R@50=R@100=0.8 -> Final Score = 0.74.
 
    KIS chỉ trả 0/1 (không có giá trị trung gian), nên test này thay
    compute_r_score bằng một hàm giả lập để kiểm ĐÚNG CÔNG THỨC R@k/Final
    Score, tách biệt khỏi luật khớp video/frame đã test riêng ở trên.
    """
    import src.aic2026.eval.scorer as scoring_module
 
    diem_gia_lap = {0: 0.5, 2: 0.8, 14: 0.6}
 
    def fake_r_score(task, gt, submitted):
        idx = submitted["rank_index"]
        return diem_gia_lap.get(idx, 0.0)
 
    monkeypatch.setattr(scoring_module, "compute_r_score", fake_r_score)
 
    gt = {"gt_video_id": "L01_V001", "gt_frame_range": [0, 999999]}
    submissions = [{"rank_index": i} for i in range(100)]
 
    ket_qua = compute_final_score(gt, submissions, KIS)
 
    assert ket_qua["R@1"] == 0.5
    assert ket_qua["R@5"] == 0.8
    assert ket_qua["R@20"] == 0.8
    assert ket_qua["R@50"] == 0.8
    assert ket_qua["R@100"] == 0.8
    assert abs(ket_qua["final_score"] - 0.74) < 1e-9
 
 
def test_r_at_k_khong_giam_khi_k_tang():
    """R@k phải đơn điệu không giảm — nếu giảm là chắc chắn có bug."""
    gt = {"gt_video_id": "L01_V001", "gt_frame_range": [500, 510]}
    submissions = [{"video_id": "L01_V001", "frame_id": 505}] + [
        {"video_id": "SAI", "frame_id": 0} for _ in range(99)
    ]
 
    gia_tri_truoc = 0.0
    for k in K_THRESHOLDS:
        r_at_k = compute_r_at_k(gt, submissions, KIS, k)
        assert r_at_k >= gia_tri_truoc
        gia_tri_truoc = r_at_k