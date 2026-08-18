"""
TASK 12 — chạy chương trình tự chấm điểm theo đúng công thức BTC.
 
    python -m scripts.run_scoring
 
Đọc dev_question.jsonl (schema THẬT — xem build_gt() bên
dưới, KHÁC với mẫu ở docs/schema/README.md), với mỗi câu tìm tệp nộp tương
ứng trong submissions/ (đặt tên theo submission_filename() của
submit/formatter.py), chấm bằng src/aic2026/eval/scoring.py, ghi kết quả chi
tiết ra derived/eval/scoring_results.csv và in ĐIỂM NỀN (baseline_score).
 
Việc này KHÔNG cần video, chỉ cần dev_question.jsonl + submissions/ đã
có.
 
MÃ THOÁT
    0   chấm được ít nhất một câu
    1   không đọc được bộ câu hỏi, hoặc không câu nào có tệp nộp
"""
 
from __future__ import annotations
 
import argparse
import csv
import json
import sys
from pathlib import Path
 
from src.aic2026.eval import K_THRESHOLDS, compute_final_score
from src.aic2026.eval.io_submissions import load_submission
from src.aic2026.paths import (
    DEV_QUERIES_PATH,
    EVAL_DIR,
    SCORING_LOG_PATH,
    SUBMISSIONS_DIR,
)
from src.aic2026.submit import KIS, QA, TRAKE
 
# Ánh xạ tên loại truy vấn CỦA NGUYÊN (dev_question_Nguyen.jsonl) sang hằng
# số KIS/QA/TRAKE mà submit/formatter.py và scorer.py dùng. CHỈ SỬA Ở ĐÂY
# nếu sau này đổi tên trong file dev.
LOAI_TRUY_VAN_SANG_TASK = {
    "mo_ta": KIS,
    "hoi_dap": QA,
    "chuoi_su_kien": TRAKE,
}
 
 
def doc_dev_questions(duong_dan: Path) -> list[dict]:
    """Đọc dev_question_Nguyen.jsonl, báo rõ số dòng nếu có dòng lỗi JSON."""
    if not duong_dan.exists():
        raise FileNotFoundError(
            f"Không thấy {duong_dan}. Task 11 (bộ câu tự chấm) chưa xong hoặc "
            "chưa chép vào đúng vị trí."
        )
 
    cau_hoi: list[dict] = []
    with duong_dan.open("r", encoding="utf-8") as f:
        for so_dong, dong in enumerate(f, start=1):
            dong = dong.strip()
            if not dong:
                continue
            try:
                cau_hoi.append(json.loads(dong))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{duong_dan.name} dòng {so_dong} không phải JSON hợp lệ "
                    f"({exc}). Sửa dòng này trước khi chạy tiếp — không bỏ "
                    "qua âm thầm, vì sẽ làm sai tổng số câu trong baseline."
                ) from exc
 
    if not cau_hoi:
        raise ValueError(f"{duong_dan.name} không có dòng nào.")
    return cau_hoi
 
 
def build_gt(q: dict) -> tuple[str, str, dict]:
    """
    Ánh xạ MỘT dòng dev_question_Nguyen.jsonl -> (query_id, task, gt)
    mà scoring.py cần. CHỈ SỬA Ở ĐÂY nếu tên trường trong file dev đổi —
    không cần sửa scoring.py.
 
    Schema thật (không phải mẫu ở docs/schema/):
        mọi loại : id, loai_truy_van, cau_hoi, video_id
        mo_ta/hoi_dap : frame_start, frame_end, pts_time
        hoi_dap       : thêm cau_tra_loi
        chuoi_su_kien : cac_giai_doan = [{su_kien, frame_start, frame_end,
                                           pts_time}, ...]  (KHÔNG có
                        frame_start/frame_end ở cấp ngoài)
    """
    query_id = str(q["id"])
    loai = q["loai_truy_van"]
    if loai not in LOAI_TRUY_VAN_SANG_TASK:
        raise ValueError(f"Câu {query_id}: loai_truy_van lạ: {loai!r}")
    task = LOAI_TRUY_VAN_SANG_TASK[loai]
 
    if task in (KIS, QA):
        gt = {
            "gt_video_id": q["video_id"],
            "gt_frame_range": [q["frame_start"], q["frame_end"]],
        }
        if task == QA:
            gt["gt_answer"] = q["cau_tra_loi"]
    else:  # TRAKE
        gt = {
            "gt_video_id": q["video_id"],
            "gt_events": [
                [giai_doan["frame_start"], giai_doan["frame_end"]]
                for giai_doan in q["cac_giai_doan"]
            ],
        }
 
    return query_id, task, gt
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description="Task 12 — chạy tự chấm điểm.")
    parser.add_argument(
        "--tep",
        default=str(DEV_QUERIES_PATH),
        help="Đường dẫn dev_questions.jsonl (mặc định lấy từ paths.py).",
    )
    parser.add_argument(
        "--submissions",
        default=str(SUBMISSIONS_DIR),
        help="Thư mục chứa tệp nộp (mặc định lấy từ paths.py).",
    )
    doi_so = parser.parse_args()
 
    duong_dan_dev = Path(doi_so.tep)
    submissions_dir = Path(doi_so.submissions)
 
    print("=" * 72)
    print("TASK 12 — TỰ CHẤM ĐIỂM THEO CÔNG THỨC BTC")
    print("=" * 72)
 
    try:
        cau_hoi_list = doc_dev_questions(duong_dan_dev)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LỖI: {exc}")
        return 1
 
    print(f"Bộ dev        : {duong_dan_dev} ({len(cau_hoi_list)} câu)")
    print(f"Tệp nộp       : {submissions_dir}")
    print()
 
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
 
    dong_ghi: list[dict] = []
    khong_co_tep_nop: list[str] = []
 
    for q in cau_hoi_list:
        try:
            query_id, task, gt = build_gt(q)
        except (KeyError, ValueError) as exc:
            print(f"  LỖI ở câu {q.get('id', '?')}: {exc}")
            continue
 
        submissions = load_submission(submissions_dir, query_id, task)
        if not submissions:
            khong_co_tep_nop.append(query_id)
            continue
 
        ket_qua = compute_final_score(gt, submissions, task)
        dong_ghi.append({"query_id": query_id, "task": task, **ket_qua})
        print(f"  {query_id:<10} ({task:<5}) final_score = {ket_qua['final_score']:.3f}")
 
    if khong_co_tep_nop:
        print()
        print(f"CẢNH BÁO: {len(khong_co_tep_nop)} câu chưa có tệp nộp trong submissions/:")
        print(f"  {', '.join(khong_co_tep_nop)}")
        print("  Các câu này KHÔNG tính vào điểm nền — chạy Task 4 cho đủ rồi chạy lại.")
 
    if not dong_ghi:
        print()
        print("Không câu nào có tệp nộp để chấm. Dừng lại.")
        return 1
 
    fieldnames = ["query_id", "task"] + [f"R@{k}" for k in K_THRESHOLDS] + ["final_score"]
    with SCORING_LOG_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dong_ghi)
 
    baseline_score = sum(d["final_score"] for d in dong_ghi) / len(dong_ghi)
 
    print()
    print("=" * 72)
    print(f"Số câu đã chấm : {len(dong_ghi)} / {len(cau_hoi_list)}")
    print(f"ĐIỂM NỀN (baseline_score) : {baseline_score:.4f}")
    print(f"Chi tiết       : {SCORING_LOG_PATH}")
    print("=" * 72)
    print()
    print("Ghi con số này vào sheet Nhật ký, kèm ngày giờ và mô tả phiên bản")
    print("hệ thống (settings.yaml lúc chạy Task 4) — Việc 12 yêu cầu.")
 
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())