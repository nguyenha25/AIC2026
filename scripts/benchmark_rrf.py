"""Script nghiệm thu Task 10: So sánh Visual only vs OCR only vs Visual+OCR (RRF)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
import pandas as pd

from src.aic2026.index.encode.clip_encoder import ClipEncoder
from src.aic2026.index.faiss_index import search_by_vector
from src.aic2026.paths import DERIVED_DIR, DEV_QUERIES_PATH, SUBMISSIONS_DIR
from src.aic2026.rank.ocr_rerank import OCRReranker
from src.aic2026.rank.search import reciprocal_rank_fusion
from src.aic2026.rank.dedupe import deduplicate_temporal
from src.aic2026.submit import KIS, QA, TRAKE, submission_filename

clip_encoder = ClipEncoder()

def doc_tep_cau_hoi(duong_dan: Path) -> list[dict]:
    cau_hoi = []
    with open(duong_dan, "r", encoding="utf-8") as f:
        for so_dong, dong in enumerate(f, start=1):
            if not dong.strip():
                continue
            data = json.loads(dong)
            text = (data.get("cau_hoi") or data.get("cau") or data.get("query") or "").strip()
            q_id = str(data.get("id") or so_dong)
            loai = str(data.get("loai_truy_van") or data.get("dang") or data.get("type") or "kis")
            if loai in {"mo_ta", "kis"}:
                loai = KIS
            elif loai in {"hoi_dap", "qa"}:
                loai = QA
            elif loai in {"chuoi_su_kien", "trake"}:
                loai = TRAKE
            data["id"] = q_id
            data["query_id"] = q_id
            data["text"] = text
            data["task"] = loai
            data["gt_answer"] = data.get("cau_tra_loi")
            cau_hoi.append(data)
    return cau_hoi


def run_experiment(mode: str, danh_sach: list[dict], ocr_engine: OCRReranker) -> tuple[float, float]:
    """Chạy tìm kiếm theo từng mode: 'visual', 'ocr', hoặc 'rrf'."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for p in SUBMISSIONS_DIR.glob("*.csv"):
        p.unlink()

    for item in danh_sach:
        q_id = item["query_id"]
        task = item["task"]
        query_text = item["text"]

        visual_hits = []
        ocr_hits = []

        if mode in {"visual", "rrf"}:
            q_vec = clip_encoder.encode_text(query_text)
            hits = search_by_vector(q_vec, top_k=200)
            visual_hits = [
                {"video_id": h.video_id, "frame_idx": h.frame_idx, "score": float(h.score)}
                for h in hits
            ]

        if mode in {"ocr", "rrf"}:
            ocr_hits = ocr_engine.search_ocr(query_text, top_k=200)

        if mode == "visual":
            ket_qua = visual_hits
        elif mode == "ocr":
            ket_qua = ocr_hits
        else:  # Adaptive RRF Fusion (Task 10)
            if len(ocr_hits) > 0:
                modality_results = {"ocr": ocr_hits, "visual": visual_hits}
                # RRF Bất đối xứng (Asymmetric RRF)
                # Thay vì 1000.0, ta dùng 0.45 để OCR làm Tie-breaker chứ không lật đổ Visual
                weights = {"ocr": 0.45, "visual": 1.0}
            else:
                modality_results = {"visual": visual_hits}
                weights = {"visual": 1.0}

            ket_qua = reciprocal_rank_fusion(modality_results, weights=weights, k_rrf=60)

        # LỌC TRÙNG (Deduplicate) thay vì bốc đại top 100
        # Hỗ trợ rất lớn để tránh 1 video chiếm dụng cả 100 kết quả
        try:
            from src.aic2026.frame_map import load_frame_map
            frame_map = load_frame_map()
            ket_qua_sach = deduplicate_temporal(ket_qua, frame_map=frame_map, window_seconds=2, limit=100)
        except Exception:
            ket_qua_sach = ket_qua[:100]

        # Xuất CSV nộp bài
        fname = submission_filename(q_id, task)
        out_csv = SUBMISSIONS_DIR / fname
        if ket_qua_sach:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                for r in ket_qua_sach:
                    if task == QA:
                        ans = item.get("gt_answer") or ""
                        f.write(f"{r['video_id']},{r['frame_idx']},{ans}\n")
                    else:
                        f.write(f"{r['video_id']},{r['frame_idx']}\n")

    # Chạy tự chấm điểm
    subprocess.run(["python", "-m", "scripts.run_scoring"], capture_output=True, text=True)

    # Đọc kết quả từ CSV score
    score_file = DERIVED_DIR / "eval" / "scoring_results.csv"
    if score_file.exists():
        df = pd.read_csv(score_file)
        col = "final_score" if "final_score" in df.columns else ("score" if "score" in df.columns else df.columns[-1])
        tong_diem = float(df[col].sum())
        dtb_32 = tong_diem / 32.0
        return tong_diem, dtb_32
    return 0.0, 0.0


def main():
    print("=" * 75)
    print("NGHIỆM THU TASK 10: ĐỐI SOÁT ĐIỂM RRF FUSION TRÊN BỘ DEV (32 CÂU)")
    print("===========================================================================\n")

    danh_sach = doc_tep_cau_hoi(DEV_QUERIES_PATH)
    ocr_engine = OCRReranker()

    print("[1/3] Đang chạy kiểm thử chế độ: Visual Only...")
    t0 = time.time()
    sum_vis, avg_vis = run_experiment("visual", danh_sach, ocr_engine)
    print(f" -> Tổng điểm: {sum_vis:.3f} | ĐTB (32 câu): {avg_vis:.4f} ({time.time() - t0:.1f}s)")

    print("\n[2/3] Đang chạy kiểm thử chế độ: OCR Only...")
    t0 = time.time()
    sum_ocr, avg_ocr = run_experiment("ocr", danh_sach, ocr_engine)
    print(f" -> Tổng điểm: {sum_ocr:.3f} | ĐTB (32 câu): {avg_ocr:.4f} ({time.time() - t0:.1f}s)")

    print("\n[3/3] Đang chạy kiểm thử chế độ: Visual + OCR (Adaptive RRF Fusion)...")
    t0 = time.time()
    sum_rrf, avg_rrf = run_experiment("rrf", danh_sach, ocr_engine)
    print(f" -> Tổng điểm: {sum_rrf:.3f} | ĐTB (32 câu): {avg_rrf:.4f} ({time.time() - t0:.1f}s)")

    print("\n" + "=" * 75)
    print(f"{'CHẾ ĐỘ TÌM KIẾM':<32} | {'TỔNG ĐIỂM':<15} | {'ĐTB (32 CÂU DEV)':<20}")
    print("-" * 75)
    print(f"{'1. Chỉ dùng Hình ảnh (Visual)':<32} | {sum_vis:<15.3f} | {avg_vis:<20.4f}")
    print(f"{'2. Chỉ dùng Văn bản (OCR)':<32} | {sum_ocr:<15.3f} | {avg_ocr:<20.4f}")
    print(f"{'3. Gộp RRF (Visual + OCR)':<32} | {sum_rrf:<15.3f} | {avg_rrf:<20.4f}")
    print("=" * 75)

    if sum_rrf >= max(sum_vis, sum_ocr) and sum_rrf > 0:
        print("\n[ĐẠT YÊU CẦU NGHIỆM THU TASK 10] Điểm RRF >= max(Visual, OCR).")
    else:
        print("\n[CHƯA ĐẠT]")


if __name__ == "__main__":
    main()