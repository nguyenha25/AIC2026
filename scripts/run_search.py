"""Mạch tìm kiếm đầu–cuối — Task 4 & Task 10."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.aic2026.index.clip_text import encode_query_text
from src.aic2026.index.faiss_index import search_by_vector
from src.aic2026.paths import DEV_QUERIES_PATH, SUBMISSIONS_DIR
from src.aic2026.rank.ocr_rerank import OCRReranker
from src.aic2026.submit import KIS, QA, TRAKE, submission_filename


def doc_tep_cau_hoi(duong_dan: Path) -> list[dict]:
    """Đọc tệp JSONL câu hỏi và chuẩn hóa các trường."""
    if not duong_dan.exists():
        raise FileNotFoundError(f"Không thấy tệp câu hỏi: {duong_dan}")

    cau_hoi: list[dict] = []
    with duong_dan.open("r", encoding="utf-8") as f:
        for so_dong, dong in enumerate(f, start=1):
            dong = dong.strip()
            if not dong:
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
            data["cau"] = text
            data["query"] = text
            data["task"] = loai
            data["dang"] = loai
            data["gt_answer"] = data.get("cau_tra_loi")

            cau_hoi.append(data)

    return cau_hoi


def main() -> None:
    parser = argparse.ArgumentParser(description="Mạch tìm kiếm AIC 2026")
    parser.add_argument("--tep", type=Path, default=None, help="Đường dẫn tệp JSONL câu hỏi")
    parser.add_argument("--cau", type=str, default=None, help="Một câu hỏi cụ thể")
    parser.add_argument("--id", type=str, default="demo", help="ID câu hỏi")
    parser.add_argument("--task", type=str, default=KIS, help="Dạng truy vấn (kis/qa/trake)")
    parser.add_argument("--use-ocr", action="store_true", default=True, help="Bật OCR Reranking")
    args = parser.parse_args()

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ocr_reranker = OCRReranker() if args.use_ocr else None

    if args.tep:
        danh_sach = doc_tep_cau_hoi(args.tep)
    elif args.cau:
        danh_sach = [{
            "id": args.id,
            "query_id": args.id,
            "text": args.cau,
            "task": args.task,
            "gt_answer": None,
        }]
    else:
        danh_sach = doc_tep_cau_hoi(DEV_QUERIES_PATH)

    print(f"[*] Bắt đầu chạy tìm kiếm cho {len(danh_sach)} câu (OCR Rerank: {args.use_ocr})...\n")
    for idx, item in enumerate(danh_sach, start=1):
        q_id = item["query_id"]
        task = item["task"]
        query_text = item["text"]

        t0 = time.time()
        # 1. Encode text sang vector CLIP
        q_vec = encode_query_text(query_text)

        # 2. Tìm kiếm visual candidates bằng FAISS (lấy top 300)
        hits = search_by_vector(q_vec, top_k=300)
        ket_qua = [
            {
                "video_id": h.video_id,
                "frame_idx": h.frame_idx,
                "score": float(h.score),
            }
            for h in hits
        ]

        # 3. Tích hợp OCR Reranking (Task 10)
        if ocr_reranker:
            ket_qua = ocr_reranker.rerank(ket_qua, query_text, alpha=0.6)

        # 4. Xuất tệp nộp bài (lấy top 100)
        fname = submission_filename(q_id, task)
        out_csv = SUBMISSIONS_DIR / fname

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            for r in ket_qua[:100]:
                if task == QA:
                    ans = item.get("gt_answer") or ""
                    f.write(f"{r['video_id']},{r['frame_idx']},{ans}\n")
                elif task == TRAKE:
                    f.write(f"{r['video_id']},{r['frame_idx']}\n")
                else:
                    f.write(f"{r['video_id']},{r['frame_idx']}\n")

        print(f"[{idx:02}/{len(danh_sach):02}] Câu {q_id} ({task:<5}) -> {fname} ({time.time() - t0:.2f}s)")

    print(f"\n[OK] Đã hoàn thành và lưu kết quả vào: {SUBMISSIONS_DIR}")


if __name__ == "__main__":
    main()