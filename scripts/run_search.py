#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from aic2026.paths import FTS_DIR, FAISS_DIR, FRAME_MAP_PARQUET
from aic2026.index.fts_index import TextSearchIndex
from aic2026.frame_map import FrameMap

from aic2026.rank.config import RankConfig
from aic2026.rank.search import MultimediaSearchEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Câu truy vấn")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--out", type=str, default="submissions/submission_phase1.csv")
    args = parser.parse_args()

    # 1. Nạp kho dữ liệu OCR (Task 9)
    db_path = FTS_DIR / "text.sqlite"
    text_idx = TextSearchIndex(db_path)

    # 2. Thử nạp kho dữ liệu Visual (Bọc trong try-except để không bị crash nếu code bên kia lỗi)
    faiss_idx = None
    clip_enc = None
    faiss_index_file = FAISS_DIR / "clip_b32.index"
    
    if faiss_index_file.exists():
        try:
            from aic2026.index.faiss_index import FaissIndex
            from aic2026.index.encode.clip_encoder import ClipEncoder
            
            print(f"[*] Phát hiện kho Visual Index, đang nạp...")
            faiss_idx = FaissIndex.load(faiss_index_file)
            clip_enc = ClipEncoder.from_pretrained()
        except ImportError:
            print("[!] Phát hiện có file index nhưng không import được class của Task 12. Chạy chế độ ONLY-OCR.")
    else:
        print("[*] Chưa có file clip_b32.index. Chạy chế độ ONLY-OCR.")

    # 3. Nạp FrameMap để tính giây chuẩn khi khử trùng lặp (Task 1)
    frame_map = FrameMap.from_parquet(FRAME_MAP_PARQUET) if FRAME_MAP_PARQUET.exists() else None

    # 4. Khởi tạo bộ tìm kiếm tổng hợp RRF
    config = RankConfig()
    engine = MultimediaSearchEngine(
        config=config, 
        text_index=text_idx,
        faiss_index=faiss_idx,
        clip_encoder=clip_enc,
        frame_map=frame_map
    )

    print(f"[*] Đang thực thi tìm kiếm cho từ khóa: '{args.query}'")
    results = engine.search(query=args.query, top_k=args.top_k)

    # 5. Xuất file nộp bài chuẩn quy chế
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for item in results:
            writer.writerow([item["video_id"], item["frame_idx"]])

    print(f"[✓] Hoàn thành! Đã xuất {len(results)} dòng ra file {out_path}")

if __name__ == "__main__":
    main()