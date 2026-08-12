import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from aic2026.index.fts_index import TextSearchIndex

def main():
    db_path = Path("index/fts/text.sqlite")
    ocr_dir = Path("derived/ocr")
    
    fts = TextSearchIndex(db_path=db_path)
    fts.build_index_from_jsonl_dir(ocr_dir)
    
    # Danh sách truy vấn kiểm thử
    test_queries = ["PHONG LINH", "THỜI SỰ", "TUYỂN DỤNG GIÁO VIÊN", "CÂN GIẢI PHÁP"]
    
    print("=== TASK 9 ACCEPTANCE TEST ===")
    for q in test_queries:
        start = time.perf_counter()
        results = fts.search_text(query=q, top_k=10)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        print(f"\nQuery: '{q}' | Latency: {elapsed_ms:.2f} ms | Found: {len(results)}")
        assert elapsed_ms < 100, f"Lỗi: Latency vượt quá 100ms ({elapsed_ms:.2f}ms)"
        
        for idx, r in enumerate(results[:3], 1):
            print(f"  Top {idx}: [{r['video_id']}] frame_idx={r['frame_idx']} | Text: {r['text'][:60]}...")

if __name__ == "__main__":
    main()