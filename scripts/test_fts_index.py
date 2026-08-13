import sys
from pathlib import Path

# Thêm src vào PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from aic2026.index.fts_index import TextSearchIndex

def main():
    ocr_dir = Path("derived/ocr")
    db_path = Path("index/fts/text.sqlite")

    print("1. Khởi tạo SQLite FTS5 Index...")
    fts = TextSearchIndex(db_path=db_path)

    print("2. Nạp dữ liệu từ derived/ocr/ vào SQLite...")
    fts.build_index_from_jsonl_dir(ocr_dir)

    # 3. Thử tìm từ khóa "PHONG LINH" vừa xuất hiện trong OCR của L22_V001
    query = "PHONG LINH"
    print(f"3. Tìm kiếm từ khóa: '{query}'...")
    results = fts.search_text(query=query, top_k=5)

    print(f"\nTìm thấy {len(results)} kết quả:")
    for r in results:
        print(f"  - [{r['video_id']}] n={r['n']}, frame_idx={r['frame_idx']}, score={r['score']:.4f}")
        print(f"    Text: {r['text']}")

if __name__ == "__main__":
    main()