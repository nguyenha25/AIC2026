import json
import pytest
from pathlib import Path
from aic2026.index.fts_index import TextSearchIndex

@pytest.fixture
def sample_ocr_dir(tmp_path):
    ocr_dir = tmp_path / "derived" / "ocr"
    ocr_dir.mkdir(parents=True)
    
    # Tạo dữ liệu giả lập chuẩn schema Task 8
    sample_data = [
        {"video_id": "L22_V001", "n": 1, "frame_idx": 0, "text": "BẢN TIN THỜI SỰ 18H", "boxes": [], "engine": "easyocr-1.7.1"},
        {"video_id": "L22_V001", "n": 2, "frame_idx": 90, "text": "", "boxes": [], "engine": "easyocr-1.7.1"},
        {"video_id": "L22_V001", "n": 3, "frame_idx": 239, "text": "MC THIÊN VŨ PHONG LINH", "boxes": [], "engine": "easyocr-1.7.1"},
    ]
    
    file_path = ocr_dir / "L22_V001.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    return ocr_dir

def test_fts_build_and_search(tmp_path, sample_ocr_dir):
    db_path = tmp_path / "index" / "fts" / "text.sqlite"
    fts = TextSearchIndex(db_path=db_path)
    
    # Build Index
    fts.build_index_from_jsonl_dir(sample_ocr_dir)
    
    # Search Exact Match
    results = fts.search_text(query="PHONG LINH", top_k=5)
    assert len(results) == 1
    assert results[0]["video_id"] == "L22_V001"
    assert results[0]["frame_idx"] == 239
    assert "PHONG LINH" in results[0]["text"]