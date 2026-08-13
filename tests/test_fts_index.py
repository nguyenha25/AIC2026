import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from aic2026.index.fts_index import TextSearchIndex


@pytest.fixture
def noisy_ocr_dir(tmp_path):
    """Mo phong du lieu OCR that: nhieu tu bi tach roi nhau boi rac OSD,
    khong nam LIEN KE nhau -- giong het tinh huong trong L22_V002.jsonl."""
    ocr_dir = tmp_path / "derived" / "ocr"
    ocr_dir.mkdir(parents=True)

    sample_data = [
        {
            "video_id": "L22_V002",
            "n": 7,
            "frame_idx": 600,
            "text": "[TTZHD 18.30.21 TIN CHÍNH giay SỬ DỤNG SMARTPHONE NHIÊU ẢNH HƯỞNG SỨC KHỎE TINH THẨN THANH THIẾU NIÊN",
            "boxes": [],
            "engine": "easyocr-1.7.1",
        },
        {
            "video_id": "L22_V002",
            "n": 9,
            "frame_idx": 750,
            # "TIN CHÍNH" va cac tu khac chen giua -- KHONG phai cum lien ke
            "text": "[H TlZHD 18830827 TIN CHÍNH già TP-HCM: ĐƯỜNG DÂY MẠl DÂM CHO NGƯỜI NƯỚC NGOÀI PHÁ",
            "boxes": [],
            "engine": "easyocr-1.7.1",
        },
        {
            "video_id": "L22_V001",
            "n": 1,
            "frame_idx": 0,
            "text": "BẢN TIN THỜI SỰ 18H",
            "boxes": [],
            "engine": "easyocr-1.7.1",
        },
    ]

    file_path = ocr_dir / "L22_V002.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return ocr_dir


def test_and_query_matches_non_adjacent_words(tmp_path, noisy_ocr_dir):
    """Truoc day: query 'TP-HCM PHÁ' se ra 0 ket qua vi 2 tu nay khong nam
    lien ke nhau trong text (bi cac tu khac chen giua). Sau khi sua (AND
    giua cac tu, khong doi hoi lien ke) phai tim thay."""
    db_path = tmp_path / "index" / "fts" / "text.sqlite"
    fts = TextSearchIndex(db_path=db_path)
    fts.build_index_from_jsonl_dir(noisy_ocr_dir)

    results = fts.search_text(query="TP-HCM PHÁ", top_k=5)
    assert len(results) == 1
    assert results[0]["video_id"] == "L22_V002"
    assert results[0]["frame_idx"] == 750


def test_exact_phrase_style_query_still_works(tmp_path, noisy_ocr_dir):
    db_path = tmp_path / "index" / "fts" / "text.sqlite"
    fts = TextSearchIndex(db_path=db_path)
    fts.build_index_from_jsonl_dir(noisy_ocr_dir)

    results = fts.search_text(query="THỜI SỰ", top_k=5)
    assert len(results) == 1
    assert results[0]["video_id"] == "L22_V001"


def test_or_fallback_when_and_finds_nothing(tmp_path, noisy_ocr_dir):
    """Neu AND ca 3 tu khong ra gi (vi du 1 tu bi OCR sai hoan toan), OR
    fallback van tim ra ket qua gan dung thay vi tra ve rong."""
    db_path = tmp_path / "index" / "fts" / "text.sqlite"
    fts = TextSearchIndex(db_path=db_path)
    fts.build_index_from_jsonl_dir(noisy_ocr_dir)

    # "XYZKHONGCO" chac chan khong co trong index -> AND ra 0, OR phai cuu duoc
    results = fts.search_text(query="THỜI SỰ XYZKHONGCO", top_k=5)
    assert len(results) >= 1
    assert results[0]["video_id"] == "L22_V001"


def test_no_or_fallback_disabled(tmp_path, noisy_ocr_dir):
    db_path = tmp_path / "index" / "fts" / "text.sqlite"
    fts = TextSearchIndex(db_path=db_path)
    fts.build_index_from_jsonl_dir(noisy_ocr_dir)

    results = fts.search_text(query="THỜI SỰ XYZKHONGCO", top_k=5, fallback_to_or=False)
    assert results == []


def test_empty_query_returns_empty(tmp_path, noisy_ocr_dir):
    db_path = tmp_path / "index" / "fts" / "text.sqlite"
    fts = TextSearchIndex(db_path=db_path)
    fts.build_index_from_jsonl_dir(noisy_ocr_dir)

    assert fts.search_text(query="   ", top_k=5) == []


def test_special_characters_do_not_raise_sql_error(tmp_path, noisy_ocr_dir):
    db_path = tmp_path / "index" / "fts" / "text.sqlite"
    fts = TextSearchIndex(db_path=db_path)
    fts.build_index_from_jsonl_dir(noisy_ocr_dir)

    # Ky tu dac biet cua FTS5 (dau ngoac kep, dau gach ngang...) khong duoc
    # lam vo cau lenh vi moi token da duoc quote rieng.
    results = fts.search_text(query='TP-HCM: "quote" injection\'--', top_k=5)
    assert isinstance(results, list)
