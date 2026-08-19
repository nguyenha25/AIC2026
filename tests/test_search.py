import pytest
from aic2026.rank.config import RankConfig
from aic2026.rank.dedupe import deduplicate_temporal
from aic2026.rank.search import reciprocal_rank_fusion, MultimediaSearchEngine


def test_rrf_scoring_math():
    visual_hits = [
        {"video_id": "L01_V001", "frame_idx": 100, "score": 0.95},  # rank 1
        {"video_id": "L01_V002", "frame_idx": 200, "score": 0.80},  # rank 2
    ]
    ocr_hits = [
        {"video_id": "L01_V002", "frame_idx": 200, "score": 12.5},  # rank 1
        {"video_id": "L01_V003", "frame_idx": 300, "score": 8.0},   # rank 2
    ]
    
    weights = {"visual": 1.0, "ocr": 1.0}
    fused = reciprocal_rank_fusion(
        {"visual": visual_hits, "ocr": ocr_hits},
        weights=weights,
        k_rrf=60,
    )
    
    # L01_V002 xuất hiện ở cả 2: 1/(60+2) + 1/(60+1)
    assert fused[0]["video_id"] == "L01_V002"
    assert fused[0]["frame_idx"] == 200
    assert pytest.approx(fused[0]["score"], 1e-4) == (1 / 62 + 1 / 61)


def test_deduplicate_temporal_window():
    # Giả lập 25 fps: frame 100 (4s), frame 150 (6s) -> cách nhau 2s (< 10s cửa sổ quy định)
    ranked = [
        {"video_id": "L01_V001", "frame_idx": 100, "score": 0.9},
        {"video_id": "L01_V001", "frame_idx": 150, "score": 0.8}, # Trùng cửa sổ -> loại
        {"video_id": "L01_V001", "frame_idx": 500, "score": 0.7}, # Cách 16s -> giữ
        {"video_id": "L01_V002", "frame_idx": 120, "score": 0.6}, # Khác video -> giữ
    ]
    
    deduped = deduplicate_temporal(ranked, frame_map=None, window_seconds=10.0, fps_fallback=25.0)
    assert len(deduped) == 3
    frames_kept = [item["frame_idx"] for item in deduped if item["video_id"] == "L01_V001"]
    assert frames_kept == [100, 500]