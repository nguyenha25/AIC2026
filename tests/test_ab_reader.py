from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.ab_reader as m


def rec(
    query_id="04",
    video_id="L26_V495",
    frame=3720,
    n=97,
    exact=False,
    semantic=False,
    latency=1000.0,
    status="ok",
    model="model",
):
    return {
        "schema_version": "1.1",
        "query_id": query_id,
        "video_id": video_id,
        "question_vi": "Câu hỏi",
        "gt_answer": "Đáp án",
        "gt_frame_start": 3600,
        "gt_frame_end": 3800,
        "oracle_frame_idx": frame,
        "oracle_n": n,
        "model": model,
        "pred_answer": "dự đoán",
        "exact_match": exact,
        "semantic_match": semantic,
        "latency_ms": latency,
        "intent": "khac",
        "status": status,
    }


def test_khoa_record():
    r = rec(query_id="23", video_id="L24_V004")
    assert m.khoa_record(r) == ("23", "L24_V004")


def test_chi_muc_rejects_duplicate():
    rows = [rec(), rec()]
    with pytest.raises(ValueError, match="trùng record"):
        m.chi_muc(rows, "BLIP")


def test_cung_oracle_frame_true():
    a = rec(frame=100, n=5)
    b = rec(frame=100, n=5)
    assert m.cung_oracle_frame(a, b) is True


def test_cung_oracle_frame_false_if_frame_differs():
    a = rec(frame=100, n=5)
    b = rec(frame=101, n=5)
    assert m.cung_oracle_frame(a, b) is False


def test_quality_winner_prefers_semantic_match():
    blip = rec(exact=True, semantic=False)
    qwen = rec(exact=False, semantic=True)
    assert m.so_sanh_chat_luong(blip, qwen) == "qwen"


def test_quality_winner_uses_exact_as_tiebreak():
    blip = rec(exact=False, semantic=True)
    qwen = rec(exact=True, semantic=True)
    assert m.so_sanh_chat_luong(blip, qwen) == "qwen"


def test_quality_winner_tie():
    blip = rec(exact=True, semantic=True)
    qwen = rec(exact=True, semantic=True)
    assert m.so_sanh_chat_luong(blip, qwen) == "tie"


def test_ghep_records_aligns_same_query_video():
    blip = [rec(latency=1000.0, model="blip")]
    qwen = [rec(latency=100000.0, model="qwen", semantic=True)]

    rows, diag = m.ghep_records(blip, qwen)

    assert len(rows) == 1
    assert diag["aligned_records"] == 1
    assert diag["oracle_frame_mismatch"] == 0
    assert rows[0]["same_oracle_frame"] is True
    assert rows[0]["quality_winner"] == "qwen"
    assert rows[0]["qwen_vs_blip_latency_ratio"] == pytest.approx(100.0)


def test_ghep_records_reports_missing_side():
    blip = [
        rec(query_id="04"),
        rec(query_id="05", video_id="L26_V100"),
    ]
    qwen = [rec(query_id="04")]

    rows, diag = m.ghep_records(blip, qwen)

    assert len(rows) == 1
    assert diag["only_blip"] == [{"query_id": "05", "video_id": "L26_V100"}]
    assert diag["only_qwen"] == []


def test_ghep_records_detects_frame_mismatch():
    blip = [rec(frame=100, n=3)]
    qwen = [rec(frame=101, n=3)]

    rows, diag = m.ghep_records(blip, qwen)

    assert len(rows) == 1
    assert rows[0]["same_oracle_frame"] is False
    assert diag["oracle_frame_mismatch"] == 1


def test_thong_ke_model():
    rows = [
        {
            "same_oracle_frame": True,
            "blip": {
                "status": "ok",
                "exact_match": True,
                "semantic_match": True,
                "latency_ms": 1000.0,
            },
        },
        {
            "same_oracle_frame": True,
            "blip": {
                "status": "ok",
                "exact_match": False,
                "semantic_match": True,
                "latency_ms": 3000.0,
            },
        },
    ]

    stats = m.thong_ke_model(rows, "blip")

    assert stats["n"] == 2
    assert stats["exact_match"] == 1
    assert stats["exact_rate"] == pytest.approx(0.5)
    assert stats["semantic_match"] == 2
    assert stats["semantic_rate"] == pytest.approx(1.0)
    assert stats["avg_latency_ms"] == pytest.approx(2000.0)
    assert stats["median_latency_ms"] == pytest.approx(2000.0)


def test_chon_khuyen_nghi_prefers_semantic_quality():
    blip = {
        "semantic_rate": 0.2,
        "exact_rate": 0.4,
        "avg_latency_ms": 1000.0,
    }
    qwen = {
        "semantic_rate": 0.8,
        "exact_rate": 0.2,
        "avg_latency_ms": 100000.0,
    }

    out = m.chon_khuyen_nghi(blip, qwen)

    assert out["recommended_reader"] == "qwen"
    assert "Semantic Match cao hơn" in out["reason"]


def test_chon_khuyen_nghi_uses_latency_only_when_quality_ties():
    blip = {
        "semantic_rate": 0.8,
        "exact_rate": 0.4,
        "avg_latency_ms": 1000.0,
    }
    qwen = {
        "semantic_rate": 0.8,
        "exact_rate": 0.4,
        "avg_latency_ms": 100000.0,
    }

    out = m.chon_khuyen_nghi(blip, qwen)

    assert out["recommended_reader"] == "blip"
    assert "latency" in out["reason"].lower()


def test_tong_hop_counts_wins_and_warns_on_small_sample():
    blip_rows = [
        rec(query_id="04", exact=False, semantic=False, latency=1000.0, model="blip"),
        rec(query_id="05", video_id="L26_V100", exact=True, semantic=True, latency=1200.0, model="blip"),
    ]
    qwen_rows = [
        rec(query_id="04", exact=False, semantic=True, latency=100000.0, model="qwen"),
        rec(query_id="05", video_id="L26_V100", exact=True, semantic=True, latency=120000.0, model="qwen"),
    ]

    aligned, diag = m.ghep_records(blip_rows, qwen_rows)
    summary = m.tong_hop(aligned, diag)

    assert summary["comparable_records"] == 2
    assert summary["quality_wins"]["qwen"] == 1
    assert summary["quality_wins"]["tie"] == 1
    assert summary["warning"] is not None
