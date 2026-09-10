import json

import pytest

from scripts.eval_02_retrieval_pareto import (
    aggregate_trake,
    build_manifest,
    evaluate_qa_reader_k,
    extract_n02_retrieval_profile,
    load_qa_r1_per_query_latency,
    pareto_front,
    percentile,
    qa_candidate_matches_gt,
    score_trake_prediction,
    _choose_sparse_video,
)


def _gt_qa(qid, video="V1", start=100, end=110):
    return {
        "id": str(qid),
        "loai_truy_van": "hoi_dap",
        "video_id": video,
        "frame_start": start,
        "frame_end": end,
        "cau_tra_loi": "x",
    }


def _cand(video, frame):
    return {"video_id": video, "frame_id": frame, "n": frame, "score": 1.0}


def test_percentile_empty_and_single():
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0


def test_percentile_p50_p95_linear():
    values = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 20.0
    assert percentile(values, 95) == pytest.approx(38.0)


def test_qa_candidate_match_requires_video_and_frame_window():
    gt = _gt_qa("1")
    assert qa_candidate_matches_gt(_cand("V1", 105), gt)
    assert not qa_candidate_matches_gt(_cand("V2", 105), gt)
    assert not qa_candidate_matches_gt(_cand("V1", 999), gt)


def test_fixed_k_and_adaptive_use_same_pool_and_fixed_recall_is_monotonic():
    gt = [_gt_qa("1"), _gt_qa("2"), _gt_qa("3")]
    r3 = [
        {"query_id": "1", "candidates": [_cand("X", 1)] * 3 + [_cand("V1", 105)] + [_cand("X", 2)] * 8},
        {"query_id": "2", "candidates": [_cand("X", 1)] * 7 + [_cand("V1", 105)] + [_cand("X", 2)] * 4},
        {"query_id": "3", "candidates": [_cand("X", 1)] * 11 + [_cand("V1", 105)]},
    ]
    s4 = [
        {"query_id": "1", "task": "qa", "semantic_k_hint": 100},
        {"query_id": "2", "task": "qa", "semantic_k_hint": 300},
        {"query_id": "3", "task": "qa", "semantic_k_hint": 500},
    ]

    out = evaluate_qa_reader_k(gt_records=gt, r3_records=r3, s4_records=s4)
    assert out["same_candidate_pool"] is True
    assert out["configs"]["fixed_4"]["reader_recall"] == pytest.approx(1 / 3)
    assert out["configs"]["fixed_8"]["reader_recall"] == pytest.approx(2 / 3)
    assert out["configs"]["fixed_12"]["reader_recall"] == 1.0
    # Adaptive chooses 4/8/12 respectively -> all three GT remain visible.
    assert out["configs"]["adaptive"]["reader_recall"] == 1.0
    assert out["configs"]["adaptive"]["reader_k"]["mean"] == 8.0


def test_adaptive_rejects_unknown_semantic_k():
    with pytest.raises(ValueError, match="semantic_k_hint"):
        evaluate_qa_reader_k(
            gt_records=[_gt_qa("1")],
            r3_records=[{"query_id": "1", "candidates": [_cand("V1", 105)]}],
            s4_records=[{"query_id": "1", "task": "qa", "semantic_k_hint": 250}],
        )


def test_qa_latency_breakdown_keeps_faiss_rerank_and_vlm_separate():
    out = evaluate_qa_reader_k(
        gt_records=[_gt_qa("1")],
        r3_records=[{"query_id": "1", "candidates": [_cand("V1", 105)] * 12}],
        s4_records=[{"query_id": "1", "task": "qa", "semantic_k_hint": 300}],
        r1_latency_by_query={"1": {"faiss_ms": 10.0, "rerank_ms": 5.0}},
        vlm_ms_per_frame=2.0,
    )
    adaptive = out["configs"]["adaptive"]
    assert adaptive["latency_ms"]["faiss"]["p95"] == 10.0
    assert adaptive["latency_ms"]["rerank"]["p95"] == 5.0
    assert adaptive["latency_ms"]["vlm"]["p95"] == 16.0  # K=8
    assert adaptive["latency_ms"]["total_complete"] is True


def test_total_latency_is_incomplete_when_vlm_missing():
    out = evaluate_qa_reader_k(
        gt_records=[_gt_qa("1")],
        r3_records=[{"query_id": "1", "candidates": [_cand("V1", 105)]}],
        s4_records=[{"query_id": "1", "task": "qa", "semantic_k_hint": 100}],
        r1_latency_by_query={"1": {"faiss_ms": 10.0, "rerank_ms": 5.0}},
    )
    assert out["configs"]["adaptive"]["latency_ms"]["total"]["p95"] is None
    assert out["configs"]["adaptive"]["latency_ms"]["total_complete"] is False


def test_pareto_removes_dominated_config():
    points = [
        {"config_id": "slow", "accuracy": 0.9, "latency_ms": 100.0},
        {"config_id": "better", "accuracy": 0.9, "latency_ms": 80.0},
    ]
    assert [p["config_id"] for p in pareto_front(points)] == ["better"]


def test_pareto_keeps_accuracy_latency_tradeoff():
    points = [
        {"config_id": "fast", "accuracy": 0.8, "latency_ms": 50.0},
        {"config_id": "accurate", "accuracy": 0.95, "latency_ms": 100.0},
        {"config_id": "bad", "accuracy": 0.7, "latency_ms": 120.0},
    ]
    assert [p["config_id"] for p in pareto_front(points)] == ["fast", "accurate"]


def test_pareto_ignores_missing_latency():
    points = [
        {"config_id": "unknown", "accuracy": 1.0, "latency_ms": None},
        {"config_id": "valid", "accuracy": 0.9, "latency_ms": 10.0},
    ]
    assert [p["config_id"] for p in pareto_front(points)] == ["valid"]


def _gt_trake():
    return {
        "id": "T1",
        "loai_truy_van": "chuoi_su_kien",
        "video_id": "V1",
        "cac_giai_doan": [
            {"su_kien": "A", "frame_start": 10, "frame_end": 12},
            {"su_kien": "B", "frame_start": 20, "frame_end": 22},
            {"su_kien": "C", "frame_start": 30, "frame_end": 32},
        ],
    }


def test_trake_wrong_video_scores_zero_immediately():
    score = score_trake_prediction(predicted_video_id="WRONG", predicted_frame_ids=[11, 21, 31], gt=_gt_trake())
    assert score["video_correct"] is False
    assert score["event_rscore"] == 0.0
    assert score["correct_events"] == 0


def test_trake_partial_event_rscore_and_frame_error():
    score = score_trake_prediction(predicted_video_id="V1", predicted_frame_ids=[11, 25, 31], gt=_gt_trake())
    assert score["event_rscore"] == pytest.approx(2 / 3)
    assert score["frame_errors"] == [0, 3, 0]


def test_trake_rejects_wrong_event_count():
    with pytest.raises(ValueError, match="prediction có"):
        score_trake_prediction(predicted_video_id="V1", predicted_frame_ids=[11, 21], gt=_gt_trake())


def test_aggregate_trake_reports_video_recall_rscore_and_latency():
    rows = [
        {**score_trake_prediction(predicted_video_id="V1", predicted_frame_ids=[11, 21, 31], gt=_gt_trake()), "latency_ms": 10.0},
        {**score_trake_prediction(predicted_video_id="WRONG", predicted_frame_ids=[11, 21, 31], gt=_gt_trake()), "latency_ms": 20.0},
    ]
    out = aggregate_trake(rows)
    assert out["video_recall"] == 0.5
    assert out["mean_event_rscore"] == 0.5
    assert out["latency_ms"]["p50"] == 15.0


def test_extract_n02_profile_reads_required_recall_levels(tmp_path):
    p = tmp_path / "n02.json"
    p.write_text(
        json.dumps(
            {
                "task": "N-02",
                "retrieval_method": "rrf",
                "dataset_info": {"num_queries": 10},
                "metrics": {
                    "recalls": {"Recall@50": 0.5, "Recall@100": 0.6, "Recall@300": 0.8, "Recall@500": 0.9},
                    "latency_ms": {"p50": 12.0, "p95": 25.0},
                },
            }
        ),
        encoding="utf-8",
    )
    out = extract_n02_retrieval_profile(p)
    assert out["recall_at_k"] == {"50": 0.5, "100": 0.6, "300": 0.8, "500": 0.9}
    assert out["latency_ms"]["p95"] == 25.0


def test_load_qa_r1_latency_uses_explicit_stage_scope(tmp_path):
    p = tmp_path / "r1.json"
    p.write_text(
        json.dumps(
            {
                "query_records": [
                    {
                        "query_id": "1",
                        "latency_ms": {
                            "source_retrieval": 10,
                            "video_fusion": 2,
                            "top300": 3,
                            "top50": 4,
                            "top12": 1,
                            "total": 20,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = load_qa_r1_per_query_latency(p)
    assert out["1"]["faiss_ms"] == 10.0
    assert out["1"]["rerank_ms"] == 9.0
    assert out["1"]["r1_total_ms"] == 20.0


def test_manifest_contains_required_reproducibility_fields():
    m = build_manifest(
        config={"run_id": "abc", "x": 1},
        query_ids=["1", "2"],
        inputs={"dev": "dev_holdout.jsonl"},
    )
    for key in ("run_id", "git_commit", "dataset_version", "query_ids", "config_hash", "machine", "missing_assets"):
        assert key in m
