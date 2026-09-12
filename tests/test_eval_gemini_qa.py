from __future__ import annotations

import json
from types import SimpleNamespace


def _dev_row(query_id: str, video_id: str):
    return {
        "query_id": query_id,
        "video_id": video_id,
        "question": "Có bao nhiêu vị trí?",
        "intent": "count_objects",
        "answer_type": "number",
        "gt": {
            "gt_video_id": video_id,
            "gt_frame_range": [100, 110],
            "gt_answer": "4",
        },
    }


def test_grouped_partition_has_no_video_leakage():
    from scripts.eval_gemini_qa import grouped_partition

    dev = {
        "1": _dev_row("1", "V1"),
        "2": _dev_row("2", "V1"),
        "3": _dev_row("3", "V2"),
        "4": _dev_row("4", "V3"),
        "5": _dev_row("5", "V4"),
    }
    first = grouped_partition(dev, holdout_size=2, seed="fixed")
    second = grouped_partition(dev, holdout_size=2, seed="fixed")

    assert first == second
    videos_by_partition = {"tune": set(), "holdout": set()}
    for query_id, partition in first.items():
        videos_by_partition[partition].add(dev[query_id]["video_id"])
    assert videos_by_partition["tune"].isdisjoint(
        videos_by_partition["holdout"]
    )


def test_build_configs_expands_only_gemini_image_grid():
    from scripts.eval_gemini_qa import build_configs

    configs = build_configs(
        ["none", "gemini", "gemini_rerank"],
        [4, 12],
    )
    assert [config.config_id for config in configs] == [
        "none",
        "gemini_img4",
        "gemini_img12",
        "gemini_rerank_img4",
        "gemini_rerank_img12",
    ]


def test_load_qa_dev_and_r4_use_real_schema(tmp_path):
    from scripts.eval_gemini_qa import load_qa_dev, load_r4

    dev_path = tmp_path / "dev.jsonl"
    dev_path.write_text(
        json.dumps({
            "id": 7,
            "loai_truy_van": "hoi_dap",
            "cau_hoi": "Có bao nhiêu vị trí?",
            "video_id": "L21_V006",
            "frame_start": 1600,
            "frame_end": 1700,
            "cau_tra_loi": "4",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    r4_path = tmp_path / "r4.jsonl"
    r4_path.write_text(
        json.dumps({
            "query_id": "7",
            "selected_candidates": [{
                "video_id": "L21_V006",
                "n": 15,
                "frame_idx": 1653,
                "pts_time": 55.13,
                "score": 0.9,
            }],
        }) + "\n",
        encoding="utf-8",
    )

    dev = load_qa_dev(dev_path)
    r4 = load_r4(r4_path)
    assert dev["7"]["gt"]["gt_answer"] == "4"
    assert dev["7"]["intent"] == "count_objects"
    assert r4["7"][0].frame_idx == 1653


def test_score_rows_separates_ceiling_and_reader(tmp_path):
    from scripts.eval_gemini_qa import EvalConfig, score_rows
    from scripts.qa_v4_pipeline import R4ReaderHit

    hit = R4ReaderHit(
        video_id="L21_V006",
        n=15,
        score=0.9,
        frame_idx=105,
        pts_time=3.5,
        source="test",
    )
    rows = [{
        "hit": hit,
        "video_id": hit.video_id,
        "frame_ids": [hit.frame_idx],
        "answer": "4",
        "answer_confidence": 0.9,
        "answer_source": "gemini",
    }]
    record = score_rows(
        config=EvalConfig("gemini_img12", "gemini", 12),
        query=_dev_row("7", "L21_V006"),
        rows=rows,
        provider_meta={
            "provider": "gemini",
            "selected_images": 12,
            "selected_videos": 4,
            "api_calls": 1,
            "cache_hit": False,
        },
        elapsed_ms=123.0,
        max_rows=100,
        variants_per_frame=1,
        submission_path=tmp_path / "query-7-qa.csv",
    )

    assert record["video_rank"] == 1
    assert record["frame_rank"] == 1
    assert record["first_correct_submission_rank"] == 1
    assert record["final_score"] == 1.0
    assert record["retrieval_ceiling"] == 1.0
    assert record["reader_loss"] == 0.0
    assert record["failure_type"] == "ok"


def test_build_oracle_hits_prioritizes_covered_gt_frames_without_answer():
    from scripts.eval_gemini_qa import build_oracle_hits

    # Cố ý không có gt_answer: candidate builder không được phép đọc đáp án.
    query = {
        "gt": {
            "gt_video_id": "L21_V006",
            "gt_frame_range": [100, 110],
        }
    }
    frame_map = {
        "L21_V006": [
            {"n": 1, "frame_idx": 80, "pts_time": 2.0},
            {"n": 2, "frame_idx": 100, "pts_time": 3.0},
            {"n": 3, "frame_idx": 105, "pts_time": 3.5},
            {"n": 4, "frame_idx": 110, "pts_time": 4.0},
            {"n": 5, "frame_idx": 120, "pts_time": 4.5},
        ]
    }

    hits = build_oracle_hits(query, frame_map, limit=4)

    assert [hit.frame_idx for hit in hits[:3]] == [105, 100, 110]
    assert all(hit.video_id == "L21_V006" for hit in hits)
    assert all(hit.source == "oracle_gt" for hit in hits[:3])
    assert hits[3].source == "oracle_context"


def test_build_oracle_hits_keeps_nearest_context_when_interval_has_no_keyframe():
    from scripts.eval_gemini_qa import build_oracle_hits

    query = {
        "gt": {
            "gt_video_id": "V1",
            "gt_frame_range": [101, 109],
        }
    }
    frame_map = {
        "V1": [
            {"n": 1, "frame_idx": 90, "pts_time": 3.0},
            {"n": 2, "frame_idx": 110, "pts_time": 4.0},
            {"n": 3, "frame_idx": 130, "pts_time": 5.0},
        ]
    }

    hits = build_oracle_hits(query, frame_map, limit=2)

    assert [hit.frame_idx for hit in hits] == [110, 90]
    assert all(hit.source == "oracle_context" for hit in hits)


def test_candidate_mode_defaults_to_r4_and_accepts_oracle_gt():
    from scripts.eval_gemini_qa import build_parser

    assert build_parser().parse_args([]).candidate_mode == "r4"
    parsed = build_parser().parse_args([
        "--candidate-mode", "oracle_gt",
        "--frame-map", r"D:\aic-data\index\frame_map.parquet",
    ])
    assert parsed.candidate_mode == "oracle_gt"


def test_oracle_reader_only_scores_only_gemini_assessed_rows():
    from scripts.eval_gemini_qa import apply_gemini_eval_rows

    rows = [
        {
            "video_id": "V1",
            "frame_ids": [100 + index],
            "answer": "local answer",
            "answer_source": "local",
        }
        for index in range(3)
    ]
    assessments = {
        1: SimpleNamespace(
            relevance=0.9,
            confidence=0.8,
            answer="khong ro",
            evidence="mờ",
        )
    }

    output = apply_gemini_eval_rows(
        rows,
        assessments,
        rerank=False,
        reader_only=True,
    )

    assert len(output) == 1
    assert output[0]["frame_ids"] == [101]
    assert output[0]["answer"] == "khong ro"
    assert output[0]["answer_source"] == "gemini"


def test_summarize_excludes_cloud_fallback_and_oracle_gap_from_score():
    from scripts.eval_gemini_qa import EvalConfig, summarize

    common = {
        "config_id": "gemini_img4",
        "candidate_mode": "oracle_gt",
        "diagnostic_only": True,
        "status": "ok",
        "latency_ms": 10.0,
        "video_hit": True,
        "frame_hit": True,
        "correct_answer_hit": True,
        "gt_frame_rows": 1,
        "correct_gt_frame_rows": 1,
        "retrieval_ceiling": 1.0,
        "reader_loss": 0.0,
        "delta_vs_none": 1.0,
        "api_calls": 1,
        "cache_hit": False,
    }
    records = [
        {
            **common,
            "query_id": "1",
            "oracle_eligible": True,
            "provider_fallback": False,
            "final_score": 1.0,
        },
        {
            **common,
            "query_id": "2",
            "oracle_eligible": True,
            "provider_fallback": True,
            "final_score": 0.0,
        },
        {
            **common,
            "query_id": "3",
            "oracle_eligible": False,
            "provider_fallback": False,
            "frame_hit": False,
            "correct_answer_hit": False,
            "gt_frame_rows": 0,
            "correct_gt_frame_rows": 0,
            "retrieval_ceiling": 0.0,
            "final_score": 0.0,
        },
    ]

    summary = summarize(
        records,
        EvalConfig("gemini_img4", "gemini", 4),
    )

    assert summary["gemini_valid_queries"] == 2
    assert summary["metric_queries"] == 1
    assert summary["excluded_oracle_gap_queries"] == 1
    assert summary["fallback_queries"] == 1
    assert summary["mean_final_score"] == 1.0
    assert summary["mean_final_score_all_status_ok"] == 1 / 3
    assert summary["evaluation_valid"] is False


def test_provider_failure_category_is_compact_and_stable():
    from scripts.eval_gemini_qa import provider_failure_category

    assert provider_failure_category("Gemini HTTP 429: quota exceeded") == (
        "quota_exceeded"
    )
    assert provider_failure_category("Gemini trả JSON không hợp lệ") == (
        "invalid_json"
    )
    assert provider_failure_category(None) is None
