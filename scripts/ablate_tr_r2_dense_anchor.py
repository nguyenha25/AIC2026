"""Ablation anchor-constrained dense DP mà không encode lại theo config.

Mỗi query chỉ chuẩn bị CLIP-L scorer một lần. Sau đó toàn bộ lưới
max-drift × anchor-penalty × forward-reward chạy trực tiếp trên score matrix
đã có, nên chi phí sweep là O(config × event × frame), rất nhỏ so với encode.

GT chỉ dùng để đánh giá và chọn cấu hình dev; không đi vào inference.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import open_clip  # noqa: F401

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from aic2026.paths import RUNS_DIR
from aic2026.trake_r2_pipeline import (
    _align_prepared_dense_scorer,
    _tr_r1_results_to_events,
    run_trake_r2_diagnostics,
)
from scripts.benchmark_tr_r2 import (
    INPUT_PATH,
    build_tr_r1_results,
    doc_jsonl,
    event_prediction_error,
    get_gt_events_from_records,
    group_records_by_query,
)


DEFAULT_OUTPUT = RUNS_DIR / "tr_r2_dense_anchor_ablation.json"
DEFAULT_DRIFTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
DEFAULT_PENALTIES = (0.0, 0.005, 0.01, 0.02, 0.04)
DEFAULT_FORWARD_REWARDS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06)


def _parse_float_grid(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Grid không được rỗng")
    if any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("Grid không được chứa số âm")
    return tuple(dict.fromkeys(parsed))


def _config_key(
    drift: float,
    penalty: float,
    forward_reward: float,
) -> str:
    return f"drift={drift:g}|penalty={penalty:g}|forward={forward_reward:g}"


def _signed_error(prediction: float, start: float, end: float) -> float:
    if prediction < start:
        return prediction - start
    if prediction > end:
        return prediction - end
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep dense anchor/forward weights trong một lần encode."
    )
    parser.add_argument(
        "--query-id",
        action="append",
        required=True,
        help="Query cần ablate; có thể truyền nhiều lần.",
    )
    parser.add_argument("--artifact", type=Path, default=INPUT_PATH)
    parser.add_argument("--video-beam-size", type=int, default=12)
    parser.add_argument("--span-penalty-weight", type=float, default=0.041)
    parser.add_argument(
        "--drifts",
        type=_parse_float_grid,
        default=DEFAULT_DRIFTS,
        help="Danh sách giây, phân cách dấu phẩy.",
    )
    parser.add_argument(
        "--penalties",
        type=_parse_float_grid,
        default=DEFAULT_PENALTIES,
    )
    parser.add_argument(
        "--forward-rewards",
        type=_parse_float_grid,
        default=DEFAULT_FORWARD_REWARDS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video_beam_size <= 0:
        raise ValueError("--video-beam-size phải > 0")
    if args.span_penalty_weight < 0:
        raise ValueError("--span-penalty-weight phải >= 0")
    if any(drift <= 0 for drift in args.drifts):
        raise ValueError("Mọi --drifts phải > 0")

    grouped = group_records_by_query(doc_jsonl(args.artifact))
    query_ids = tuple(dict.fromkeys(str(value) for value in args.query_id))
    missing = [query_id for query_id in query_ids if query_id not in grouped]
    if missing:
        raise ValueError("Query không có trong artifact: " + ", ".join(missing))

    configs = [
        (float(drift), float(penalty), float(forward_reward))
        for drift in args.drifts
        for penalty in args.penalties
        for forward_reward in args.forward_rewards
    ]

    print("=" * 80)
    print("TR-R2 DENSE ANCHOR/FORWARD ABLATION")
    print("=" * 80)
    print(f"Queries : {', '.join(query_ids)}")
    print(f"Configs : {len(configs)}")
    print(f"Output  : {args.output}")

    aggregates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "correct_video_events": 0,
            "hits": 0,
            "errors": [],
            "signed_errors": [],
            "query_hits": {},
        }
    )
    query_outputs: list[dict[str, Any]] = []

    for query_index, query_id in enumerate(query_ids, start=1):
        records = grouped[query_id]
        gt_events = get_gt_events_from_records(records)
        gt_video_id = str(gt_events[0]["video_id"])
        tr_r1_results = build_tr_r1_results(records)
        events_tr_r1 = _tr_r1_results_to_events(tr_r1_results)

        print(
            f"\n[{query_index}/{len(query_ids)}] query={query_id} "
            "prepare scorer...",
            flush=True,
        )
        started = time.perf_counter()
        diagnostics = run_trake_r2_diagnostics(
            tr_r1_results,
            video_beam_size=int(args.video_beam_size),
            span_penalty_weight=float(args.span_penalty_weight),
            adaptive_dense_rerank=False,
            dense_rerank_top_k=1,
        )
        prepare_seconds = time.perf_counter() - started

        predicted_video_id = str(diagnostics["video_id"])
        video_match = predicted_video_id == gt_video_id
        scorer = diagnostics["scorer"]
        windows = diagnostics["windows"]
        sparse_selection = diagnostics["sparse_selection"]
        anchor_times = sparse_selection["chosen_times"]

        config_outputs: list[dict[str, Any]] = []
        for drift, penalty, forward_reward in configs:
            alignment = _align_prepared_dense_scorer(
                events_tr_r1,
                scorer,
                video_id=predicted_video_id,
                windows=windows,
                min_gap=1,
                anchor_times=anchor_times,
                anchor_max_drift_seconds=drift,
                anchor_penalty_per_second=penalty,
                anchor_forward_reward_per_second=forward_reward,
            )

            details = []
            errors = []
            signed_errors = []
            hits = 0
            for event_id, gt in zip(events_tr_r1, gt_events):
                prediction = float(alignment["chosen_times"][event_id])
                error = event_prediction_error(
                    prediction,
                    float(gt["start_time"]),
                    float(gt["end_time"]),
                )
                signed = _signed_error(
                    prediction,
                    float(gt["start_time"]),
                    float(gt["end_time"]),
                )
                hit = bool(video_match and error == 0.0)
                hits += int(hit)
                if video_match:
                    errors.append(float(error))
                    signed_errors.append(float(signed))
                details.append(
                    {
                        "event_id": event_id,
                        "gt_start": float(gt["start_time"]),
                        "gt_end": float(gt["end_time"]),
                        "sparse_time": float(anchor_times[event_id]),
                        "prediction_time": prediction,
                        "error_seconds": float(error),
                        "signed_error_seconds": float(signed),
                        "hit": hit,
                    }
                )

            key = _config_key(drift, penalty, forward_reward)
            aggregate = aggregates[key]
            aggregate["config"] = {
                "max_drift_seconds": drift,
                "anchor_penalty_per_second": penalty,
                "forward_reward_per_second": forward_reward,
            }
            if video_match:
                aggregate["correct_video_events"] += len(gt_events)
                aggregate["hits"] += hits
                aggregate["errors"].extend(errors)
                aggregate["signed_errors"].extend(signed_errors)
            aggregate["query_hits"][query_id] = hits

            config_outputs.append(
                {
                    "config_key": key,
                    "hits": hits,
                    "mean_error_seconds": (
                        statistics.mean(errors) if errors else None
                    ),
                    "events": details,
                }
            )

        query_outputs.append(
            {
                "query_id": query_id,
                "gt_video_id": gt_video_id,
                "predicted_video_id": predicted_video_id,
                "video_match": video_match,
                "prepare_seconds": prepare_seconds,
                "dense_frame_count": len(scorer.frames),
                "configs": config_outputs,
            }
        )

    leaderboard = []
    for key, aggregate in aggregates.items():
        errors = aggregate.pop("errors")
        signed_errors = aggregate.pop("signed_errors")
        total = int(aggregate["correct_video_events"])
        leaderboard.append(
            {
                "config_key": key,
                **aggregate,
                "hit_rate_given_correct_video": (
                    int(aggregate["hits"]) / total if total else 0.0
                ),
                "mean_error_seconds": (
                    statistics.mean(errors) if errors else None
                ),
                "median_error_seconds": (
                    statistics.median(errors) if errors else None
                ),
                "mean_signed_error_seconds": (
                    statistics.mean(signed_errors) if signed_errors else None
                ),
            }
        )

    leaderboard.sort(
        key=lambda row: (
            -int(row["hits"]),
            (
                float(row["mean_error_seconds"])
                if row["mean_error_seconds"] is not None
                else float("inf")
            ),
            float(row["config"]["max_drift_seconds"]),
            float(row["config"]["forward_reward_per_second"]),
            float(row["config"]["anchor_penalty_per_second"]),
        )
    )

    result = {
        "task": "TR-R2 dense anchor/forward ablation",
        "artifact": str(args.artifact),
        "query_ids": list(query_ids),
        "num_configs": len(configs),
        "leaderboard": leaderboard,
        "recommended_dev_config": leaderboard[0] if leaderboard else None,
        "queries": query_outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nrank hits rate mean_err drift penalty forward query_hits")
    for rank, row in enumerate(leaderboard[:15], start=1):
        config = row["config"]
        mean_error = row["mean_error_seconds"]
        print(
            f"{rank:>4} {row['hits']:>4}/{row['correct_video_events']:<4} "
            f"{row['hit_rate_given_correct_video']:.4f} "
            f"{mean_error:.4f} "
            f"{config['max_drift_seconds']:g} "
            f"{config['anchor_penalty_per_second']:g} "
            f"{config['forward_reward_per_second']:g} "
            f"{row['query_hits']}"
        )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
