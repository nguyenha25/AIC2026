"""Benchmark production TR-R1 profile-union và tùy chọn TR-R2 sparse-only."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from aic2026.paths import DEV_DIR, RUNS_DIR
from aic2026.semantic.parser import RuleBasedParser
from aic2026.trake_retrieval import TRR1Config, tim_nhieu_su_kien_profile_union
from aic2026.trake_r2_pipeline import run_trake_r2_sparse_selection
from scripts.run_trake_e2 import build_queryplan, doc_jsonl, get_gt_stages


DEFAULT_DEV = DEV_DIR / "dev_questions.jsonl"
DEFAULT_OUTPUT = RUNS_DIR / "tr_r1_profile_union_benchmark.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đo Recall@B của production TR-R1 profile-union."
    )
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--query-id", action="append", default=None)
    parser.add_argument("--video-beam-size", type=int, default=12)
    parser.add_argument(
        "--run-sparse",
        action="store_true",
        help="Chạy thêm TR-R2 sparse selection trên đúng profile-union beam.",
    )
    parser.add_argument(
        "--span-penalty-weight",
        type=float,
        default=0.041,
        help="Span penalty của TR-R2 sparse (mặc định: 0.041).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = parse_args()
    if args.video_beam_size <= 0:
        raise ValueError("--video-beam-size phải > 0")
    if args.span_penalty_weight < 0:
        raise ValueError("--span-penalty-weight phải >= 0")

    rows = [
        row
        for row in doc_jsonl(args.dev)
        if row.get("loai_truy_van") == "chuoi_su_kien"
    ]
    if args.query_id:
        wanted = {str(value) for value in args.query_id}
        rows = [row for row in rows if str(row.get("id")) in wanted]
        found = {str(row.get("id")) for row in rows}
        missing = wanted - found
        if missing:
            raise ValueError("Không tìm thấy query: " + ", ".join(sorted(missing)))
    if not rows:
        raise RuntimeError("Không có query TRAKE để benchmark")

    config = TRR1Config(
        top_k=500,
        max_region_duration_seconds=10.0,
        region_merge_gap_seconds=2.0,
        min_region_duration_seconds=0.5,
        max_regions_per_event=10,
        video_consensus_weight=0.45,
        video_rrf_k=60.0,
    )
    parser = RuleBasedParser()
    query_results: list[dict[str, Any]] = []
    retrieval_latency_ms: list[float] = []
    sparse_latency_ms: list[float] = []

    print("=" * 76)
    title = "TR-R1 PRODUCTION PROFILE-UNION BENCHMARK"
    if args.run_sparse:
        title += " + TR-R2 SPARSE"
    print(title)
    print("=" * 76)
    print(f"Queries : {len(rows)}")
    print(f"Beam    : {args.video_beam_size}")
    if args.run_sparse:
        print(f"Span w  : {args.span_penalty_weight}")

    for index, row in enumerate(rows, start=1):
        query_id = str(row["id"])
        stages = get_gt_stages(row)
        plan = build_queryplan(stages, parser)
        events = [
            {
                "event_id": event.event_id,
                "text": event.text,
                "relation": event.relation,
            }
            for event in plan.events
        ]
        print(f"[{index}/{len(rows)}] query={query_id}", flush=True)
        started = time.perf_counter()
        union = tim_nhieu_su_kien_profile_union(
            events,
            config=config,
            video_beam_size=args.video_beam_size,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        retrieval_latency_ms.append(elapsed)
        beam = list(union["candidate_video_ids"])
        gt_video_id = str(row["video_id"])
        gt_rank = beam.index(gt_video_id) + 1 if gt_video_id in beam else None
        query_result: dict[str, Any] = {
            "query_id": query_id,
            "gt_video_id": gt_video_id,
            "gt_beam_rank": gt_rank,
            "beam_hit": gt_rank is not None,
            "beam": beam,
            "source_video_ids": union["source_video_ids"],
            "visual_variants_by_event": union["visual_variants_by_event"],
            # Giữ tên cũ để report trước đây vẫn đọc được.
            "latency_ms": elapsed,
            "retrieval_latency_ms": elapsed,
        }

        if args.run_sparse:
            sparse_started = time.perf_counter()
            sparse = run_trake_r2_sparse_selection(
                union["results"],
                video_beam_size=args.video_beam_size,
                candidate_video_ids=beam,
                span_penalty_weight=args.span_penalty_weight,
            )
            sparse_elapsed = (time.perf_counter() - sparse_started) * 1000.0
            sparse_latency_ms.append(sparse_elapsed)
            predicted_video_id = str(sparse["video_id"])
            query_result.update(
                {
                    "predicted_video_id": predicted_video_id,
                    "video_match": predicted_video_id == gt_video_id,
                    "sparse_latency_ms": sparse_elapsed,
                    "sparse_selection": sparse["selection"],
                }
            )
            print(
                "  sparse="
                f"{predicted_video_id} match={query_result['video_match']}",
                flush=True,
            )

        query_results.append(query_result)

    budgets = sorted({1, 5, 12, int(args.video_beam_size)})
    summary = {
        "queries": len(query_results),
        "video_beam_size": int(args.video_beam_size),
        "config_key": "rrf|sequence=1.5|visual=0.5|consensus=1|k=100",
        "recall_at_b": {
            str(budget): sum(
                row["gt_beam_rank"] is not None
                and int(row["gt_beam_rank"]) <= budget
                for row in query_results
            )
            / len(query_results)
            for budget in budgets
        },
        "retrieval_latency_ms": {
            "p50": percentile(retrieval_latency_ms, 50.0),
            "p95": percentile(retrieval_latency_ms, 95.0),
            "mean": statistics.mean(retrieval_latency_ms),
        },
    }
    # Tương thích report cũ: latency_ms vẫn là riêng TR-R1 retrieval.
    summary["latency_ms"] = dict(summary["retrieval_latency_ms"])
    if args.run_sparse:
        beam_hits = sum(bool(row["beam_hit"]) for row in query_results)
        video_correct = sum(bool(row["video_match"]) for row in query_results)
        summary.update(
            {
                "run_sparse": True,
                "span_penalty_weight": float(args.span_penalty_weight),
                "beam_hits": beam_hits,
                "video_correct": video_correct,
                "video_accuracy": video_correct / len(query_results),
                "conditional_accuracy_given_beam": (
                    video_correct / beam_hits if beam_hits else 0.0
                ),
                "sparse_latency_ms": {
                    "p50": percentile(sparse_latency_ms, 50.0),
                    "p95": percentile(sparse_latency_ms, 95.0),
                    "mean": statistics.mean(sparse_latency_ms),
                },
            }
        )
    report = {
        "task": "TR-R1 production profile-union benchmark",
        "warning": "GT is used only for offline evaluation after ranking.",
        "summary": summary,
        "queries": query_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Recall@B:", summary["recall_at_b"])
    print(
        "TR-R1 latency p50: "
        f"{summary['retrieval_latency_ms']['p50']:.2f} ms"
    )
    print(
        "TR-R1 latency p95: "
        f"{summary['retrieval_latency_ms']['p95']:.2f} ms"
    )
    print("GT ranks:", {row["query_id"]: row["gt_beam_rank"] for row in query_results})
    if args.run_sparse:
        print(
            "Sparse video accuracy: "
            f"{summary['video_correct']}/{summary['queries']} "
            f"({summary['video_accuracy']:.4f})"
        )
        print(
            "Conditional accuracy: "
            f"{summary['conditional_accuracy_given_beam']:.4f}"
        )
        print(
            "Sparse predictions:",
            {
                row["query_id"]: row["predicted_video_id"]
                for row in query_results
            },
        )
        print(
            "Sparse latency p50: "
            f"{summary['sparse_latency_ms']['p50']:.2f} ms"
        )
        print(
            "Sparse latency p95: "
            f"{summary['sparse_latency_ms']['p95']:.2f} ms"
        )
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
