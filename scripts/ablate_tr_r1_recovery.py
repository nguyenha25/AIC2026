"""Targeted TR-R1 beam-recovery ablation with one raw CLIP retrieval pass.

Mặc định chỉ chạy năm query mà GT video không có trong candidate output của
baseline TR-E2 dual-profile: 07, 15, 31, 45, 70.

Raw CLIP-L hits được lấy đúng một lần cho mỗi event rồi giữ trong RAM. Grid
chỉ chạy lại grouping, video consensus và RRF beam; không encode CLIP lại theo
từng config. GT chỉ dùng để báo cáo offline, không đi vào retrieval/ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import open_clip  # noqa: F401,E402 - import sớm để ổn định Windows runtime

from aic2026.paths import RUNS_DIR  # noqa: E402
from aic2026.semantic.parser import RuleBasedParser  # noqa: E402
from aic2026.trake_retrieval import (  # noqa: E402
    TRR1Config,
    _merge_query_variant_hits,
    _tim_hit_clip_l_mot_query,
    _trr1_visual_action_variants,
    _video_consensus_scores,
    _video_sequence_scores,
    tim_hit_clip_l,
    tim_nhieu_su_kien,
)
from aic2026.trake_r2_windows import rank_video_candidates_rrf  # noqa: E402
from scripts.benchmark_tr_r1 import (  # noqa: E402
    DEV_QUESTIONS,
    build_queryplan,
    doc_jsonl,
    get_gt_stages,
    get_gt_video_id,
)


DEFAULT_QUERY_IDS = ("07", "15", "31", "45", "70")
REGION_LIMITS = (10, 20, 30, 40)
CONSENSUS_WEIGHTS = (0.0, 0.25, 0.45, 0.65, 0.8)
RRF_K_VALUES = (20.0, 60.0, 100.0)
RESCUE_QUOTAS = (0, 2, 4, 6, 8, 12)
SEQUENCE_WEIGHTS = (0.25, 0.50, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation phục hồi GT video vào TR-R1 beam, không encode lại mỗi config."
    )
    parser.add_argument("--dev", type=Path, default=DEV_QUESTIONS)
    parser.add_argument("--query-id", action="append", default=None)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--max-query-variants", type=int, default=4)
    parser.add_argument("--video-beam-size", type=int, default=12)
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=RUNS_DIR / "tr_r1_recovery_raw_hits_cache.json",
    )
    parser.add_argument("--refresh-raw-cache", action="store_true")
    parser.add_argument(
        "--visual-raw-cache",
        type=Path,
        default=RUNS_DIR / "tr_r1_visual_action_hits_cache.json",
    )
    ablation_mode = parser.add_mutually_exclusive_group()
    ablation_mode.add_argument(
        "--sequence-ablation",
        action="store_true",
        help=(
            "Sweep 97 cấu hình sequence-coverage tập trung; dùng raw cache "
            "và vẫn giữ production sequence weight bằng 0."
        ),
    )
    ablation_mode.add_argument(
        "--sequence-span-ablation",
        action="store_true",
        help=(
            "Sweep 18 cấu hình compact-span trên sequence prior từ raw cache; "
            "production span weight vẫn bằng 0."
        ),
    )
    ablation_mode.add_argument(
        "--visual-query-ablation",
        action="store_true",
        help=(
            "Encode thêm một visual-action variant cho event phù hợp, cache "
            "riêng rồi sweep 13 cấu hình recovery."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNS_DIR / "tr_r1_recovery_ablation.json",
    )
    return parser.parse_args()


def _events_regions(results: tuple[Any, ...]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(result.event_id): [
            {
                "video_id": str(region.video_id),
                "start_time": float(region.start_time),
                "end_time": float(region.end_time),
                "score": float(region.score),
                "hits": region.hits,
            }
            for region in result.regions
        ]
        for result in results
    }


def _config_key(
    regions: int,
    consensus: float,
    rrf_k: float,
    rescue: int,
    sequence: float,
    sequence_span: float,
    sequence_span_scale: float,
) -> str:
    return (
        f"regions={regions}|consensus={consensus:g}|"
        f"rrf_k={rrf_k:g}|rescue={rescue}|sequence={sequence:g}|"
        f"sequence_span={sequence_span:g}|span_scale={sequence_span_scale:g}"
    )


def _config_grid(
    sequence_ablation: bool,
    sequence_span_ablation: bool,
    visual_query_ablation: bool,
) -> list[tuple[int, float, float, int, float, float, float]]:
    if visual_query_ablation:
        baseline = (10, 0.45, 60.0, 0, 0.0, 0.0, 60.0)
        focused = [
            (region_limit, 0.45, 60.0, rescue_quota, sequence_weight, 0.0, 60.0)
            for region_limit in (10, 20)
            for rescue_quota in (4, 8)
            for sequence_weight in (0.50, 0.75, 1.0)
        ]
        return [baseline, *focused]

    if sequence_span_ablation:
        baseline = (10, 0.45, 60.0, 0, 0.0, 0.0, 60.0)
        sequence_baseline = (10, 0.45, 60.0, 4, 1.0, 0.0, 60.0)
        focused = [
            (10, 0.45, 60.0, 4, 1.0, span_weight, span_scale)
            for span_weight in (0.10, 0.20, 0.30, 0.40)
            for span_scale in (15.0, 30.0, 60.0, 120.0)
        ]
        return [baseline, sequence_baseline, *focused]

    if not sequence_ablation:
        return [
            (region_limit, consensus_weight, rrf_k, rescue_quota, 0.0, 0.0, 60.0)
            for region_limit in REGION_LIMITS
            for consensus_weight in CONSENSUS_WEIGHTS
            for rrf_k in RRF_K_VALUES
            for rescue_quota in RESCUE_QUOTAS
        ]

    focused = [
        (region_limit, consensus_weight, rrf_k, rescue_quota, sequence_weight, 0.0, 60.0)
        for region_limit in (10, 20)
        for consensus_weight in (0.45, 0.65)
        for rrf_k in (60.0, 100.0)
        for rescue_quota in (4, 8, 12)
        for sequence_weight in SEQUENCE_WEIGHTS
    ]
    baseline = (10, 0.45, 60.0, 0, 0.0, 0.0, 60.0)
    return [baseline, *focused]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _cache_signature(
    dev_path: Path,
    query_events: dict[str, list[dict[str, Any]]],
    *,
    top_k: int,
    max_query_variants: int,
) -> dict[str, Any]:
    dev_bytes = dev_path.read_bytes()
    return {
        "dev_sha256": hashlib.sha256(dev_bytes).hexdigest(),
        "top_k": int(top_k),
        "max_query_variants": int(max_query_variants),
        "query_events": {
            query_id: [
                {
                    "event_id": event["event_id"],
                    "text": event["text"],
                }
                for event in events
            ]
            for query_id, events in sorted(query_events.items())
        },
    }


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.max_query_variants <= 0:
        raise ValueError("--top-k và --max-query-variants phải > 0")
    if args.video_beam_size <= 0:
        raise ValueError("--video-beam-size phải > 0")

    wanted = {str(value) for value in (args.query_id or DEFAULT_QUERY_IDS)}
    rows = [
        row
        for row in doc_jsonl(args.dev)
        if str(row.get("id")) in wanted
        and row.get("loai_truy_van") == "chuoi_su_kien"
    ]
    found = {str(row.get("id")) for row in rows}
    if found != wanted:
        raise ValueError("Thiếu query: " + ", ".join(sorted(wanted - found)))

    parser = RuleBasedParser()
    query_events: dict[str, list[dict[str, Any]]] = {}
    row_by_query: dict[str, dict[str, Any]] = {}

    for row in rows:
        query_id = str(row["id"])
        plan = build_queryplan(get_gt_stages(row), parser)
        query_events[query_id] = [
            {
                "event_id": str(event.event_id),
                "text": str(event.text),
                "relation": event.relation,
            }
            for event in plan.events
        ]
        row_by_query[query_id] = row

    signature = _cache_signature(
        args.dev,
        query_events,
        top_k=args.top_k,
        max_query_variants=args.max_query_variants,
    )
    cached_hits: dict[str, Any] = {}
    raw_cache_hit = False
    if args.raw_cache.exists() and not args.refresh_raw_cache:
        try:
            cached_payload = json.loads(args.raw_cache.read_text(encoding="utf-8"))
            if cached_payload.get("signature") == signature:
                cached_hits = dict(cached_payload.get("hits", {}))
                raw_cache_hit = True
        except (OSError, ValueError, TypeError):
            cached_hits = {}

    visual_variants = {
        query_id: {
            event["text"]: _trr1_visual_action_variants(event["text"])
            for event in events
            if _trr1_visual_action_variants(event["text"])
        }
        for query_id, events in query_events.items()
    }
    visual_signature = {
        "schema_version": 1,
        "top_k": int(args.top_k),
        "variants": visual_variants,
    }
    visual_cached_hits: dict[str, Any] = {}
    visual_cache_hit = False
    if args.visual_query_ablation and args.visual_raw_cache.exists():
        try:
            visual_payload = json.loads(
                args.visual_raw_cache.read_text(encoding="utf-8")
            )
            if visual_payload.get("signature") == visual_signature:
                visual_cached_hits = dict(visual_payload.get("hits", {}))
                visual_cache_hit = True
        except (OSError, ValueError, TypeError):
            visual_cached_hits = {}

    prepared: dict[str, dict[str, Any]] = {}
    config_grid = _config_grid(
        args.sequence_ablation,
        args.sequence_span_ablation,
        args.visual_query_ablation,
    )

    print("=" * 76)
    print("TR-R1 TARGETED BEAM RECOVERY")
    print("=" * 76)
    print("Queries : " + ", ".join(sorted(wanted)))
    print(f"Raw retrieval: top_k={args.top_k}, variants={args.max_query_variants}")
    mode = (
        "visual action query"
        if args.visual_query_ablation
        else "sequence compact span"
        if args.sequence_span_ablation
        else "sequence coverage"
        if args.sequence_ablation
        else "baseline rescue"
    )
    print(f"Mode    : {mode}")
    print(f"Configs : {len(config_grid)}")

    for index, query_id in enumerate(query_events, start=1):
        row = row_by_query[query_id]
        events = query_events[query_id]
        query_cache = cached_hits.get(query_id, {})
        cache_complete = all(event["text"] in query_cache for event in events)
        source = "disk" if cache_complete else "CLIP"
        print(
            f"[{index}/{len(rows)}] query={query_id}: "
            f"cache {len(events)} event from {source}...",
            flush=True,
        )
        base_hits = (
            {event["text"]: list(query_cache[event["text"]]) for event in events}
            if cache_complete
            else {
                event["text"]: tim_hit_clip_l(
                    event["text"],
                    top_k=args.top_k,
                    use_query_expansion=True,
                    max_query_variants=args.max_query_variants,
                )
                for event in events
            }
        )
        cached_hits[query_id] = base_hits
        raw_hits = base_hits

        if args.visual_query_ablation:
            query_visual_cache = visual_cached_hits.get(query_id, {})
            augmented_hits: dict[str, list[dict[str, Any]]] = {}
            encoded_variants = 0
            disk_variants = 0

            for event in events:
                text = event["text"]
                extra_hit_lists: list[list[dict[str, Any]]] = []
                for variant in visual_variants.get(query_id, {}).get(text, []):
                    if variant in query_visual_cache:
                        variant_hits = list(query_visual_cache[variant])
                        disk_variants += 1
                    else:
                        variant_hits = _tim_hit_clip_l_mot_query(
                            variant,
                            args.top_k,
                        )
                        query_visual_cache[variant] = variant_hits
                        encoded_variants += 1
                    extra_hit_lists.append(variant_hits)

                augmented_hits[text] = (
                    _merge_query_variant_hits([base_hits[text], *extra_hit_lists])
                    if extra_hit_lists
                    else base_hits[text]
                )

            visual_cached_hits[query_id] = query_visual_cache
            raw_hits = augmented_hits
            print(
                f"[TR-R1-VISUAL] query={query_id} "
                f"disk={disk_variants} encoded={encoded_variants}",
                flush=True,
            )
        gt_video_id = get_gt_video_id(row)
        raw_hits_in_order = [raw_hits[event["text"]] for event in events]
        consensus_scores = _video_consensus_scores(raw_hits_in_order)
        consensus_order = sorted(
            consensus_scores,
            key=lambda video_id: (-float(consensus_scores[video_id]), video_id),
        )
        sequence_scores = _video_sequence_scores(raw_hits_in_order)
        sequence_order = sorted(
            sequence_scores,
            key=lambda video_id: (-float(sequence_scores[video_id]), video_id),
        )
        gt_consensus_rank = (
            consensus_order.index(gt_video_id) + 1
            if gt_video_id in consensus_order
            else None
        )
        gt_sequence_rank = (
            sequence_order.index(gt_video_id) + 1
            if gt_video_id in sequence_order
            else None
        )
        event_diagnostics = []
        for event in events:
            hits = raw_hits[event["text"]]
            gt_ranks = [
                rank
                for rank, hit in enumerate(hits, start=1)
                if str(hit.get("video_id")) == gt_video_id
            ]
            event_diagnostics.append(
                {
                    "event_id": event["event_id"],
                    "best_gt_raw_rank": min(gt_ranks) if gt_ranks else None,
                    "gt_raw_hit_count": len(gt_ranks),
                }
            )
        prepared[query_id] = {
            "events": events,
            "gt_video_id": gt_video_id,
            "raw_hits": raw_hits,
            "raw_gt_video_event_count": sum(
                any(str(hit.get("video_id")) == gt_video_id for hit in raw_hits[event["text"]])
                for event in events
            ),
            "gt_consensus_rank": gt_consensus_rank,
            "gt_sequence_rank": gt_sequence_rank,
            "gt_sequence_score": sequence_scores.get(gt_video_id),
            "event_diagnostics": event_diagnostics,
            "consensus_top12": consensus_order[:12],
            "sequence_top12": sequence_order[:12],
        }

    args.raw_cache.parent.mkdir(parents=True, exist_ok=True)
    args.raw_cache.write_text(
        json.dumps(
            _json_safe({"signature": signature, "hits": cached_hits}),
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    if args.visual_query_ablation:
        args.visual_raw_cache.parent.mkdir(parents=True, exist_ok=True)
        args.visual_raw_cache.write_text(
            json.dumps(
                _json_safe(
                    {
                        "signature": visual_signature,
                        "hits": visual_cached_hits,
                    }
                ),
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

    configs: list[dict[str, Any]] = []
    for (
        region_limit,
        consensus_weight,
        rrf_k,
        rescue_quota,
        sequence_weight,
        sequence_span_weight,
        sequence_span_scale,
    ) in config_grid:
        query_rows: list[dict[str, Any]] = []
        for query_id, item in prepared.items():
            config = TRR1Config(
                top_k=args.top_k,
                max_region_duration_seconds=10.0,
                region_merge_gap_seconds=2.0,
                min_region_duration_seconds=0.5,
                max_regions_per_event=region_limit,
                use_query_expansion=True,
                max_query_variants=args.max_query_variants,
                video_consensus_weight=consensus_weight,
                video_rrf_k=rrf_k,
                video_sequence_weight=sequence_weight,
                video_sequence_span_weight=sequence_span_weight,
                video_sequence_span_scale_seconds=sequence_span_scale,
                consensus_rescue_videos=rescue_quota,
            )
            raw_hits = item["raw_hits"]
            results = tim_nhieu_su_kien(
                item["events"],
                config=config,
                retriever=lambda text, _top_k, cache=raw_hits: cache[text],
            )
            beam = rank_video_candidates_rrf(
                _events_regions(results),
                k=int(rrf_k),
                limit=args.video_beam_size,
            )
            gt_video_id = str(item["gt_video_id"])
            gt_rank = beam.index(gt_video_id) + 1 if gt_video_id in beam else None
            query_rows.append(
                {
                    "query_id": query_id,
                    "gt_video_id": gt_video_id,
                    "gt_beam_rank": gt_rank,
                    "beam": beam,
                }
            )

        ranks = [row["gt_beam_rank"] for row in query_rows if row["gt_beam_rank"]]
        configs.append(
            {
                "config_key": _config_key(
                    region_limit,
                    consensus_weight,
                    rrf_k,
                    rescue_quota,
                    sequence_weight,
                    sequence_span_weight,
                    sequence_span_scale,
                ),
                "max_regions_per_event": region_limit,
                "video_consensus_weight": consensus_weight,
                "video_rrf_k": rrf_k,
                "video_sequence_weight": sequence_weight,
                "video_sequence_span_weight": sequence_span_weight,
                "video_sequence_span_scale_seconds": sequence_span_scale,
                "consensus_rescue_videos": rescue_quota,
                "beam_hits": len(ranks),
                "mean_gt_rank_given_hit": mean(ranks) if ranks else None,
                "queries": query_rows,
            }
        )

    configs.sort(
        key=lambda row: (
            -int(row["beam_hits"]),
            float(row["mean_gt_rank_given_hit"] or 10**9),
            int(row["max_regions_per_event"]),
            int(row["consensus_rescue_videos"]),
            -float(row["video_sequence_weight"]),
            -float(row["video_sequence_span_weight"]),
            abs(float(row["video_consensus_weight"]) - 0.45),
            abs(float(row["video_rrf_k"]) - 60.0),
        )
    )
    baseline_key = _config_key(10, 0.45, 60.0, 0, 0.0, 0.0, 60.0)
    baseline = next(row for row in configs if row["config_key"] == baseline_key)
    report = {
        "task": "TR-R1 targeted beam recovery",
        "ablation_mode": mode.replace(" ", "_"),
        "warning": "GT is used only for offline evaluation.",
        "query_ids": sorted(wanted),
        "retrieval": {
            "top_k_per_variant": args.top_k,
            "max_query_variants": args.max_query_variants,
            "raw_retrieval_cached_once_per_event": True,
            "persistent_raw_cache": str(args.raw_cache),
            "persistent_raw_cache_hit": raw_cache_hit,
            "visual_raw_cache": str(args.visual_raw_cache),
            "visual_raw_cache_hit": visual_cache_hit,
            "visual_action_variants": visual_variants,
        },
        "video_beam_size": args.video_beam_size,
        "raw_gt_video_event_count": {
            query_id: int(item["raw_gt_video_event_count"])
            for query_id, item in prepared.items()
        },
        "raw_gt_diagnostics": {
            query_id: {
                "gt_video_id": item["gt_video_id"],
                "gt_consensus_rank": item["gt_consensus_rank"],
                "gt_sequence_rank": item["gt_sequence_rank"],
                "gt_sequence_score": item["gt_sequence_score"],
                "events": item["event_diagnostics"],
                "consensus_top12": item["consensus_top12"],
                "sequence_top12": item["sequence_top12"],
            }
            for query_id, item in prepared.items()
        },
        "baseline": baseline,
        "recommended_dev_config": configs[0],
        "ranking": configs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("rank hits mean_rank regions rescue consensus rrf_k sequence span_w span_s")
    for rank, row in enumerate(configs[:15], start=1):
        mean_rank = row["mean_gt_rank_given_hit"]
        shown_mean = f"{mean_rank:.2f}" if mean_rank is not None else "-"
        print(
            f"{rank:>4} {row['beam_hits']:>4}/{len(rows):<2} {shown_mean:>9} "
            f"{row['max_regions_per_event']:>7} "
            f"{row['consensus_rescue_videos']:>6} "
            f"{row['video_consensus_weight']:>9g} {row['video_rrf_k']:>5g} "
            f"{row['video_sequence_weight']:>8g} "
            f"{row['video_sequence_span_weight']:>6g} "
            f"{row['video_sequence_span_scale_seconds']:>6g}"
        )
    print()
    print("Raw GT-video event count:", report["raw_gt_video_event_count"])
    print("Raw GT diagnostics:")
    for query_id, item in report["raw_gt_diagnostics"].items():
        best_ranks = {
            event["event_id"]: event["best_gt_raw_rank"]
            for event in item["events"]
        }
        print(
            f"  {query_id}: consensus_rank={item['gt_consensus_rank']} "
            f"sequence_rank={item['gt_sequence_rank']} "
            f"event_best_ranks={best_ranks}"
        )
    print("Baseline:", baseline["config_key"], f"hits={baseline['beam_hits']}/{len(rows)}")
    print(
        "Recommended (dev):",
        configs[0]["config_key"],
        f"hits={configs[0]['beam_hits']}/{len(rows)}",
    )
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
