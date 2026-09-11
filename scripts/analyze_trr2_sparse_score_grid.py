"""Ablation offline trọng số TR-R2 sparse từ candidate score đã lưu.

Không encode CLIP và không dùng dense frame. Phép tính là chính xác khi chỉ
thay ``video_prior_weight``, ``order_gain_weight`` và
``span_penalty_weight`` vì ba trọng số này không làm đổi path DP nội bộ của
từng video. ``region_prior_weight`` không được quét vì nó có thể đổi path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TARGETS = ("16", "24", "71")
DEFAULT_GUARDS = ("20", "32", "69")
DEFAULT_VIDEO_WEIGHTS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2)
DEFAULT_ORDER_WEIGHTS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
DEFAULT_SPAN_WEIGHTS = (
    0.0,
    0.01,
    0.02,
    0.03,
    0.041,
    0.05,
    0.06,
    0.08,
    0.1,
    0.15,
    0.2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quét scorer TR-R2 sparse từ report profile-union; "
            "không encode CLIP lại."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-query", action="append", default=None)
    parser.add_argument("--guard-query", action="append", default=None)
    return parser.parse_args()


def _finite_float(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} phải là số hữu hạn")
    return number


def _candidate_score(
    candidate: dict[str, Any],
    *,
    video_weight: float,
    order_weight: float,
    span_weight: float,
) -> float:
    return (
        _finite_float(candidate["mean_score"], name="mean_score")
        + video_weight
        * _finite_float(candidate.get("video_prior", 0.0), name="video_prior")
        + order_weight
        * _finite_float(candidate.get("order_gain", 0.0), name="order_gain")
        - span_weight
        * _finite_float(candidate.get("span_penalty", 0.0), name="span_penalty")
    )


def rerank_query(
    query: dict[str, Any],
    *,
    video_weight: float,
    order_weight: float,
    span_weight: float,
) -> list[dict[str, Any]]:
    selection = query.get("sparse_selection") or {}
    candidates = selection.get("candidate_scores") or []
    if not candidates:
        raise ValueError(
            f"Query {query.get('query_id')!r} thiếu sparse candidate_scores"
        )

    scored = []
    for candidate in candidates:
        row = dict(candidate)
        row["ablated_final_score"] = _candidate_score(
            row,
            video_weight=video_weight,
            order_weight=order_weight,
            span_weight=span_weight,
        )
        scored.append(row)

    return sorted(
        scored,
        key=lambda row: (
            -float(row["ablated_final_score"]),
            int(row["beam_rank"]),
            str(row["video_id"]),
        ),
    )


def evaluate_config(
    queries: Iterable[dict[str, Any]],
    *,
    video_weight: float,
    order_weight: float,
    span_weight: float,
    targets: set[str],
    guards: set[str],
) -> dict[str, Any]:
    predictions: dict[str, str] = {}
    gt_rank_by_query: dict[str, int | None] = {}
    hit_by_query: dict[str, bool] = {}

    for query in queries:
        query_id = str(query["query_id"])
        gt_video_id = str(query["gt_video_id"])
        ranked = rerank_query(
            query,
            video_weight=video_weight,
            order_weight=order_weight,
            span_weight=span_weight,
        )
        predicted_video_id = str(ranked[0]["video_id"])
        gt_rank = next(
            (
                rank
                for rank, candidate in enumerate(ranked, start=1)
                if str(candidate["video_id"]) == gt_video_id
            ),
            None,
        )
        predictions[query_id] = predicted_video_id
        gt_rank_by_query[query_id] = gt_rank
        hit_by_query[query_id] = predicted_video_id == gt_video_id

    target_ids = sorted(targets & set(hit_by_query))
    guard_ids = sorted(guards & set(hit_by_query))
    video_hits = sum(hit_by_query.values())
    target_hits = sum(hit_by_query[query_id] for query_id in target_ids)
    guard_hits = sum(hit_by_query[query_id] for query_id in guard_ids)
    distance_from_current = (
        abs(video_weight)
        + abs(order_weight)
        + abs(span_weight - 0.041)
    )

    return {
        "config_key": (
            f"video={video_weight:g}|order={order_weight:g}|span={span_weight:g}"
        ),
        "weights": {
            "video": video_weight,
            "order": order_weight,
            "span": span_weight,
        },
        "video_hits": video_hits,
        "target_hits": target_hits,
        "guard_hits": guard_hits,
        "guard_total": len(guard_ids),
        "preserves_all_guards": guard_hits == len(guard_ids),
        "predictions": predictions,
        "hit_by_query": hit_by_query,
        "gt_rank_by_query": gt_rank_by_query,
        "distance_from_current": distance_from_current,
    }


def _ranking_key(row: dict[str, Any]) -> tuple[int, int, int, float]:
    return (
        int(row["video_hits"]),
        int(row["target_hits"]),
        int(row["guard_hits"]),
        -float(row["distance_from_current"]),
    )


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    queries = list(report.get("queries") or [])
    if not queries:
        raise ValueError("Report không có queries")

    targets = set(args.target_query or DEFAULT_TARGETS)
    guards = set(args.guard_query or DEFAULT_GUARDS)
    query_ids = {str(query["query_id"]) for query in queries}
    missing_targets = targets - query_ids
    missing_guards = guards - query_ids
    if missing_targets:
        raise ValueError(
            "Thiếu target query: " + ", ".join(sorted(missing_targets))
        )
    if missing_guards:
        raise ValueError(
            "Thiếu guard query: " + ", ".join(sorted(missing_guards))
        )

    configs = [
        evaluate_config(
            queries,
            video_weight=video_weight,
            order_weight=order_weight,
            span_weight=span_weight,
            targets=targets,
            guards=guards,
        )
        for video_weight, order_weight, span_weight in itertools.product(
            DEFAULT_VIDEO_WEIGHTS,
            DEFAULT_ORDER_WEIGHTS,
            DEFAULT_SPAN_WEIGHTS,
        )
    ]
    ranking = sorted(configs, key=_ranking_key, reverse=True)
    guarded_ranking = [row for row in ranking if row["preserves_all_guards"]]
    current = next(
        row
        for row in configs
        if row["weights"] == {"video": 0.0, "order": 0.0, "span": 0.041}
    )
    best_overall = ranking[0]
    best_guarded = guarded_ranking[0] if guarded_ranking else current
    recommended = (
        best_guarded
        if int(best_guarded["video_hits"]) > int(current["video_hits"])
        else current
    )

    target_feasibility: dict[str, dict[str, Any]] = {}
    for query_id in sorted(targets):
        best_rank = min(
            int(row["gt_rank_by_query"][query_id])
            for row in configs
            if row["gt_rank_by_query"][query_id] is not None
        )
        guarded_top1 = [
            row["config_key"]
            for row in guarded_ranking
            if row["hit_by_query"][query_id]
        ]
        target_feasibility[query_id] = {
            "best_gt_rank_in_grid": best_rank,
            "can_win_while_preserving_guards": bool(guarded_top1),
            "first_guarded_winning_config": (
                guarded_top1[0] if guarded_top1 else None
            ),
        }

    output = {
        "task": "TR-R2 sparse offline score-grid ablation",
        "input": str(args.input),
        "note": (
            "Exact reranking for video/order/span weights; no CLIP encode. "
            "Region weight is fixed at zero because it can change the DP path."
        ),
        "num_queries": len(queries),
        "num_configs": len(configs),
        "targets": sorted(targets),
        "guards": sorted(guards),
        "current_config": current,
        "best_overall": best_overall,
        "best_guarded": best_guarded,
        "recommended_config": recommended,
        "improves_current_without_guard_regression": (
            int(recommended["video_hits"]) > int(current["video_hits"])
        ),
        "target_feasibility": target_feasibility,
        "ranking": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("TR-R2 SPARSE OFFLINE SCORE-GRID ABLATION")
    print("=" * 88)
    print(f"Queries : {len(queries)}")
    print(f"Configs : {len(configs)}")
    print(f"Targets : {', '.join(sorted(targets))}")
    print(f"Guards  : {', '.join(sorted(guards))}")
    print()
    print("rank hits target guard video order span config")
    for rank, row in enumerate(ranking[:15], start=1):
        weights = row["weights"]
        print(
            f"{rank:>4} {row['video_hits']:>4}/{len(queries):<2} "
            f"{row['target_hits']:>3}/{len(targets):<2} "
            f"{row['guard_hits']:>3}/{len(guards):<2} "
            f"{weights['video']:>5g} {weights['order']:>5g} "
            f"{weights['span']:>5g} {row['config_key']}"
        )
    print()
    print("Current    :", current["config_key"], f"hits={current['video_hits']}")
    print(
        "Best overall:",
        best_overall["config_key"],
        f"hits={best_overall['video_hits']}",
    )
    print(
        "Best guarded:",
        best_guarded["config_key"],
        f"hits={best_guarded['video_hits']}",
    )
    print(
        "Recommended :",
        recommended["config_key"],
        f"hits={recommended['video_hits']}",
    )
    print("Target feasibility:", target_feasibility)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
