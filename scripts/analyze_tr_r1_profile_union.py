"""Đo beam union của nhiều TR-R1 retrieval profile mà không gọi model.

Input là report sequence coverage và visual-query ablation đã chạy. Script
hợp nhất ba danh sách hoàn toàn không dùng GT để xếp hạng candidate:

1. beam của sequence profile;
2. beam của visual-action profile;
3. top raw video-consensus của visual profile.

GT chỉ được đọc sau khi ranking hoàn tất để báo Recall@B offline.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


SOURCE_NAMES = ("sequence", "visual", "consensus")
WEIGHT_VALUES = (0.25, 0.5, 1.0, 1.5, 2.0)
RRF_K_VALUES = (20.0, 60.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze TR-R1 profile-union beam từ report JSON có sẵn."
    )
    parser.add_argument("--sequence-report", type=Path, required=True)
    parser.add_argument("--visual-report", type=Path, required=True)
    parser.add_argument("--budget", type=int, action="append", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Report phải là JSON object: {path}")
    return payload


def _query_rows(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    config = report.get("recommended_dev_config")
    if not isinstance(config, Mapping):
        raise ValueError("Report thiếu recommended_dev_config")

    rows = config.get("queries")
    if not isinstance(rows, list):
        raise ValueError("recommended_dev_config thiếu queries")

    return {
        str(row["query_id"]): dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("query_id") is not None
    }


def _dedupe(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        video_id = str(value or "").strip()
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        output.append(video_id)
    return output


def weighted_rrf(
    source_lists: Mapping[str, Sequence[str]],
    *,
    weights: Mapping[str, float],
    k: float,
) -> list[str]:
    if k < 0:
        raise ValueError("k phải >= 0")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for source_name in SOURCE_NAMES:
        weight = float(weights[source_name])
        if weight < 0:
            raise ValueError("source weight phải >= 0")
        for rank, video_id in enumerate(
            _dedupe(source_lists.get(source_name, [])),
            start=1,
        ):
            scores[video_id] = scores.get(video_id, 0.0) + weight / (k + rank)
            best_rank[video_id] = min(best_rank.get(video_id, rank), rank)

    return sorted(
        scores,
        key=lambda video_id: (-scores[video_id], best_rank[video_id], video_id),
    )


def round_robin(
    source_lists: Mapping[str, Sequence[str]],
    source_order: Sequence[str],
) -> list[str]:
    lists = {
        source_name: _dedupe(source_lists.get(source_name, []))
        for source_name in source_order
    }
    max_length = max((len(values) for values in lists.values()), default=0)
    output: list[str] = []
    seen: set[str] = set()

    for rank in range(max_length):
        for source_name in source_order:
            values = lists[source_name]
            if rank >= len(values):
                continue
            video_id = values[rank]
            if video_id in seen:
                continue
            seen.add(video_id)
            output.append(video_id)

    return output


def _evaluate(
    ranked_by_query: Mapping[str, Sequence[str]],
    gt_by_query: Mapping[str, str],
    budgets: Sequence[int],
) -> dict[str, Any]:
    gt_ranks: dict[str, int | None] = {}
    for query_id, gt_video_id in gt_by_query.items():
        ranked = list(ranked_by_query[query_id])
        gt_ranks[query_id] = (
            ranked.index(gt_video_id) + 1 if gt_video_id in ranked else None
        )

    finite_ranks = [rank for rank in gt_ranks.values() if rank is not None]
    return {
        "recall_at_b": {
            str(budget): sum(
                rank is not None and rank <= budget
                for rank in gt_ranks.values()
            )
            for budget in budgets
        },
        "mean_gt_rank_given_union_hit": (
            mean(finite_ranks) if finite_ranks else None
        ),
        "gt_rank_by_query": gt_ranks,
        "union_size_by_query": {
            query_id: len(ranked)
            for query_id, ranked in ranked_by_query.items()
        },
        "ranked_video_ids_by_query": {
            query_id: list(ranked)
            for query_id, ranked in ranked_by_query.items()
        },
    }


def main() -> None:
    args = parse_args()
    budgets = sorted(set(args.budget or [12, 18, 24, 36]))
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("Mọi --budget phải > 0")

    sequence_report = _load_json(args.sequence_report)
    visual_report = _load_json(args.visual_report)
    sequence_rows = _query_rows(sequence_report)
    visual_rows = _query_rows(visual_report)
    query_ids = sorted(set(sequence_rows) & set(visual_rows))
    if not query_ids:
        raise ValueError("Hai report không có query chung")

    visual_diagnostics = visual_report.get("raw_gt_diagnostics")
    if not isinstance(visual_diagnostics, Mapping):
        raise ValueError("Visual report thiếu raw_gt_diagnostics")

    source_lists_by_query: dict[str, dict[str, list[str]]] = {}
    gt_by_query: dict[str, str] = {}
    for query_id in query_ids:
        sequence_row = sequence_rows[query_id]
        visual_row = visual_rows[query_id]
        diagnostic = visual_diagnostics.get(query_id)
        if not isinstance(diagnostic, Mapping):
            raise ValueError(f"Thiếu visual diagnostic query={query_id}")

        sequence_gt = str(sequence_row.get("gt_video_id", ""))
        visual_gt = str(visual_row.get("gt_video_id", ""))
        if not sequence_gt or sequence_gt != visual_gt:
            raise ValueError(f"GT không nhất quán query={query_id}")

        gt_by_query[query_id] = sequence_gt
        source_lists_by_query[query_id] = {
            "sequence": _dedupe(sequence_row.get("beam", [])),
            "visual": _dedupe(visual_row.get("beam", [])),
            "consensus": _dedupe(diagnostic.get("consensus_top12", [])),
        }

    strategies: list[dict[str, Any]] = []
    for sequence_weight, visual_weight, consensus_weight, rrf_k in itertools.product(
        WEIGHT_VALUES,
        WEIGHT_VALUES,
        WEIGHT_VALUES,
        RRF_K_VALUES,
    ):
        weights = {
            "sequence": sequence_weight,
            "visual": visual_weight,
            "consensus": consensus_weight,
        }
        ranked_by_query = {
            query_id: weighted_rrf(source_lists, weights=weights, k=rrf_k)
            for query_id, source_lists in source_lists_by_query.items()
        }
        strategies.append(
            {
                "strategy": "weighted_rrf",
                "config_key": (
                    f"rrf|sequence={sequence_weight:g}|visual={visual_weight:g}|"
                    f"consensus={consensus_weight:g}|k={rrf_k:g}"
                ),
                "weights": weights,
                "rrf_k": rrf_k,
                **_evaluate(ranked_by_query, gt_by_query, budgets),
            }
        )

    for source_order in itertools.permutations(SOURCE_NAMES):
        ranked_by_query = {
            query_id: round_robin(source_lists, source_order)
            for query_id, source_lists in source_lists_by_query.items()
        }
        strategies.append(
            {
                "strategy": "round_robin",
                "config_key": "round_robin|" + ">".join(source_order),
                "source_order": list(source_order),
                **_evaluate(ranked_by_query, gt_by_query, budgets),
            }
        )

    strategies.sort(
        key=lambda row: (
            tuple(-int(row["recall_at_b"][str(budget)]) for budget in budgets),
            float(row["mean_gt_rank_given_union_hit"] or 10**9),
            str(row["config_key"]),
        )
    )
    report = {
        "task": "TR-R1 profile union analysis",
        "warning": "GT is used only for offline evaluation after candidate ranking.",
        "query_ids": query_ids,
        "budgets": budgets,
        "sources": list(SOURCE_NAMES),
        "source_lists_by_query": source_lists_by_query,
        "recommended_dev_strategy": strategies[0],
        "ranking": strategies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("TR-R1 PROFILE UNION ANALYSIS")
    print("=" * 88)
    print("Queries : " + ", ".join(query_ids))
    print("Budgets : " + ", ".join(str(value) for value in budgets))
    print(f"Configs : {len(strategies)}")
    print()
    header = "rank " + " ".join(f"R@{budget}" for budget in budgets) + " mean_rank config"
    print(header)
    for rank, row in enumerate(strategies[:15], start=1):
        recalls = " ".join(
            f"{row['recall_at_b'][str(budget)]}/{len(query_ids)}"
            for budget in budgets
        )
        mean_rank = row["mean_gt_rank_given_union_hit"]
        shown_mean = f"{mean_rank:.2f}" if mean_rank is not None else "-"
        print(f"{rank:>4} {recalls} {shown_mean:>9} {row['config_key']}")

    recommended = strategies[0]
    print()
    print("Recommended (dev):", recommended["config_key"])
    print("GT ranks:", recommended["gt_rank_by_query"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
