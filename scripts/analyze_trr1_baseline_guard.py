"""Tìm trọng số baseline guard cho TR-R1 profile-union bằng artifact có sẵn.

Không gọi CLIP hoặc DP. Ba source sequence/visual/consensus được đọc từ
benchmark production-union; source baseline được dựng lại từ TR-R1 JSONL.
GT chỉ được dùng sau ranking để đánh giá offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from aic2026.trake_retrieval import fuse_video_profile_lists_rrf


BASELINE_WEIGHTS = tuple(round(index * 0.1, 1) for index in range(31))
RRF_K_VALUES = (20.0, 60.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze baseline guard cho TR-R1 profile-union."
    )
    parser.add_argument("--union-report", type=Path, required=True)
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument("--video-beam-size", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON phải là object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} không phải object")
            rows.append(row)
    return rows


def _baseline_beams(
    records: Sequence[Mapping[str, Any]],
    *,
    rrf_k: float = 60.0,
    limit: int = 12,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    gt_by_query: dict[str, str] = {}
    for record in records:
        query_id = str(record.get("query_id", "")).strip()
        if not query_id:
            raise ValueError("Baseline artifact có record thiếu query_id")
        grouped.setdefault(query_id, []).append(record)
        gt = record.get("gt")
        if isinstance(gt, Mapping) and gt.get("video_id"):
            gt_by_query[query_id] = str(gt["video_id"])

    beams: dict[str, list[str]] = {}
    for query_id, query_records in grouped.items():
        scores: dict[str, float] = {}
        ordered_records = sorted(
            query_records,
            key=lambda row: int(row.get("event_index", 0)),
        )
        for record in ordered_records:
            result = record.get("tr_r1")
            if not isinstance(result, Mapping):
                raise ValueError(f"query={query_id}: thiếu tr_r1")
            regions = result.get("regions")
            if not isinstance(regions, list):
                raise ValueError(f"query={query_id}: tr_r1.regions không phải list")
            seen: set[str] = set()
            unique_rank = 0
            for region in regions:
                if not isinstance(region, Mapping):
                    continue
                video_id = str(region.get("video_id", "")).strip()
                if not video_id or video_id in seen:
                    continue
                seen.add(video_id)
                unique_rank += 1
                scores[video_id] = scores.get(video_id, 0.0) + 1.0 / (
                    rrf_k + unique_rank
                )
        beams[query_id] = sorted(
            scores,
            key=lambda video_id: (-scores[video_id], video_id),
        )[:limit]
    return beams, gt_by_query


def _evaluate(
    ranked_by_query: Mapping[str, Sequence[str]],
    gt_by_query: Mapping[str, str],
    *,
    budget: int,
) -> dict[str, Any]:
    ranks = {
        query_id: (
            list(ranked_by_query[query_id]).index(gt_video_id) + 1
            if gt_video_id in ranked_by_query[query_id]
            else None
        )
        for query_id, gt_video_id in gt_by_query.items()
    }
    finite = [rank for rank in ranks.values() if rank is not None]
    return {
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks.values()),
        "recall_at_b": sum(
            rank is not None and rank <= budget for rank in ranks.values()
        ),
        "mean_gt_rank_given_hit": mean(finite) if finite else None,
        "gt_rank_by_query": ranks,
        "ranked_video_ids_by_query": {
            query_id: list(values)
            for query_id, values in ranked_by_query.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.video_beam_size <= 0:
        raise ValueError("--video-beam-size phải > 0")

    union_report = _read_json(args.union_report)
    union_rows = union_report.get("queries")
    if not isinstance(union_rows, list) or not union_rows:
        raise ValueError("Union report thiếu queries")
    baseline_beams, baseline_gt = _baseline_beams(
        _read_jsonl(args.baseline_artifact),
        limit=args.video_beam_size,
    )

    source_lists_by_query: dict[str, dict[str, list[str]]] = {}
    gt_by_query: dict[str, str] = {}
    current_beam_by_query: dict[str, list[str]] = {}
    for row in union_rows:
        if not isinstance(row, Mapping):
            continue
        query_id = str(row.get("query_id", ""))
        if query_id not in baseline_beams:
            raise ValueError(f"Baseline artifact thiếu query={query_id}")
        gt_video_id = str(row.get("gt_video_id", ""))
        if baseline_gt.get(query_id) not in {None, gt_video_id}:
            raise ValueError(f"GT không nhất quán query={query_id}")
        sources = row.get("source_video_ids")
        if not isinstance(sources, Mapping):
            raise ValueError(f"Union report thiếu source list query={query_id}")
        source_lists_by_query[query_id] = {
            "sequence": list(sources.get("sequence", [])),
            "visual": list(sources.get("visual", [])),
            "consensus": list(sources.get("consensus", [])),
            "baseline": baseline_beams[query_id],
        }
        gt_by_query[query_id] = gt_video_id
        current_beam_by_query[query_id] = list(row.get("beam", []))

    strategies: list[dict[str, Any]] = []
    for baseline_weight in BASELINE_WEIGHTS:
        for rrf_k in RRF_K_VALUES:
            weights = {
                "sequence": 1.5,
                "visual": 0.5,
                "consensus": 1.0,
                "baseline": baseline_weight,
            }
            ranked_by_query = {
                query_id: fuse_video_profile_lists_rrf(
                    source_lists,
                    source_weights=weights,
                    rrf_k=rrf_k,
                    limit=args.video_beam_size,
                )
                for query_id, source_lists in source_lists_by_query.items()
            }
            strategies.append(
                {
                    "config_key": f"baseline={baseline_weight:g}|k={rrf_k:g}",
                    "weights": weights,
                    "rrf_k": rrf_k,
                    **_evaluate(
                        ranked_by_query,
                        gt_by_query,
                        budget=args.video_beam_size,
                    ),
                }
            )

    strategies.sort(
        key=lambda row: (
            -int(row["recall_at_b"]),
            -int(row["recall_at_5"]),
            float(row["mean_gt_rank_given_hit"] or 10**9),
            float(row["weights"]["baseline"]),
            abs(float(row["rrf_k"]) - 100.0),
        )
    )
    current = _evaluate(
        current_beam_by_query,
        gt_by_query,
        budget=args.video_beam_size,
    )
    report = {
        "task": "TR-R1 profile-union baseline guard",
        "warning": "GT is used only for offline evaluation after ranking.",
        "video_beam_size": args.video_beam_size,
        "current_union": current,
        "baseline_beam_by_query": baseline_beams,
        "recommended_dev_strategy": strategies[0],
        "ranking": strategies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("TR-R1 PROFILE-UNION BASELINE GUARD")
    print("=" * 88)
    print(f"Queries : {len(gt_by_query)}")
    print(f"Configs : {len(strategies)}")
    print(
        f"Current : {current['recall_at_b']}/{len(gt_by_query)} | "
        f"ranks={current['gt_rank_by_query']}"
    )
    print()
    print("rank hits r@5 mean_rank baseline_w k gt_ranks")
    for index, row in enumerate(strategies[:15], start=1):
        print(
            f"{index:>4} {row['recall_at_b']:>4}/{len(gt_by_query):<2} "
            f"{row['recall_at_5']:>3}/{len(gt_by_query):<2} "
            f"{row['mean_gt_rank_given_hit']:>9.2f} "
            f"{row['weights']['baseline']:>10g} {row['rrf_k']:>3g} "
            f"{row['gt_rank_by_query']}"
        )
    print()
    print("Recommended (dev):", strategies[0]["config_key"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
