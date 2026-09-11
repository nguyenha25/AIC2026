"""Nối sparse profile-union paths vào baseline TR-E2 dual-profile.

Baseline candidates luôn giữ nguyên thứ tự và đứng trước. Script chỉ thêm
candidate hợp lệ, chưa xuất hiện, từ report TR-R1 profile-union + TR-R2 sparse.
Không chạy retrieval, dense alignment hoặc boundary refinement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Giữ baseline TR-E2 dual-profile và nối thêm sparse candidates "
            "từ profile-union report."
        )
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--profile-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument(
        "--allow-partial-profile",
        action="store_true",
        help=(
            "Cho phép report chỉ chứa một số query; query còn lại được giữ "
            "nguyên từ baseline."
        ),
    )
    return parser.parse_args()


def _normalized_candidate(
    candidate: Mapping[str, Any],
    *,
    expected_events: int,
    sparse_rank: int,
) -> dict[str, Any] | None:
    video_id = str(candidate.get("video_id", "")).strip()
    frame_idx = [int(value) for value in candidate.get("chosen_frame_idx", [])]
    if not video_id or len(frame_idx) != expected_events:
        return None
    if any(left >= right for left, right in zip(frame_idx, frame_idx[1:])):
        return None

    return {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "score": float(
            candidate.get(
                "final_score",
                candidate.get("mean_score", candidate.get("total_score", 0.0)),
            )
        ),
        "source": "trr1_profile_union_sparse_dp",
        "sparse_rank": int(sparse_rank),
        "profile_union_beam_rank": int(candidate.get("beam_rank", sparse_rank)),
    }


def merge_query_candidates(
    baseline_query: Mapping[str, Any],
    profile_query: Mapping[str, Any],
    *,
    max_candidates: int,
) -> dict[str, Any]:
    if max_candidates <= 0:
        raise ValueError("max_candidates phải > 0")

    output = dict(baseline_query)
    events = list(output.get("events") or [])
    expected_events = len(events)
    if expected_events <= 0:
        raise ValueError(
            f"Query {output.get('query_id')!r} không có events trong baseline"
        )

    baseline_candidates = [
        dict(candidate)
        for candidate in output.get("ranked_candidates") or []
    ]
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    for candidate in baseline_candidates:
        if len(merged) >= max_candidates:
            break
        video_id = str(candidate.get("video_id", "")).strip()
        frame_idx = [int(value) for value in candidate.get("frame_idx", [])]
        if not video_id or len(frame_idx) != expected_events:
            continue
        if any(left >= right for left, right in zip(frame_idx, frame_idx[1:])):
            continue
        key = (video_id, tuple(frame_idx))
        if key in seen:
            continue
        normalized = dict(candidate)
        normalized["video_id"] = video_id
        normalized["frame_idx"] = frame_idx
        seen.add(key)
        merged.append(normalized)

    kept_baseline = len(merged)
    selection = profile_query.get("sparse_selection") or {}
    profile_candidates = list(selection.get("candidate_scores") or [])
    appended = 0
    invalid = 0
    duplicates = 0

    for sparse_rank, candidate in enumerate(profile_candidates, start=1):
        if not isinstance(candidate, Mapping):
            invalid += 1
            continue
        normalized = _normalized_candidate(
            candidate,
            expected_events=expected_events,
            sparse_rank=sparse_rank,
        )
        if normalized is None:
            invalid += 1
            continue
        key = (normalized["video_id"], tuple(normalized["frame_idx"]))
        if key in seen:
            duplicates += 1
            continue
        if len(merged) >= max_candidates:
            break
        seen.add(key)
        merged.append(normalized)
        appended += 1

    output["ranked_candidates"] = merged[:max_candidates]
    output["ranked_candidate_merge"] = {
        "policy": "baseline_first_then_profile_union_sparse",
        "baseline_candidates_kept": kept_baseline,
        "profile_candidates_available": len(profile_candidates),
        "profile_candidates_appended": appended,
        "profile_candidates_duplicate": duplicates,
        "profile_candidates_invalid": invalid,
        "total_candidates": len(output["ranked_candidates"]),
        "max_candidates": max_candidates,
    }
    return output


def merge_reports(
    baseline: Sequence[Mapping[str, Any]],
    profile_report: Mapping[str, Any],
    *,
    max_candidates: int,
    allow_partial_profile: bool = False,
) -> list[dict[str, Any]]:
    profile_by_query = {
        str(query["query_id"]): query
        for query in profile_report.get("queries") or []
    }
    baseline_ids = {str(query["query_id"]) for query in baseline}
    missing = baseline_ids - set(profile_by_query)
    if missing and not allow_partial_profile:
        raise ValueError(
            "Profile report thiếu query: " + ", ".join(sorted(missing))
        )

    merged: list[dict[str, Any]] = []
    for query in baseline:
        query_id = str(query["query_id"])
        profile_query = profile_by_query.get(query_id)
        if profile_query is None:
            merged.append(dict(query))
            continue
        merged.append(
            merge_query_candidates(
                query,
                profile_query,
                max_candidates=max_candidates,
            )
        )
    return merged


def main() -> None:
    args = parse_args()
    if args.max_candidates <= 0:
        raise ValueError("--max-candidates phải > 0")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    profile_report = json.loads(args.profile_report.read_text(encoding="utf-8"))
    if not isinstance(baseline, list) or not baseline:
        raise ValueError("Baseline phải là JSON array không rỗng")
    if not isinstance(profile_report, dict):
        raise ValueError("Profile report phải là JSON object")

    merged = merge_reports(
        baseline,
        profile_report,
        max_candidates=args.max_candidates,
        allow_partial_profile=args.allow_partial_profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print("TR-E2 DUAL-PROFILE + TR-R1 PROFILE-UNION SPARSE MERGE")
    print("=" * 80)
    print(f"Queries : {len(merged)}")
    print(f"Max/query: {args.max_candidates}")
    print()
    print("query baseline appended duplicate invalid total videos")
    for query in merged:
        info = query.get("ranked_candidate_merge")
        videos = {
            str(candidate["video_id"])
            for candidate in query["ranked_candidates"]
        }
        if info is None:
            print(
                f"{str(query['query_id']):>5} "
                f"{len(query['ranked_candidates']):>8} "
                f"{0:>8} {0:>9} {0:>7} "
                f"{len(query['ranked_candidates']):>5} "
                f"{len(videos):>6}"
            )
            continue
        print(
            f"{str(query['query_id']):>5} "
            f"{info['baseline_candidates_kept']:>8} "
            f"{info['profile_candidates_appended']:>8} "
            f"{info['profile_candidates_duplicate']:>9} "
            f"{info['profile_candidates_invalid']:>7} "
            f"{info['total_candidates']:>5} "
            f"{len(videos):>6}"
        )
    print()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
