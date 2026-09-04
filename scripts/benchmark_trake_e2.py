"""Benchmark TR-E2 so với TR-R2 trên tập TR-E1 clean."""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import open_clip  # noqa: F401

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from aic2026.paths import RUNS_DIR
from aic2026.semantic.parser import RuleBasedParser
from aic2026.trake_e2_boundary import refine_from_dense_scorer
from aic2026.trake_r2_pipeline import run_trake_r2_diagnostics
from aic2026.trake_retrieval import TRR1Config, tim_nhieu_su_kien

from scripts.run_trake_e2 import (
    DEV_QUESTIONS,
    TOP_K,
    MAX_REGION_DURATION_SECONDS,
    REGION_MERGE_GAP_SECONDS,
    MIN_REGION_DURATION_SECONDS,
    MAX_REGIONS_PER_EVENT,
    VIDEO_CONSENSUS_WEIGHT,
    VIDEO_RRF_K,
    TR_R2_STEP,
    TR_R2_MIN_GAP,
    TR_R2_RRF_K,
    TR_R2_WINDOW_PADDING_SECONDS,
    TR_R2_BATCH_SIZE,
    doc_jsonl,
    get_gt_stages,
    build_queryplan,
    get_chosen_times,
    enforce_boundary_order,
)

SCHEMA_VERSION = "1.0"
E2_SMOOTHING_RADIUS = 0
E2_PEAK_SEARCH_RADIUS = 2
E2_BOUNDARY_MAX_RADIUS = 3


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[1]
        p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return p.stdout.strip() or None
    except Exception:
        return None


def find_latest_trake_clean() -> Path:
    files = sorted(
        RUNS_DIR.glob("*_TR-E1_tap_trake_sach/contracts/trake_clean.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(
            "Không tìm thấy TR-E1 trake_clean.jsonl. "
            "Chạy trước: python -u -m scripts.tao_tap_trake_sach"
        )
    return files[0]


def load_clean_ids(path: Path) -> set[str]:
    ids = set()
    for row in doc_jsonl(path):
        value = row.get("query_id", row.get("id"))
        if value is None:
            raise ValueError(f"{path}: record thiếu query_id/id")
        ids.add(str(value))
    if not ids:
        raise ValueError(f"{path}: clean contract rỗng")
    return ids


def split_by_video(
    rows: list[dict[str, Any]],
    holdout_fraction: float,
) -> dict[str, str]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction phải nằm trong (0,1)")

    vids = sorted({str(r["video_id"]) for r in rows})
    if len(vids) < 2:
        raise ValueError("Cần ít nhất 2 video để split")

    ranked = sorted(
        vids,
        key=lambda v: hashlib.sha256(v.encode("utf-8")).hexdigest(),
    )
    n_holdout = max(1, min(len(ranked) - 1, round(len(ranked) * holdout_fraction)))
    holdout = set(ranked[:n_holdout])

    return {
        v: ("dev_holdout" if v in holdout else "dev_tune")
        for v in vids
    }


def nearest_dense_frame(scorer, t: float):
    frames = list(scorer.frames)
    if not frames:
        raise ValueError("Dense scorer không có frame")
    return min(
        frames,
        key=lambda f: (abs(float(f.pts_time) - float(t)), int(f.frame_idx)),
    )


def distance_to_interval(frame_idx: int, start: int, end: int) -> int:
    if frame_idx < start:
        return start - frame_idx
    if frame_idx > end:
        return frame_idx - end
    return 0


def interval_iou(a: int, b: int, c: int, d: int) -> float:
    if a > b:
        a, b = b, a
    if c > d:
        c, d = d, c
    inter = max(0, min(b, d) - max(a, c) + 1)
    union = max(b, d) - min(a, c) + 1
    return inter / union if union > 0 else 0.0


def build_trr1_config() -> TRR1Config:
    return TRR1Config(
        top_k=TOP_K,
        max_region_duration_seconds=MAX_REGION_DURATION_SECONDS,
        region_merge_gap_seconds=REGION_MERGE_GAP_SECONDS,
        min_region_duration_seconds=MIN_REGION_DURATION_SECONDS,
        max_regions_per_event=MAX_REGIONS_PER_EVENT,
        video_consensus_weight=VIDEO_CONSENSUS_WEIGHT,
        video_rrf_k=VIDEO_RRF_K,
    )


def benchmark_one(
    row: dict[str, Any],
    split_name: str,
    parser_s1: RuleBasedParser,
    config: TRR1Config,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    qid = str(row["id"])
    gt_video = str(row["video_id"])

    # GT được đọc để scoring, nhưng boundary/time không đi vào inference.
    stages = get_gt_stages(row)
    plan = build_queryplan(stages, parser_s1)
    event_ids = [str(e.event_id) for e in plan.events]

    if len(event_ids) != len(stages):
        raise ValueError(
            f"Query {qid}: parser có {len(event_ids)} event, GT có {len(stages)}"
        )

    r1 = tim_nhieu_su_kien(
        [
            {"event_id": e.event_id, "text": e.text, "relation": e.relation}
            for e in plan.events
        ],
        config=config,
    )
    if not r1:
        raise ValueError(f"Query {qid}: TR-R1 rỗng")

    diag = run_trake_r2_diagnostics(
        r1,
        step=TR_R2_STEP,
        min_gap=TR_R2_MIN_GAP,
        rrf_k=TR_R2_RRF_K,
        window_padding_seconds=TR_R2_WINDOW_PADDING_SECONDS,
        batch_size=TR_R2_BATCH_SIZE,
    )
    scorer = diag.get("scorer")
    alignment = diag.get("alignment")
    pred_video = str(diag.get("video_id", ""))

    if scorer is None or not isinstance(alignment, dict):
        raise ValueError(f"Query {qid}: TR-R2 diagnostics thiếu scorer/alignment")

    chosen = get_chosen_times(alignment)

    boundaries = refine_from_dense_scorer(
        scorer=scorer,
        event_ids=event_ids,
        chosen_times=chosen,
        smoothing_radius=E2_SMOOTHING_RADIUS,
        peak_search_radius=E2_PEAK_SEARCH_RADIUS,
        boundary_max_radius=E2_BOUNDARY_MAX_RADIUS,
    )
    boundaries = enforce_boundary_order(
        boundaries,
        scorer=scorer,
        chosen_times=chosen,
    )
    by_event = {str(x.event_id): x for x in boundaries}

    details = []
    for i, (eid, gt) in enumerate(zip(event_ids, stages), start=1):
        gs = int(gt["frame_start"])
        ge = int(gt["frame_end"])
        gt_t = float(gt["pts_time"])

        r2_chosen_t = float(chosen[eid])
        r2f = nearest_dense_frame(scorer, r2_chosen_t)
        r2_idx = int(r2f.frame_idx)
        r2_t = float(r2f.pts_time)

        e2 = by_event[eid]
        e2_idx = int(e2.representative_frame_idx)
        e2_t = float(e2.representative_time)

        same_video = pred_video == gt_video
        r2_hit = same_video and gs <= r2_idx <= ge
        e2_hit = same_video and gs <= e2_idx <= ge

        if same_video:
            r2_dist = distance_to_interval(r2_idx, gs, ge)
            e2_dist = distance_to_interval(e2_idx, gs, ge)
            r2_err = abs(r2_t - gt_t)
            e2_err = abs(e2_t - gt_t)
            iou = interval_iou(
                int(e2.start_frame_idx),
                int(e2.end_frame_idx),
                gs,
                ge,
            )
            if e2_dist < r2_dist:
                comparison = "improved"
            elif e2_dist > r2_dist:
                comparison = "worse"
            else:
                comparison = "same"
        else:
            r2_dist = e2_dist = None
            r2_err = e2_err = None
            iou = 0.0
            comparison = "same_wrong_video"

        details.append(
            {
                "query_id": qid,
                "split": split_name,
                "event_index": i,
                "event_id": eid,
                "event_text": str(gt["su_kien"]),
                "gt_video_id": gt_video,
                "predicted_video_id": pred_video,
                "same_video": same_video,
                "gt_frame_start": gs,
                "gt_frame_end": ge,
                "gt_pts_time": gt_t,
                "r2_chosen_time": r2_chosen_t,
                "r2_frame_idx": r2_idx,
                "r2_real_pts_time": r2_t,
                "r2_hit": r2_hit,
                "r2_frame_distance": r2_dist,
                "r2_abs_time_error": r2_err,
                "e2_start_frame_idx": int(e2.start_frame_idx),
                "e2_end_frame_idx": int(e2.end_frame_idx),
                "e2_start_time": float(e2.start_time),
                "e2_end_time": float(e2.end_time),
                "e2_frame_idx": e2_idx,
                "e2_pts_time": e2_t,
                "e2_hit": e2_hit,
                "e2_frame_distance": e2_dist,
                "e2_abs_time_error": e2_err,
                "e2_boundary_iou": iou,
                "e2_confidence": float(e2.confidence),
                "e2_status": str(e2.status),
                "comparison": comparison,
            }
        )

    qrow = {
        "schema_version": SCHEMA_VERSION,
        "query_id": qid,
        "split": split_name,
        "gt_video_id": gt_video,
        "predicted_video_id": pred_video,
        "video_correct": pred_video == gt_video,
        "num_events": len(details),
        "r2_hits": sum(bool(x["r2_hit"]) for x in details),
        "e2_hits": sum(bool(x["e2_hit"]) for x in details),
        "improved_events": sum(x["comparison"] == "improved" for x in details),
        "same_events": sum(
            x["comparison"] in {"same", "same_wrong_video"} for x in details
        ),
        "worse_events": sum(x["comparison"] == "worse" for x in details),
        "status": "ok",
    }
    return qrow, details


def avg(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def summarize(
    qrows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    def one(split_name: str) -> dict[str, Any]:
        qs = qrows if split_name == "all" else [
            q for q in qrows if q["split"] == split_name
        ]
        es = events if split_name == "all" else [
            e for e in events if e["split"] == split_name
        ]
        if not es:
            return {
                "queries": len(qs),
                "events": 0,
                "video_accuracy": None,
                "r2_hit_rate": None,
                "e2_hit_rate": None,
                "improved": 0,
                "same": 0,
                "worse": 0,
                "r2_mean_frame_distance": None,
                "e2_mean_frame_distance": None,
                "r2_mae_seconds": None,
                "e2_mae_seconds": None,
                "e2_mean_boundary_iou": None,
            }

        same_video = [e for e in es if e["same_video"]]
        return {
            "queries": len(qs),
            "events": len(es),
            "video_accuracy": (
                sum(bool(q["video_correct"]) for q in qs) / len(qs)
                if qs else None
            ),
            "r2_hit_rate": sum(bool(e["r2_hit"]) for e in es) / len(es),
            "e2_hit_rate": sum(bool(e["e2_hit"]) for e in es) / len(es),
            "improved": sum(e["comparison"] == "improved" for e in es),
            "same": sum(
                e["comparison"] in {"same", "same_wrong_video"} for e in es
            ),
            "worse": sum(e["comparison"] == "worse" for e in es),
            "r2_mean_frame_distance": avg(
                [float(e["r2_frame_distance"]) for e in same_video]
            ),
            "e2_mean_frame_distance": avg(
                [float(e["e2_frame_distance"]) for e in same_video]
            ),
            "r2_mae_seconds": avg(
                [float(e["r2_abs_time_error"]) for e in same_video]
            ),
            "e2_mae_seconds": avg(
                [float(e["e2_abs_time_error"]) for e in same_video]
            ),
            "e2_mean_boundary_iou": avg(
                [float(e["e2_boundary_iou"]) for e in same_video]
            ),
        }

    return {
        "dev_tune": one("dev_tune"),
        "dev_holdout": one("dev_holdout"),
        "all": one("all"),
    }


def fmt(x: Any) -> str:
    return "n/a" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def print_metrics(name: str, m: dict[str, Any]) -> None:
    print(f"\n[{name}]")
    print(f"  queries                : {m['queries']}")
    print(f"  events                 : {m['events']}")
    print(f"  video accuracy         : {fmt(m['video_accuracy'])}")
    print(f"  TR-R2 hit rate         : {fmt(m['r2_hit_rate'])}")
    print(f"  TR-E2 hit rate         : {fmt(m['e2_hit_rate'])}")
    print(
        f"  improved / same / worse: "
        f"{m['improved']} / {m['same']} / {m['worse']}"
    )
    print(f"  R2 mean frame distance : {fmt(m['r2_mean_frame_distance'])}")
    print(f"  E2 mean frame distance : {fmt(m['e2_mean_frame_distance'])}")
    print(f"  R2 MAE seconds         : {fmt(m['r2_mae_seconds'])}")
    print(f"  E2 MAE seconds         : {fmt(m['e2_mae_seconds'])}")
    print(f"  E2 mean boundary IoU   : {fmt(m['e2_mean_boundary_iou'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description="TR-E2 benchmark")
    ap.add_argument("--dev", type=Path, default=DEV_QUESTIONS)
    ap.add_argument("--trake-clean", type=Path, default=None)
    ap.add_argument("--holdout-fraction", type=float, default=1.0 / 3.0)
    ap.add_argument("--query-id", action="append", default=None)
    args = ap.parse_args()

    clean_path = args.trake_clean or find_latest_trake_clean()
    clean_ids = load_clean_ids(clean_path)

    rows = [
        r for r in doc_jsonl(args.dev)
        if r.get("loai_truy_van") == "chuoi_su_kien"
        and str(r.get("id")) in clean_ids
    ]

    if args.query_id:
        wanted = {str(x) for x in args.query_id}
        rows = [r for r in rows if str(r.get("id")) in wanted]
        found = {str(r.get("id")) for r in rows}
        missing = wanted - found
        if missing:
            raise ValueError(
                "Query không có trong TR-E1 clean set: "
                + ", ".join(sorted(missing))
            )

    if not rows:
        raise RuntimeError("Không có TRAKE clean để benchmark")

    split_map = split_by_video(rows, args.holdout_fraction)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_id = f"{ts}_TR-E2_benchmark"
    run_dir = RUNS_DIR / run_id
    contracts = run_dir / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)

    parser_s1 = RuleBasedParser()
    config = build_trr1_config()

    qrows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    print("=" * 80)
    print("TR-E2 — BENCHMARK")
    print("=" * 80)
    print(f"Dev          : {args.dev}")
    print(f"TR-E1 clean  : {clean_path}")
    print(f"Sẽ benchmark : {len(rows)}")
    print(f"Run dir      : {run_dir}")

    for i, row in enumerate(rows, start=1):
        qid = str(row["id"])
        vid = str(row["video_id"])
        split_name = split_map[vid]
        print(f"\n[{i}/{len(rows)}] query={qid} video={vid} split={split_name}")

        try:
            qrow, detail = benchmark_one(
                row,
                split_name,
                parser_s1,
                config,
            )
            qrows.append(qrow)
            events.extend(detail)
            print(
                f"  OK | R2={qrow['r2_hits']}/{qrow['num_events']} "
                f"| E2={qrow['e2_hits']}/{qrow['num_events']} "
                f"| improved={qrow['improved_events']} "
                f"| worse={qrow['worse_events']}"
            )
        except Exception as exc:
            failure = {
                "query_id": qid,
                "video_id": vid,
                "split": split_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(
                f"  FAIL | {failure['error_type']}: {failure['error']}"
            )

    metrics = summarize(qrows, events)
    h = metrics["dev_holdout"]

    non_regression = (
        h["r2_hit_rate"] is not None
        and h["e2_hit_rate"] is not None
        and h["e2_hit_rate"] >= h["r2_hit_rate"]
    )

    strict_improvement = False
    if h["r2_hit_rate"] is not None and h["e2_hit_rate"] is not None:
        if h["e2_hit_rate"] > h["r2_hit_rate"]:
            strict_improvement = True
        elif (
            h["e2_hit_rate"] == h["r2_hit_rate"]
            and h["r2_mean_frame_distance"] is not None
            and h["e2_mean_frame_distance"] is not None
            and h["e2_mean_frame_distance"] < h["r2_mean_frame_distance"]
        ):
            strict_improvement = True

    passed = (
        not failures
        and non_regression
        and strict_improvement
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "runtime_failures": len(failures),
        "metrics": metrics,
        "acceptance": {
            "criterion": (
                "Holdout không giảm hit-rate và phải cải thiện: "
                "hit-rate cao hơn, hoặc hit-rate bằng nhưng "
                "mean frame-distance tới GT interval thấp hơn."
            ),
            "holdout_non_regression": non_regression,
            "holdout_strict_improvement": strict_improvement,
            "passed": passed,
        },
    }

    write_jsonl(contracts / "trake_e2_benchmark_queries.jsonl", qrows)
    write_jsonl(contracts / "trake_e2_benchmark_events.jsonl", events)
    write_csv(run_dir / "trake_e2_benchmark_events.csv", events)
    if failures:
        write_jsonl(run_dir / "failures.jsonl", failures)

    write_json(
        run_dir / "split.json",
        {
            "dev_tune": sorted(
                v for v, s in split_map.items() if s == "dev_tune"
            ),
            "dev_holdout": sorted(
                v for v, s in split_map.items() if s == "dev_holdout"
            ),
        },
    )
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "task": "TR-E2 benchmark",
            "git_commit": git_commit(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dev_path": str(args.dev),
            "trake_clean_path": str(clean_path),
            "query_count": len(rows),
            "successful_queries": len(qrows),
            "failed_queries": len(failures),
            "event_count": len(events),
            "e2_config": {
                "smoothing_radius": E2_SMOOTHING_RADIUS,
                "peak_search_radius": E2_PEAK_SEARCH_RADIUS,
                "boundary_max_radius": E2_BOUNDARY_MAX_RADIUS,
            },
        },
    )

    print("\n" + "=" * 80)
    print("KẾT QUẢ")
    print("=" * 80)
    print_metrics("dev_tune", metrics["dev_tune"])
    print_metrics("dev_holdout", metrics["dev_holdout"])
    print_metrics("all", metrics["all"])

    print("\nNGHIỆM THU TR-E2")
    print(f"  runtime failures       : {len(failures)}")
    print(f"  holdout non-regression : {non_regression}")
    print(f"  strict improvement     : {strict_improvement}")
    print(f"  KẾT LUẬN               : {'ĐẠT' if passed else 'CHƯA ĐẠT'}")
    print(f"\nĐã ghi: {run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
