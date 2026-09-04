"""
TR-R2 — BENCHMARK DENSE TEMPORAL ALIGNMENT

PHẠM VI
-------
Đọc artifact TR-R1 đã export:

    tr_r1_candidates.jsonl
        ↓
    group event theo query_id
        ↓
    reconstruct TRR1Result
        ↓
    run_trake_r2()
        ↓
    dense CLIP-L scoring
        ↓
    strict-increasing temporal DP
        ↓
    đánh giá với GT trong artifact

KHÔNG LÀM TRONG FILE NÀY
------------------------
- không chạy lại TR-R1
- không chạy FAISS retrieval
- không thay đổi TR-R1 regions
- không dùng GT để chọn video/window
- không chạy submission

GT chỉ dùng để đánh giá prediction của TR-R2.
"""

from __future__ import annotations

# ============================================================================
# ENVIRONMENT
# ============================================================================

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
    "python",
)

# Import open_clip sớm, nhưng không tạo model ở đây.
import open_clip  # noqa: F401


# ============================================================================
# NORMAL IMPORTS
# ============================================================================

import json
import statistics
import time
from pathlib import Path
from typing import Any

from aic2026.paths import DATA_ROOT, RUNS_DIR
from aic2026.trake_retrieval import (
    CoarseRegion,
    TRR1Result,
)
from aic2026.trake_r2_pipeline import run_trake_r2


# ============================================================================
# PATHS / CONFIG
# ============================================================================

INPUT_PATH = RUNS_DIR / "tr_r1_candidates.jsonl"
OUTPUT_PATH = RUNS_DIR / "tr_r2_benchmark.json"

STEP = 0.16
MIN_GAP = 1
RRF_K = 60
WINDOW_PADDING_SECONDS = 0.0
BATCH_SIZE = 16


# ============================================================================
# IO
# ============================================================================


def doc_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy JSONL: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no}: JSON không hợp lệ"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_no}: mỗi dòng phải là object"
                )

            rows.append(row)

    return rows


def ghi_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    tmp.replace(path)


# ============================================================================
# ARTIFACT -> TRR1Result
# ============================================================================


def region_from_dict(
    data: dict[str, Any],
) -> CoarseRegion:
    """
    Reconstruct một CoarseRegion từ JSON artifact.
    """

    required = {
        "video_id",
        "start_time",
        "end_time",
        "score",
        "hits",
    }

    missing = required - set(data)

    if missing:
        raise ValueError(
            "TR-R1 region thiếu field: "
            + ", ".join(sorted(missing))
        )

    return CoarseRegion(
        video_id=str(
            data["video_id"]
        ),
        start_time=float(
            data["start_time"]
        ),
        end_time=float(
            data["end_time"]
        ),
        score=float(
            data["score"]
        ),
        hits=tuple(
            data["hits"]
        ),
    )


def result_from_record(
    record: dict[str, Any],
) -> TRR1Result:
    """
    Reconstruct TRR1Result từ một event record
    trong tr_r1_candidates.jsonl.
    """

    tr_r1 = record.get("tr_r1")

    if not isinstance(
        tr_r1,
        dict,
    ):
        raise ValueError(
            f"Query {record.get('query_id')!r}, "
            f"event {record.get('event_id')!r}: "
            "thiếu tr_r1"
        )

    event_id = tr_r1.get(
        "event_id",
        record.get("event_id"),
    )

    text = tr_r1.get(
        "text",
        record.get("text"),
    )

    relation = tr_r1.get(
        "relation",
        record.get("relation"),
    )

    regions_data = tr_r1.get(
        "regions"
    )

    if not isinstance(
        regions_data,
        list,
    ):
        raise ValueError(
            f"Query {record.get('query_id')!r}, "
            f"event {event_id!r}: "
            "tr_r1.regions không phải list"
        )

    if not isinstance(
        event_id,
        str,
    ) or not event_id:
        raise ValueError(
            "TR-R1 result thiếu event_id"
        )

    if not isinstance(
        text,
        str,
    ):
        raise ValueError(
            f"Event {event_id!r}: text không hợp lệ"
        )

    regions = tuple(
        region_from_dict(region)
        for region in regions_data
    )

    return TRR1Result(
        event_id=event_id,
        text=text,
        relation=(
            str(relation)
            if relation is not None
            else None
        ),
        regions=regions,
    )


def group_records_by_query(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group artifact theo query_id.

    Event order được giữ theo event_index.
    """

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for record in records:
        query_id = record.get(
            "query_id"
        )

        if query_id is None:
            raise ValueError(
                "Artifact record thiếu query_id"
            )

        query_id = str(
            query_id
        )

        grouped.setdefault(
            query_id,
            [],
        ).append(record)

    for query_id, items in grouped.items():
        items.sort(
            key=lambda item: int(
                item.get(
                    "event_index",
                    0,
                )
            )
        )

    return grouped


def build_tr_r1_results(
    records: list[dict[str, Any]],
) -> tuple[TRR1Result, ...]:
    """
    Một query artifact -> tuple[TRR1Result].

    Đây chính là input production của run_trake_r2().
    """

    ordered = sorted(
        records,
        key=lambda item: int(
            item.get(
                "event_index",
                0,
            )
        ),
    )

    results = tuple(
        result_from_record(record)
        for record in ordered
    )

    if not results:
        raise ValueError(
            "Query không có TR-R1 event."
        )

    event_ids = [
        result.event_id
        for result in results
    ]

    if len(event_ids) != len(
        set(event_ids)
    ):
        raise ValueError(
            f"Trùng event_id trong query: "
            f"{event_ids}"
        )

    return results


# ============================================================================
# GT
# ============================================================================


def get_gt_events_from_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Lấy GT trực tiếp từ artifact TR-R1.

    GT trong artifact có schema:

        record["gt"] = {
            "su_kien": ...,
            "video_id": ...,
            "frame_start": ...,
            "frame_end": ...,
            "fps": ...,
            "pts_time": ...,
            "start_time": ...,
            "end_time": ...,
        }

    LƯU Ý:
    GT chỉ dùng cho evaluation.
    Không được dùng GT để chọn video/window cho TR-R2.
    """

    ordered = sorted(
        records,
        key=lambda item: int(
            item.get(
                "event_index",
                0,
            )
        ),
    )

    gt_events: list[
        dict[str, Any]
    ] = []

    for record in ordered:
        gt = record.get("gt")

        if not isinstance(
            gt,
            dict,
        ):
            raise ValueError(
                f"Query {record.get('query_id')!r}, "
                f"event {record.get('event_id')!r}: "
                "thiếu gt trong tr_r1_candidates.jsonl"
            )

        required = {
            "video_id",
            "frame_start",
            "frame_end",
            "fps",
            "pts_time",
            "start_time",
            "end_time",
        }

        missing = required - set(gt)

        if missing:
            raise ValueError(
                f"Query {record.get('query_id')!r}, "
                f"event {record.get('event_id')!r}: "
                "GT thiếu field: "
                + ", ".join(
                    sorted(missing)
                )
            )

        gt_events.append(
            {
                "event_id": record.get(
                    "event_id"
                ),
                "event_index": int(
                    record.get(
                        "event_index",
                        0,
                    )
                ),
                "su_kien": record.get(
                    "text"
                ),
                "video_id": str(
                    gt["video_id"]
                ),
                "frame_start": int(
                    gt["frame_start"]
                ),
                "frame_end": int(
                    gt["frame_end"]
                ),
                "fps": float(
                    gt["fps"]
                ),
                "pts_time": float(
                    gt["pts_time"]
                ),
                "start_time": float(
                    gt["start_time"]
                ),
                "end_time": float(
                    gt["end_time"]
                ),
            }
        )

    return gt_events


# ============================================================================
# METRICS
# ============================================================================


def temporal_iou(
    prediction_time: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """
    TR-R2 hiện trả một chosen timestamp.

    Xem prediction như một điểm thời gian và GT là interval.

    Nếu timestamp nằm trong GT:
        IoU = 1.

    Nếu ngoài GT:
        IoU = 0.
    """

    if (
        gt_start
        <= prediction_time
        <= gt_end
    ):
        return 1.0

    return 0.0


def event_prediction_error(
    prediction_time: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """
    Khoảng cách từ prediction tới GT interval.

    Nếu prediction nằm trong interval:
        error = 0.

    Nếu ở ngoài:
        khoảng cách tới boundary gần nhất.
    """

    if (
        gt_start
        <= prediction_time
        <= gt_end
    ):
        return 0.0

    if prediction_time < gt_start:
        return gt_start - prediction_time

    return prediction_time - gt_end


def sequence_is_strictly_increasing(
    chosen_times: list[float],
) -> bool:
    return all(
        left < right
        for left, right in zip(
            chosen_times,
            chosen_times[1:],
        )
    )


# ============================================================================
# PERCENTILE
# ============================================================================


def percentile(
    values: list[float],
    p: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(
        values
    )

    if len(ordered) == 1:
        return ordered[0]

    rank = (
        len(ordered) - 1
    ) * (p / 100.0)

    lower = int(rank)

    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = (
        rank - lower
    )

    return (
        ordered[lower]
        + fraction
        * (
            ordered[upper]
            - ordered[lower]
        )
    )


# ============================================================================
# BENCHMARK
# ============================================================================


def benchmark() -> dict[str, Any]:
    records = doc_jsonl(
        INPUT_PATH
    )

    grouped = group_records_by_query(
        records
    )

    if not grouped:
        raise RuntimeError(
            "TR-R1 artifact không có query."
        )

    query_results: list[
        dict[str, Any]
    ] = []

    latency_ms: list[float] = []

    total_events = 0
    video_correct_events = 0
    temporal_hit_events = 0

    temporal_errors: list[float] = []
    ious: list[float] = []

    sequence_valid_queries = 0

    total_queries = len(
        grouped
    )

    for query_index, (
        query_id,
        records_for_query,
    ) in enumerate(
        sorted(
            grouped.items(),
            key=lambda item: item[0],
        ),
        start=1,
    ):
        print(
            f"[TR-R2] "
            f"{query_index}/{total_queries} "
            f"query={query_id}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # GT lấy từ artifact.
        #
        # Chỉ dùng cho evaluation.
        # ------------------------------------------------------------------

        gt_events = get_gt_events_from_records(
            records_for_query
        )

        if not gt_events:
            raise ValueError(
                f"Query {query_id!r} không có GT event."
            )

        gt_video_id = str(
            gt_events[0]["video_id"]
        )

        # ------------------------------------------------------------------
        # Reconstruct TR-R1 prediction.
        # ------------------------------------------------------------------

        tr_r1_results = build_tr_r1_results(
            records_for_query
        )

        if len(tr_r1_results) != len(
            gt_events
        ):
            raise ValueError(
                f"Query {query_id!r}: "
                f"TR-R1 có {len(tr_r1_results)} events, "
                f"GT có {len(gt_events)} events"
            )

        # ------------------------------------------------------------------
        # QUAN TRỌNG:
        #
        # run_trake_r2() tự chọn video/window từ TR-R1 regions.
        #
        # Không truyền GT video/window vào đây.
        # ------------------------------------------------------------------

        started = time.perf_counter()

        result = run_trake_r2(
            tr_r1_results,
            step=STEP,
            min_gap=MIN_GAP,
            rrf_k=RRF_K,
            window_padding_seconds=(
                WINDOW_PADDING_SECONDS
            ),
            batch_size=BATCH_SIZE,
        )

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000.0

        latency_ms.append(
            elapsed
        )

        predicted_video_id = str(
            result["video_id"]
        )

        chosen_times = result.get(
            "chosen_times",
            {},
        )

        ordered_event_ids = [
            item.event_id
            for item in tr_r1_results
        ]

        ordered_chosen_times: list[
            float
        ] = []

        event_results: list[
            dict[str, Any]
        ] = []

        for (
            event_index,
            (
                event_id,
                gt,
            ),
        ) in enumerate(
            zip(
                ordered_event_ids,
                gt_events,
            )
        ):
            prediction_time_raw = (
                chosen_times.get(
                    event_id
                )
            )

            if prediction_time_raw is None:
                raise ValueError(
                    f"Query {query_id!r}, "
                    f"event {event_id!r}: "
                    "TR-R2 không trả chosen_time"
                )

            prediction_time = float(
                prediction_time_raw
            )

            ordered_chosen_times.append(
                prediction_time
            )

            video_correct = (
                predicted_video_id
                == gt["video_id"]
            )

            error = event_prediction_error(
                prediction_time,
                gt["start_time"],
                gt["end_time"],
            )

            iou = temporal_iou(
                prediction_time,
                gt["start_time"],
                gt["end_time"],
            )

            temporal_hit = (
                error == 0.0
            )

            total_events += 1

            if video_correct:
                video_correct_events += 1

            if temporal_hit:
                temporal_hit_events += 1

            temporal_errors.append(
                error
            )

            ious.append(
                iou
            )

            event_results.append(
                {
                    "event_id": event_id,
                    "event_index": event_index,
                    "text": tr_r1_results[
                        event_index
                    ].text,
                    "gt": {
                        "video_id": gt[
                            "video_id"
                        ],
                        "frame_start": gt[
                            "frame_start"
                        ],
                        "frame_end": gt[
                            "frame_end"
                        ],
                        "fps": gt[
                            "fps"
                        ],
                        "pts_time": gt[
                            "pts_time"
                        ],
                        "start_time": gt[
                            "start_time"
                        ],
                        "end_time": gt[
                            "end_time"
                        ],
                    },
                    "prediction": {
                        "video_id": predicted_video_id,
                        "pts_time": prediction_time,
                    },
                    "video_match": video_correct,
                    "temporal_error_seconds": error,
                    "temporal_iou": iou,
                }
            )

        sequence_valid = (
            sequence_is_strictly_increasing(
                ordered_chosen_times
            )
        )

        if sequence_valid:
            sequence_valid_queries += 1

        query_results.append(
            {
                "query_id": query_id,
                "gt_video_id": gt_video_id,
                "predicted_video_id": predicted_video_id,
                "video_match": (
                    predicted_video_id
                    == gt_video_id
                ),
                "window": result.get(
                    "window"
                ),
                "dense_frame_count": len(
                    result.get(
                        "dense_frame_times",
                        [],
                    )
                ),
                "chosen_times": {
                    event_id: float(
                        chosen_times[event_id]
                    )
                    for event_id in ordered_event_ids
                },
                "total_score": float(
                    result.get(
                        "total_score",
                        0.0,
                    )
                ),
                "sequence_valid": sequence_valid,
                "latency_ms": elapsed,
                "events": event_results,
            }
        )

    return {
        "task": "TR-R2",
        "description": (
            "Dense CLIP-L temporal alignment "
            "with strict-increasing DP"
        ),
        "input_artifact": str(
            INPUT_PATH
        ),
        "data_root": str(
            DATA_ROOT
        ),
        "config": {
            "step": STEP,
            "min_gap": MIN_GAP,
            "rrf_k": RRF_K,
            "window_padding_seconds": (
                WINDOW_PADDING_SECONDS
            ),
            "batch_size": BATCH_SIZE,
        },
        "evaluation": {
            "note": (
                "GT lấy từ tr_r1_candidates.jsonl "
                "và chỉ dùng cho evaluation. "
                "Video/window của TR-R2 được chọn "
                "hoàn toàn từ TR-R1 output."
            ),
            "temporal_hit_definition": (
                "Chosen timestamp nằm trong GT interval."
            ),
            "temporal_error_definition": (
                "Khoảng cách từ chosen timestamp tới "
                "GT interval; bằng 0 nếu nằm trong interval."
            ),
            "temporal_iou_definition": (
                "1 nếu chosen timestamp nằm trong GT interval, "
                "0 nếu nằm ngoài."
            ),
        },
        "summary": {
            "num_queries": len(
                query_results
            ),
            "num_events": total_events,
            "video_match_events": (
                video_correct_events
            ),
            "video_accuracy": (
                video_correct_events
                / total_events
                if total_events
                else 0.0
            ),
            "temporal_hit_events": (
                temporal_hit_events
            ),
            "temporal_hit_rate": (
                temporal_hit_events
                / total_events
                if total_events
                else 0.0
            ),
            "mean_temporal_error_seconds": (
                statistics.mean(
                    temporal_errors
                )
                if temporal_errors
                else 0.0
            ),
            "median_temporal_error_seconds": (
                statistics.median(
                    temporal_errors
                )
                if temporal_errors
                else 0.0
            ),
            "mean_temporal_iou": (
                statistics.mean(
                    ious
                )
                if ious
                else 0.0
            ),
            "sequence_valid_queries": (
                sequence_valid_queries
            ),
            "sequence_valid_rate": (
                sequence_valid_queries
                / len(query_results)
                if query_results
                else 0.0
            ),
            "latency_ms": {
                "p50": percentile(
                    latency_ms,
                    50.0,
                ),
                "p95": percentile(
                    latency_ms,
                    95.0,
                ),
                "max": (
                    max(latency_ms)
                    if latency_ms
                    else 0.0
                ),
            },
        },
        "queries": query_results,
    }


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    print("=" * 72)
    print(
        "TR-R2 — DENSE CLIP-L TEMPORAL ALIGNMENT"
    )
    print("=" * 72)

    print(
        f"DATA_ROOT : {DATA_ROOT}"
    )

    print(
        f"INPUT     : {INPUT_PATH}"
    )

    print(
        f"OUTPUT    : {OUTPUT_PATH}"
    )

    print()

    result = benchmark()

    ghi_json(
        OUTPUT_PATH,
        result,
    )

    summary = result[
        "summary"
    ]

    print()
    print("=" * 72)
    print("TR-R2 RESULT")
    print("=" * 72)

    print(
        f"Queries       : "
        f"{summary['num_queries']}"
    )

    print(
        f"Events        : "
        f"{summary['num_events']}"
    )

    print(
        f"Video accuracy: "
        f"{summary['video_accuracy']:.4f}"
    )

    print(
        f"Temporal hit  : "
        f"{summary['temporal_hit_rate']:.4f}"
    )

    print(
        f"Mean error    : "
        f"{summary['mean_temporal_error_seconds']:.4f} s"
    )

    print(
        f"Median error  : "
        f"{summary['median_temporal_error_seconds']:.4f} s"
    )

    print(
        f"Mean IoU      : "
        f"{summary['mean_temporal_iou']:.4f}"
    )

    print(
        f"Sequence valid: "
        f"{summary['sequence_valid_rate']:.4f}"
    )

    print(
        f"Latency p50   : "
        f"{summary['latency_ms']['p50']:.2f} ms"
    )

    print(
        f"Latency p95   : "
        f"{summary['latency_ms']['p95']:.2f} ms"
    )

    print(
        f"Latency max   : "
        f"{summary['latency_ms']['max']:.2f} ms"
    )

    print()

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()