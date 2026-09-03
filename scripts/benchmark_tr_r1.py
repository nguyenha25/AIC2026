"""
TR-R1 — BENCHMARK TÌM VÙNG THỜI GIAN THÔ BẰNG CLIP-L

PHẠM VI
-------
Chỉ benchmark TR-R1:

    câu TRAKE
        ↓
    event text từ cac_giai_doan
        ↓
    RuleBasedParser.parse_trake() [TR-S1 QueryPlan hiện có]
        ↓
    TrakeEvent.text
        ↓
    CLIP-L Top-K
        ↓
    coarse temporal regions
        ↓
    lưu raw result + GT interval

KHÔNG LÀM TRONG FILE NÀY
------------------------
- dense extraction
- frames 0.16 s
- local dense search
- DP
- temporal smoothing
- chọn frame đại diện
- submission
- quyết định acceptance PASS/FAIL

R1 chỉ trả:
    video_id
    start_time
    end_time
    score
    hits

GT chỉ dùng để đánh giá.

Acceptance nội bộ của task được chấm riêng bởi:
    scripts/tr_r1_acceptance.py
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------

import os

# ---------------------------------------------------------------------------
# Windows native-runtime stability:
# open_clip must be imported BEFORE the semantic / Marian / Transformers
# stack initializes, otherwise sentencepiece can crash with 0xC0000005.
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# IMPORTANT: import the module early, but DO NOT create the model here.
import open_clip

# ---------------------------------------------------------------------------
# NORMAL IMPORTS
# ---------------------------------------------------------------------------

import json
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd

from aic2026.paths import DATA_ROOT, DEV_DIR, RUNS_DIR
from aic2026.semantic.parser import RuleBasedParser
from aic2026.semantic.schema import QueryPlanTRAKE
from aic2026.trake_retrieval import (
    TRR1Config,
    TRR1Result,
    tim_nhieu_su_kien,
    trr1_result_to_dict,
)


# ---------------------------------------------------------------------------
# PATHS / CONFIG
# ---------------------------------------------------------------------------

DEV_QUESTIONS = DEV_DIR / "dev_questions.jsonl"

FRAME_MAP_PATH = DATA_ROOT / "index" / "frame_map.parquet"

OUTPUT_PATH = RUNS_DIR / "tr_r1_benchmark.json"

TOP_K = 500

MAX_REGION_DURATION_SECONDS = 10.0
REGION_MERGE_GAP_SECONDS = 2.0
MIN_REGION_DURATION_SECONDS = 0.5
MAX_REGIONS_PER_EVENT = 10
VIDEO_CONSENSUS_WEIGHT = 0.45
VIDEO_RRF_K = 60.0


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def doc_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy dev questions: {path}"
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


# ---------------------------------------------------------------------------
# FRAME MAP
# ---------------------------------------------------------------------------


def load_frame_map() -> pd.DataFrame:
    if not FRAME_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Không thấy frame_map: {FRAME_MAP_PATH}"
        )

    frame_map = pd.read_parquet(
        FRAME_MAP_PATH
    )

    required = {
        "video_id",
        "n",
        "pts_time",
        "fps",
        "frame_idx",
    }

    missing = required - set(
        frame_map.columns
    )

    if missing:
        raise ValueError(
            "frame_map thiếu cột: "
            + ", ".join(
                sorted(missing)
            )
        )

    return frame_map


def build_frame_lookup(
    frame_map: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """
    Lookup:

        (video_id, frame_idx) -> pts_time

    Dùng frame_idx vì đó là ID frame thật.

    Không dùng n làm submission ID.

    Lookup này hiện chỉ được giữ để diagnostic / tương thích
    với benchmark cũ. GT interval mới được tính từ frame_start,
    frame_end và FPS của video.
    """

    lookup: dict[
        tuple[str, int],
        float,
    ] = {}

    for row in frame_map.itertuples(
        index=False
    ):
        key = (
            str(row.video_id),
            int(row.frame_idx),
        )

        lookup[key] = float(
            row.pts_time
        )

    return lookup


def build_video_fps_lookup(
    frame_map: pd.DataFrame,
) -> dict[str, float]:
    """
    Lookup:

        video_id -> fps

    FPS được lấy từ frame_map.

    Mỗi video phải có một FPS hợp lệ và nhất quán.
    """

    fps_lookup: dict[str, float] = {}

    for row in frame_map.itertuples(
        index=False
    ):
        video_id = str(
            row.video_id
        )

        try:
            fps = float(
                row.fps
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"FPS không hợp lệ cho "
                f"video_id={video_id!r}: "
                f"{row.fps!r}"
            ) from exc

        if fps <= 0.0:
            raise ValueError(
                f"FPS phải > 0 cho "
                f"video_id={video_id!r}: "
                f"{fps}"
            )

        previous = fps_lookup.get(
            video_id
        )

        if previous is None:
            fps_lookup[video_id] = fps
            continue

        # FPS của cùng video phải nhất quán.
        # Cho phép sai số floating-point rất nhỏ.
        if abs(previous - fps) > 1e-6:
            raise ValueError(
                f"Video {video_id!r} có FPS "
                f"không nhất quán trong frame_map: "
                f"{previous} vs {fps}"
            )

    return fps_lookup


# ---------------------------------------------------------------------------
# FRAME / VIDEO HELPERS
# ---------------------------------------------------------------------------


def get_gt_video_id(
    row: dict[str, Any],
) -> str:
    video_id = row.get(
        "video_id"
    )

    if (
        not isinstance(video_id, str)
        or not video_id.strip()
    ):
        raise ValueError(
            f"Query {row.get('id')!r} thiếu video_id"
        )

    return video_id


def frame_to_time(
    frame_lookup: dict[tuple[str, int], float],
    video_id: str,
    frame_idx: int,
) -> float:
    key = (
        video_id,
        int(frame_idx),
    )

    if key not in frame_lookup:
        raise ValueError(
            f"Không tìm thấy frame_map cho "
            f"video_id={video_id!r}, "
            f"frame_idx={frame_idx}"
        )

    return float(
        frame_lookup[key]
    )


# ---------------------------------------------------------------------------
# TR-S1 -> QUERYPLAN
# ---------------------------------------------------------------------------


def build_queryplan(
    gt_stages: list[dict[str, Any]],
    parser: RuleBasedParser,
):
    """
    Build QueryPlan cho benchmark TR-R1 từ các event
    được khai báo trong cac_giai_doan.

    Không parse toàn bộ cau_hoi vì cau_hoi TRAKE có thể chứa:
        - context mở đầu video
        - nhãn E1/E2/E3
        - phần giải thích trong ngoặc

    Các thành phần đó có thể khiến parser tạo event phụ và làm
    lệch số event so với GT.

    Ở đây chỉ dùng text của từng cac_giai_doan làm event input
    cho parser TR-S1.
    """

    su_kien = [
        str(stage["su_kien"])
        for stage in gt_stages
    ]

    if not su_kien:
        raise ValueError(
            "Không có event để build QueryPlan"
        )

    # parse_trake() yêu cầu:
    #     parse_trake(query, events)
    #
    # query chỉ dùng làm metadata/context của QueryPlan;
    # events là danh sách event đã xác định từ cac_giai_doan.
    plan = parser.parse_trake(
        "",
        su_kien,
    )

    return plan


# ---------------------------------------------------------------------------
# GT — TRAKE
# ---------------------------------------------------------------------------


def get_gt_stages(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Đọc GT TRAKE từ:

        row["cac_giai_doan"]

    Mỗi stage phải có:
        su_kien
        frame_start
        frame_end
        pts_time

    GT được giữ theo đúng thứ tự event.
    """

    stages = row.get(
        "cac_giai_doan"
    )

    if (
        not isinstance(stages, list)
        or not stages
    ):
        raise ValueError(
            f"Query {row.get('id')!r} thiếu "
            "cac_giai_doan"
        )

    normalized: list[
        dict[str, Any]
    ] = []

    for index, stage in enumerate(
        stages,
        start=1,
    ):
        if not isinstance(
            stage,
            dict,
        ):
            raise ValueError(
                f"Query {row.get('id')!r}, "
                f"stage {index}: phải là object"
            )

        required = {
            "su_kien",
            "frame_start",
            "frame_end",
            "pts_time",
        }

        missing = (
            required - set(stage)
        )

        if missing:
            raise ValueError(
                f"Query {row.get('id')!r}, "
                f"stage {index}: thiếu "
                + ", ".join(
                    sorted(missing)
                )
            )

        try:
            frame_start = int(
                stage["frame_start"]
            )

            frame_end = int(
                stage["frame_end"]
            )

            pts_time = float(
                stage["pts_time"]
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Query {row.get('id')!r}, "
                f"stage {index}: "
                "GT frame/time không hợp lệ"
            ) from exc

        if frame_start > frame_end:
            frame_start, frame_end = (
                frame_end,
                frame_start,
            )

        normalized.append(
            {
                "event_index": index - 1,
                "su_kien": str(
                    stage["su_kien"]
                ),
                "frame_start": frame_start,
                "frame_end": frame_end,
                "pts_time": pts_time,
            }
        )

    return normalized


def get_gt_time_interval(
    row: dict[str, Any],
    stage: dict[str, Any],
    video_fps_lookup: dict[str, float],
) -> tuple[str, float, float, float]:
    """
    Chuyển MỘT stage GT thành temporal interval thật.

    video_id:
        lấy từ root query.

    start_time:
        frame_start / fps

    end_time:
        (frame_end + 1) / fps vì khoảng frame là inclusive [start, end].

    pts_time:
        giữ nguyên trong output như mốc GT gốc để diagnostic.

    Quan trọng:
        Không dùng pts_time làm cả start và end.

    pts_time trong GT hiện tương ứng với thời điểm bắt đầu
    của stage; frame_start/frame_end mới xác định phạm vi
    temporal để tính IoU.
    """

    video_id = get_gt_video_id(
        row
    )

    if video_id not in video_fps_lookup:
        raise ValueError(
            f"Không tìm thấy FPS trong frame_map cho "
            f"video_id={video_id!r}"
        )

    fps = float(
        video_fps_lookup[video_id]
    )

    if fps <= 0.0:
        raise ValueError(
            f"FPS không hợp lệ cho "
            f"video_id={video_id!r}: {fps}"
        )

    try:
        frame_start = int(
            stage["frame_start"]
        )

        frame_end = int(
            stage["frame_end"]
        )

        gt_pts_time = float(
            stage["pts_time"]
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"Query {row.get('id')!r}: "
            "GT frame/time không hợp lệ"
        ) from exc

    if frame_start > frame_end:
        frame_start, frame_end = (
            frame_end,
            frame_start,
        )

    start_time = (
        frame_start / fps
    )

    end_time = (
        (frame_end + 1) / fps
    )

    if end_time < start_time:
        raise ValueError(
            f"Query {row.get('id')!r}: "
            f"GT interval không hợp lệ: "
            f"{start_time} -> {end_time}"
        )

    return (
        video_id,
        start_time,
        end_time,
        gt_pts_time,
    )


# ---------------------------------------------------------------------------
# EVENT / GT ALIGNMENT
# ---------------------------------------------------------------------------


def validate_event_gt_alignment(
    plan: QueryPlanTRAKE,
    gt_stages: list[dict[str, Any]],
) -> None:
    """
    Đảm bảo QueryPlan và GT có cùng số event.

    Không tự merge / truncate / repeat event ở benchmark.
    """

    num_events = len(
        plan.events
    )

    num_gt = len(
        gt_stages
    )

    if num_events != num_gt:
        raise ValueError(
            f"Query {plan.query_id!r}: "
            f"QueryPlan có {num_events} events nhưng "
            f"GT có {num_gt} cac_giai_doan"
        )


# ---------------------------------------------------------------------------
# REGION EVALUATION
# ---------------------------------------------------------------------------


def temporal_iou(
    region_start: float,
    region_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """
    Temporal IoU giữa coarse region và GT interval.

    Interval được xem là [start, end].
    """

    inter_start = max(
        region_start,
        gt_start,
    )

    inter_end = min(
        region_end,
        gt_end,
    )

    intersection = max(
        0.0,
        inter_end - inter_start,
    )

    union = max(
        region_end,
        gt_end,
    ) - min(
        region_start,
        gt_start,
    )

    if union <= 0.0:
        return 0.0

    return (
        intersection / union
    )


def region_iou(
    region: Any,
    *,
    gt_video_id: str,
    gt_start_time: float,
    gt_end_time: float,
) -> float:
    """
    Trả temporal IoU của một coarse region với GT.

    Nếu khác video:
        IoU = 0.
    """

    if region.video_id != gt_video_id:
        return 0.0

    return temporal_iou(
        float(region.start_time),
        float(region.end_time),
        gt_start_time,
        gt_end_time,
    )


def region_contains_gt(
    region: Any,
    *,
    gt_video_id: str,
    gt_start_time: float,
    gt_end_time: float,
) -> bool:
    """
    Helper diagnostic.

    Region được xem là cover GT nếu:
        cùng video
        +
        temporal IoU > 0.

    Acceptance nội bộ KHÔNG dùng helper này để quyết định
    PASS/FAIL; acceptance nằm ở tr_r1_acceptance.py với
    IoU >= 0.30 và Top-10.
    """

    return (
        region_iou(
            region,
            gt_video_id=gt_video_id,
            gt_start_time=gt_start_time,
            gt_end_time=gt_end_time,
        )
        > 0.0
    )


def event_is_covered(
    result: TRR1Result,
    *,
    gt_video_id: str,
    gt_start_time: float,
    gt_end_time: float,
) -> bool:
    """
    Diagnostic coverage:

        GT có temporal overlap với ít nhất một region.

    Đây KHÔNG phải acceptance nội bộ của TR-R1.

    Acceptance nội bộ được tính riêng bởi:
        scripts/tr_r1_acceptance.py
    """

    return any(
        region_contains_gt(
            region,
            gt_video_id=gt_video_id,
            gt_start_time=gt_start_time,
            gt_end_time=gt_end_time,
        )
        for region in result.regions
    )


# ---------------------------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------------------------


def benchmark() -> dict[str, Any]:
    rows = doc_jsonl(
        DEV_QUESTIONS
    )

    trake_rows = [
        row
        for row in rows
        if row.get(
            "loai_truy_van"
        ) == "chuoi_su_kien"
    ]

    if not trake_rows:
        raise RuntimeError(
            "Dev set không có query "
            "loai_truy_van='chuoi_su_kien'."
        )

    frame_map = load_frame_map()

    frame_lookup = build_frame_lookup(
        frame_map
    )

    video_fps_lookup = (
        build_video_fps_lookup(
            frame_map
        )
    )

    parser = RuleBasedParser()

    config = TRR1Config(
        top_k=TOP_K,
        max_region_duration_seconds=(
            MAX_REGION_DURATION_SECONDS
        ),
        region_merge_gap_seconds=(
            REGION_MERGE_GAP_SECONDS
        ),
        min_region_duration_seconds=(
            MIN_REGION_DURATION_SECONDS
        ),
        max_regions_per_event=(
            MAX_REGIONS_PER_EVENT
        ),
        video_consensus_weight=(
            VIDEO_CONSENSUS_WEIGHT
        ),
        video_rrf_k=VIDEO_RRF_K,
    )

    query_results: list[
        dict[str, Any]
    ] = []

    total_events = 0
    overlap_events = 0
    video_match_events = 0
    candidate_frame_events = 0

    latency_ms: list[float] = []

    for row in trake_rows:
        query_id = str(
            row["id"]
        )

        print(
            f"[TR-R1] query={query_id}",
            flush=True,
        )

        # ---------------------------------------------------------------
        # GT chỉ dùng để xác định danh sách event benchmark.
        # ---------------------------------------------------------------

        gt_stages = get_gt_stages(
            row
        )

        plan = build_queryplan(
            gt_stages,
            parser,
        )

        validate_event_gt_alignment(
            plan,
            gt_stages,
        )

        started = time.perf_counter()

        results = tim_nhieu_su_kien(
            [
                {
                    "event_id": event.event_id,
                    "text": event.text,
                    "relation": event.relation,
                }
                for event in plan.events
            ],
            config=config,
        )

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000.0

        latency_ms.append(
            elapsed
        )

        if len(results) != len(
            plan.events
        ):
            raise ValueError(
                f"Query {query_id!r}: "
                f"TR-R1 trả {len(results)} results "
                f"cho {len(plan.events)} events"
            )

        event_results: list[
            dict[str, Any]
        ] = []

        for (
            event_index,
            (event, result, stage),
        ) in enumerate(
            zip(
                plan.events,
                results,
                gt_stages,
            )
        ):
            (
                gt_video_id,
                gt_start,
                gt_end,
                gt_pts_time,
            ) = get_gt_time_interval(
                row,
                stage,
                video_fps_lookup,
            )

            overlap = event_is_covered(
                result,
                gt_video_id=gt_video_id,
                gt_start_time=gt_start,
                gt_end_time=gt_end,
            )

            total_events += 1

            video_match = any(
                region.video_id == gt_video_id
                for region in result.regions
            )
            if video_match:
                video_match_events += 1

            candidate_frame_hit = any(
                region.video_id == gt_video_id
                and any(
                    hit.get("frame_idx") is not None
                    and int(stage["frame_start"])
                    <= int(hit["frame_idx"])
                    <= int(stage["frame_end"])
                    for hit in region.hits
                )
                for region in result.regions
            )
            if candidate_frame_hit:
                candidate_frame_events += 1

            if overlap:
                overlap_events += 1

            # -----------------------------------------------------------
            # Diagnostic: rank tốt nhất của GT region trong toàn bộ
            # coarse regions.
            #
            # Không dùng để quyết định acceptance.
            # -----------------------------------------------------------

            region_diagnostics: list[
                dict[str, Any]
            ] = []

            for region_rank, region in enumerate(
                result.regions,
                start=1,
            ):
                iou = region_iou(
                    region,
                    gt_video_id=gt_video_id,
                    gt_start_time=gt_start,
                    gt_end_time=gt_end,
                )

                region_diagnostics.append(
                    {
                        "rank": region_rank,
                        "video_id": region.video_id,
                        "start_time": region.start_time,
                        "end_time": region.end_time,
                        "score": region.score,
                        "iou": iou,
                    }
                )

            positive_regions = [
                item
                for item in region_diagnostics
                if item["iou"] > 0.0
            ]

            best_overlap_rank = (
                min(
                    item["rank"]
                    for item in positive_regions
                )
                if positive_regions
                else None
            )

            best_iou = (
                max(
                    item["iou"]
                    for item in region_diagnostics
                )
                if region_diagnostics
                else 0.0
            )

            # -----------------------------------------------------------
            # Lưu toàn bộ coarse regions theo thứ tự score do
            # trake_retrieval trả về.
            #
            # Acceptance scorer nội bộ sẽ tự lấy Top-10.
            # -----------------------------------------------------------

            serialized_result = (
                trr1_result_to_dict(
                    result
                )
            )

            event_results.append(
                {
                    "event_id": event.event_id,
                    "event_index": event_index,
                    "text": event.text,
                    "relation": event.relation,
                    "entities": list(
                        event.entities
                    ),
                    "actions": list(
                        event.actions
                    ),
                    "gt": {
                        "su_kien": stage[
                            "su_kien"
                        ],
                        "video_id": gt_video_id,
                        "frame_start": stage[
                            "frame_start"
                        ],
                        "frame_end": stage[
                            "frame_end"
                        ],
                        "fps": video_fps_lookup[
                            gt_video_id
                        ],
                        "pts_time": gt_pts_time,
                        "start_time": gt_start,
                        "end_time": gt_end,
                    },
                    "overlap": overlap,
                    "video_match": video_match,
                    "candidate_frame_hit": candidate_frame_hit,
                    "best_iou": best_iou,
                    "best_overlap_rank": best_overlap_rank,
                    "region_diagnostics": region_diagnostics,
                    "result": serialized_result,
                }
            )

        query_results.append(
            {
                "query_id": query_id,
                "video_id": get_gt_video_id(
                    row
                ),
                "latency_ms": elapsed,
                "num_events": len(
                    plan.events
                ),
                "events": event_results,
            }
        )

    overlap_recall = (
        overlap_events / total_events
        if total_events
        else 0.0
    )

    return {
        "task": "TR-R1",
        "description": (
            "Tìm video và timestamp thô bằng CLIP-L"
        ),
        "data_root": str(
            DATA_ROOT
        ),
        "dev_questions": str(
            DEV_QUESTIONS
        ),
        "frame_map": str(
            FRAME_MAP_PATH
        ),
        "config": {
            "top_k": config.top_k,
            "max_region_duration_seconds": (
                config.max_region_duration_seconds
            ),
            "region_merge_gap_seconds": (
                config.region_merge_gap_seconds
            ),
            "min_region_duration_seconds": (
                config.min_region_duration_seconds
            ),
            "max_regions_per_event": (
                config.max_regions_per_event
            ),
            "video_consensus_weight": (
                config.video_consensus_weight
            ),
            "video_rrf_k": config.video_rrf_k,
        },
        "evaluation": {
            "note": (
                "Benchmark lưu raw coarse regions và GT temporal "
                "interval. Acceptance nội bộ được chấm riêng "
                "bởi scripts/tr_r1_acceptance.py."
            ),
            "gt_interval_source": (
                "frame_start/frame_end converted using per-video fps"
            ),
            "overlap_recall_definition": (
                "Diagnostic only: same video + temporal IoU > 0 "
                "with at least one returned coarse region."
            ),
            "acceptance_definition": (
                "Internal TR-R1 acceptance is evaluated separately "
                "by scripts/tr_r1_acceptance.py using IoU >= 0.30 "
                "and Top-10."
            ),
        },
        "summary": {
            "num_dev_queries": len(
                rows
            ),
            "num_trake_queries": len(
                trake_rows
            ),
            "num_events": total_events,
            "overlap_events": overlap_events,
            "overlap_recall": overlap_recall,
            "video_match_events": video_match_events,
            "video_match_recall": (
                video_match_events / total_events
                if total_events
                else 0.0
            ),
            "candidate_frame_events": candidate_frame_events,
            "candidate_frame_recall": (
                candidate_frame_events / total_events
                if total_events
                else 0.0
            ),
            "latency_ms": {
                "p50": (
                    statistics.median(
                        latency_ms
                    )
                    if latency_ms
                    else 0.0
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


# ---------------------------------------------------------------------------
# PERCENTILE
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print(
        "TR-R1 — CLIP-L COARSE TEMPORAL RETRIEVAL"
    )
    print("=" * 72)

    print(
        f"DATA_ROOT : {DATA_ROOT}"
    )

    print(
        f"DEV       : {DEV_QUESTIONS}"
    )

    print(
        f"FRAME_MAP : {FRAME_MAP_PATH}"
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
    print("TR-R1 RESULT")
    print("=" * 72)

    print(
        f"TRAKE queries : "
        f"{summary['num_trake_queries']}"
    )

    print(
        f"Events        : "
        f"{summary['num_events']}"
    )

    print(
        f"Overlap       : "
        f"{summary['overlap_events']}"
    )

    print(
        f"Overlap recall: "
        f"{summary['overlap_recall']:.4f}"
    )

    print(
        f"Video recall  : "
        f"{summary['video_match_recall']:.4f}"
    )

    print(
        f"Frame recall  : "
        f"{summary['candidate_frame_recall']:.4f}"
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
