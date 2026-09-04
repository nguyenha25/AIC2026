"""
TR-E2 — chạy boundary refinement trên output TR-R2.

Luồng:

dev_questions.jsonl
    -> lấy câu TRAKE
    -> RuleBasedParser
    -> TR-R1 coarse retrieval
    -> TR-R2 diagnostics
    -> DenseClipLScorer
    -> TR-E2 boundary refinement
    -> in kết quả JSON

Lưu ý:
- cac_giai_doan[].su_kien chỉ được dùng để lấy text event,
  giống benchmark TR-R1 hiện tại.
- frame_start / frame_end / pts_time KHÔNG được đưa vào inference.
- Không dùng GT boundary để refine.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# WINDOWS NATIVE RUNTIME STABILITY
# ---------------------------------------------------------------------------

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
    "python",
)

# IMPORTANT:
# open_clip phải được import sớm giống benchmark_tr_r1.py
import open_clip  # noqa: F401


# ---------------------------------------------------------------------------
# NORMAL IMPORTS
# ---------------------------------------------------------------------------

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aic2026.paths import DEV_DIR
from aic2026.semantic.parser import RuleBasedParser
from aic2026.trake_e2_boundary import (
    refine_from_dense_scorer,
)
from aic2026.trake_r2_pipeline import (
    run_trake_r2_diagnostics,
)
from aic2026.trake_retrieval import (
    TRR1Config,
    tim_nhieu_su_kien,
)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DEV_QUESTIONS = DEV_DIR / "dev_questions.jsonl"

TOP_K = 500

MAX_REGION_DURATION_SECONDS = 10.0
REGION_MERGE_GAP_SECONDS = 2.0
MIN_REGION_DURATION_SECONDS = 0.5
MAX_REGIONS_PER_EVENT = 10
VIDEO_CONSENSUS_WEIGHT = 0.45
VIDEO_RRF_K = 60.0

TR_R2_STEP = 0.16
TR_R2_MIN_GAP = 1
TR_R2_RRF_K = 60
TR_R2_WINDOW_PADDING_SECONDS = 0.0
TR_R2_BATCH_SIZE = 16


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


# ---------------------------------------------------------------------------
# QUERY / EVENT
# ---------------------------------------------------------------------------


def get_gt_stages(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Chỉ chuẩn hóa cac_giai_doan.

    Trong inference TR-E2:
    - chỉ su_kien được dùng làm event text
    - frame_start/frame_end/pts_time chỉ được validate,
      không đưa vào TR-R1/TR-R2/TR-E2.
    """

    stages = row.get("cac_giai_doan")

    if not isinstance(stages, list) or not stages:
        raise ValueError(
            f"Query {row.get('id')!r} thiếu cac_giai_doan"
        )

    normalized: list[dict[str, Any]] = []

    for index, stage in enumerate(
        stages,
        start=1,
    ):
        if not isinstance(stage, dict):
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

        missing = required - set(stage)

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


def build_queryplan(
    stages: list[dict[str, Any]],
    parser: RuleBasedParser,
):
    event_texts = [
        str(stage["su_kien"])
        for stage in stages
    ]

    if not event_texts:
        raise ValueError(
            "Không có event để build QueryPlan"
        )

    return parser.parse_trake(
        "",
        event_texts,
    )


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def get_chosen_times(
    alignment: dict[str, Any],
) -> dict[str, float]:
    """
    Chuẩn hóa chosen_times từ output TR-R2.

    run_trake_r2_diagnostics hiện trả alignment chứa:
        chosen_times

    Có thể chosen_times đã là dict.
    """

    chosen_times = alignment.get(
        "chosen_times"
    )

    if not isinstance(
        chosen_times,
        dict,
    ):
        raise ValueError(
            "TR-R2 diagnostics không có "
            "alignment['chosen_times'] dạng dict"
        )

    return {
        str(event_id): float(value)
        for event_id, value
        in chosen_times.items()
    }


def enforce_boundary_order(
    boundaries,
    *,
    scorer,
    chosen_times: dict[str, float],
    continuity_gap_seconds: float = 0.50,
):
    """
    Joint strict-order correction cho toàn bộ event.

    Không sửa greedily từng event.

    Mỗi event:
    - chỉ được chọn frame trong temporal component chứa TR-R2 anchor;
    - ưu tiên gần representative TR-E2;
    - đồng thời ưu tiên gần TR-R2 chosen_time.

    Sau đó dùng dynamic programming để tìm toàn bộ chuỗi
    representative_pos tăng nghiêm ngặt.
    """

    from dataclasses import replace
    import math

    frames = list(scorer.frames)

    if not frames:
        raise ValueError(
            "TR-E2 không có dense frame để enforce order"
        )

    if not boundaries:
        return []

    frame_times = [
        float(frame.pts_time)
        for frame in frames
    ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def nearest_position(
        target_time: float,
    ) -> int:
        return min(
            range(len(frames)),
            key=lambda pos: (
                abs(
                    frame_times[pos]
                    - target_time
                ),
                pos,
            ),
        )

    def temporal_component(
        anchor_pos: int,
    ) -> tuple[int, int]:
        left = anchor_pos
        right = anchor_pos

        while left > 0:
            gap = (
                frame_times[left]
                - frame_times[left - 1]
            )

            if (
                gap <= 0.0
                or gap > continuity_gap_seconds
            ):
                break

            left -= 1

        while right + 1 < len(frames):
            gap = (
                frame_times[right + 1]
                - frame_times[right]
            )

            if (
                gap <= 0.0
                or gap > continuity_gap_seconds
            ):
                break

            right += 1

        return left, right

    # ------------------------------------------------------------------
    # Candidate frame cho từng event
    # ------------------------------------------------------------------

    candidate_lists: list[list[int]] = []
    anchor_times: list[float] = []

    for result in boundaries:
        event_id = str(
            result.event_id
        )

        if event_id not in chosen_times:
            raise ValueError(
                f"Thiếu TR-R2 chosen_time "
                f"cho event {event_id!r}"
            )

        anchor_time = float(
            chosen_times[event_id]
        )

        anchor_times.append(
            anchor_time
        )

        anchor_pos = nearest_position(
            anchor_time
        )

        left, right = temporal_component(
            anchor_pos
        )

        candidates = list(
            range(
                left,
                right + 1,
            )
        )

        if not candidates:
            raise ValueError(
                f"Event {event_id!r}: "
                "không có candidate dense frame"
            )

        candidate_lists.append(
            candidates
        )

    # ------------------------------------------------------------------
    # Cost:
    #
    # ưu tiên representative TR-E2,
    # nhưng vẫn neo vào TR-R2 anchor để tránh drift quá xa.
    # ------------------------------------------------------------------

    def candidate_cost(
        event_index: int,
        pos: int,
    ) -> float:
        result = boundaries[
            event_index
        ]

        t = frame_times[pos]

        refined_time = float(
            result.representative_time
        )

        anchor_time = anchor_times[
            event_index
        ]

        refined_distance = abs(
            t - refined_time
        )

        anchor_distance = abs(
            t - anchor_time
        )

        return (
            refined_distance
            + 0.50 * anchor_distance
        )

    # ------------------------------------------------------------------
    # Dynamic programming strict increasing positions
    # ------------------------------------------------------------------

    n_events = len(
        boundaries
    )

    dp: list[dict[int, float]] = [
        {}
        for _ in range(n_events)
    ]

    back: list[dict[int, int | None]] = [
        {}
        for _ in range(n_events)
    ]

    # Event đầu tiên.
    for pos in candidate_lists[0]:
        dp[0][pos] = candidate_cost(
            0,
            pos,
        )

        back[0][pos] = None

    # Các event tiếp theo.
    for event_index in range(
        1,
        n_events,
    ):
        previous_items = sorted(
            dp[event_index - 1].items()
        )

        if not previous_items:
            raise ValueError(
                "TR-E2 DP không còn state "
                f"ở event {event_index}"
            )

        # Prefix-best cho mọi previous_pos < current_pos.
        best_cost = math.inf
        best_previous_pos: int | None = None
        previous_pointer = 0

        for current_pos in candidate_lists[
            event_index
        ]:
            while (
                previous_pointer
                < len(previous_items)
                and previous_items[
                    previous_pointer
                ][0] < current_pos
            ):
                previous_pos, previous_cost = (
                    previous_items[
                        previous_pointer
                    ]
                )

                if previous_cost < best_cost:
                    best_cost = previous_cost
                    best_previous_pos = (
                        previous_pos
                    )

                previous_pointer += 1

            if best_previous_pos is None:
                continue

            dp[event_index][
                current_pos
            ] = (
                best_cost
                + candidate_cost(
                    event_index,
                    current_pos,
                )
            )

            back[event_index][
                current_pos
            ] = best_previous_pos

        if not dp[event_index]:
            event_id = boundaries[
                event_index
            ].event_id

            raise ValueError(
                "TR-E2 joint DP không tìm được "
                "chuỗi strict-order cho "
                f"event {event_id!r}"
            )

    # ------------------------------------------------------------------
    # Backtrack
    # ------------------------------------------------------------------

    last_pos = min(
        dp[-1],
        key=lambda pos: dp[-1][pos],
    )

    chosen_positions = [
        0
        for _ in range(n_events)
    ]

    chosen_positions[-1] = (
        last_pos
    )

    for event_index in range(
        n_events - 1,
        0,
        -1,
    ):
        previous_pos = back[
            event_index
        ][chosen_positions[event_index]]

        if previous_pos is None:
            raise ValueError(
                "TR-E2 DP backtrack bị lỗi"
            )

        chosen_positions[
            event_index - 1
        ] = previous_pos

    # ------------------------------------------------------------------
    # Tạo boundary kết quả
    # ------------------------------------------------------------------

    fixed = []

    for event_index, result in enumerate(
        boundaries
    ):
        candidate_pos = (
            chosen_positions[
                event_index
            ]
        )

        old_pos = int(
            result.representative_pos
        )

        # Không đổi thì giữ nguyên toàn bộ result.
        if candidate_pos == old_pos:
            fixed.append(
                result
            )
            continue

        frame = frames[
            candidate_pos
        ]

        new_start_pos = min(
            int(result.start_pos),
            candidate_pos,
        )

        new_end_pos = max(
            int(result.end_pos),
            candidate_pos,
        )

        start_frame = frames[
            new_start_pos
        ]

        end_frame = frames[
            new_end_pos
        ]

        corrected = replace(
            result,

            start_pos=new_start_pos,
            end_pos=new_end_pos,

            start_frame_idx=int(
                start_frame.frame_idx
            ),
            end_frame_idx=int(
                end_frame.frame_idx
            ),

            start_time=float(
                start_frame.pts_time
            ),
            end_time=float(
                end_frame.pts_time
            ),

            representative_pos=(
                candidate_pos
            ),

            representative_frame_idx=int(
                frame.frame_idx
            ),

            representative_time=float(
                frame.pts_time
            ),

            status="order_adjusted",
        )

        fixed.append(
            corrected
        )

    # ------------------------------------------------------------------
    # Final invariants
    # ------------------------------------------------------------------

    previous_pos: int | None = None
    previous_time: float | None = None

    for result in fixed:
        if not (
            result.start_pos
            <= result.representative_pos
            <= result.end_pos
        ):
            raise ValueError(
                f"Event {result.event_id!r}: "
                "representative_pos nằm ngoài boundary"
            )

        if not (
            result.start_time
            <= result.representative_time
            <= result.end_time
        ):
            raise ValueError(
                f"Event {result.event_id!r}: "
                "representative_time nằm ngoài boundary"
            )

        current_pos = int(
            result.representative_pos
        )

        current_time = float(
            result.representative_time
        )

        if (
            previous_pos is not None
            and current_pos <= previous_pos
        ):
            raise ValueError(
                "TR-E2 vẫn vi phạm strict "
                "representative_pos order"
            )

        if (
            previous_time is not None
            and current_time <= previous_time
        ):
            raise ValueError(
                "TR-E2 vẫn vi phạm strict time order: "
                f"{previous_time:.3f} -> "
                f"{current_time:.3f}"
            )

        previous_pos = current_pos
        previous_time = current_time

    return fixed


# ---------------------------------------------------------------------------
# RUN ONE QUERY
# ---------------------------------------------------------------------------


def run_one_query(
    *,
    row: dict[str, Any],
    parser_s1: RuleBasedParser,
    trr1_config: TRR1Config,
) -> dict[str, Any]:
    query_id = str(
        row.get("id", "")
    )


    stages = get_gt_stages(
        row
    )

    plan = build_queryplan(
        stages,
        parser_s1,
    )

    event_ids = [
        str(event.event_id)
        for event in plan.events
    ]

    if not event_ids:
        raise ValueError(
            f"Query {query_id!r}: "
            "QueryPlan không có event"
        )

    print()
    print("=" * 70)
    print(
        f"QUERY {query_id} "
        f"| {len(event_ids)} event"
    )
    print("=" * 70)

    for event in plan.events:
        print(
            f"{event.event_id} | "
            f"{event.relation} | "
            f"{event.text}"
        )

    # ------------------------------------------------------------------
    # TR-R1
    # ------------------------------------------------------------------

    print()
    print("[1/3] TR-R1 coarse retrieval...")

    tr_r1_results = tim_nhieu_su_kien(
        [
            {
                "event_id": event.event_id,
                "text": event.text,
                "relation": event.relation,
            }
            for event in plan.events
        ],
        config=trr1_config,
    )

    if not tr_r1_results:
        raise ValueError(
            f"Query {query_id!r}: "
            "TR-R1 không trả kết quả"
        )

    print(
        f"TR-R1 OK: "
        f"{len(tr_r1_results)} event"
    )

    # ------------------------------------------------------------------
    # TR-R2
    # ------------------------------------------------------------------

    print()
    print("[2/3] TR-R2 dense alignment...")

    diagnostics = (
        run_trake_r2_diagnostics(
            tr_r1_results,
            step=TR_R2_STEP,
            min_gap=TR_R2_MIN_GAP,
            rrf_k=TR_R2_RRF_K,
            window_padding_seconds=(
                TR_R2_WINDOW_PADDING_SECONDS
            ),
            batch_size=TR_R2_BATCH_SIZE,
        )
    )

    scorer = diagnostics.get(
        "scorer"
    )
    alignment = diagnostics.get(
        "alignment"
    )

    if scorer is None:
        raise ValueError(
            f"Query {query_id!r}: "
            "TR-R2 diagnostics thiếu scorer"
        )

    if not isinstance(
        alignment,
        dict,
    ):
        raise ValueError(
            f"Query {query_id!r}: "
            "TR-R2 diagnostics thiếu alignment"
        )

    chosen_times = get_chosen_times(
        alignment
    )

    print(
        f"TR-R2 OK | "
        f"video={diagnostics.get('video_id')}"
    )

    for event_id in event_ids:
        print(
            f"  {event_id}: "
            f"{chosen_times[event_id]:.3f}s"
        )

    # ------------------------------------------------------------------
    # TR-E2
    # ------------------------------------------------------------------

    print()
    print(
        "[3/3] TR-E2 boundary refinement..."
    )

    boundaries = refine_from_dense_scorer(
        scorer=scorer,
        event_ids=event_ids,
        chosen_times=chosen_times
    )

    boundaries = enforce_boundary_order(
        boundaries,
        scorer=scorer,
        chosen_times=chosen_times,
    )

    print("TR-E2 OK")

    for result in boundaries:
        print(
            f"  {result.event_id}: "
            f"[{result.start_time:.3f}s, "
            f"{result.end_time:.3f}s] "
            f"-> rep="
            f"{result.representative_time:.3f}s "
            f"(frame "
            f"{result.representative_frame_idx}) "
            f"| conf="
            f"{result.confidence:.3f}"
        )

    return {
        "query_id": query_id,
        "video_id": diagnostics.get(
            "video_id"
        ),
        "events": [
            {
                "event_id": event.event_id,
                "text": event.text,
                "relation": event.relation,
            }
            for event in plan.events
        ],
        "tr_r2": {
            "chosen_times": chosen_times,
            "total_score": alignment.get(
                "total_score"
            ),
        },
        "tr_e2": [
            asdict(result)
            for result in boundaries
        ],
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    cli = argparse.ArgumentParser(
        description=(
            "TR-E2 boundary refinement "
            "trên output TR-R2"
        )
    )

    cli.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Số query TRAKE cần chạy. "
            "0 = chạy tất cả."
        ),
    )

    cli.add_argument(
        "--query-id",
        action="append",
        default=None,
        help=(
            "Chỉ chạy query_id được chỉ định. "
            "Có thể truyền nhiều lần, ví dụ "
            "--query-id 08 --query-id 15"
        ),
    )

    args = cli.parse_args()

    trr1_config = TRR1Config(
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

    if args.query_id:
        wanted_ids = {
            str(query_id)
            for query_id in args.query_id
        }

        trake_rows = [
            row
            for row in trake_rows
            if str(row.get("id")) in wanted_ids
        ]

        found_ids = {
            str(row.get("id"))
            for row in trake_rows
        }

        missing_ids = (
            wanted_ids - found_ids
        )

        if missing_ids:
            raise ValueError(
                "Không tìm thấy TRAKE query id: "
                + ", ".join(sorted(missing_ids))
            )

    if not trake_rows:
        raise RuntimeError(
            "Không tìm thấy query TRAKE."
        )

    if args.limit > 0:
        selected_rows = (
            trake_rows[:args.limit]
        )
    else:
        selected_rows = trake_rows

    print("=" * 70)
    print(
        "TR-E2 — BOUNDARY REFINEMENT"
    )
    print("=" * 70)
    print(
        f"DEV      : {DEV_QUESTIONS}"
    )
    print(
        f"Tổng câu : {len(rows)}"
    )
    print(
        f"TRAKE    : {len(trake_rows)}"
    )
    print(
        f"Sẽ chạy  : {len(selected_rows)}"
    )

    parser_s1 = RuleBasedParser()

    outputs: list[
        dict[str, Any]
    ] = []

    failures = 0

    for row in selected_rows:
        try:
            output = run_one_query(
                row=row,
                parser_s1=parser_s1,
                trr1_config=trr1_config,
            )

            outputs.append(
                output
            )

        except Exception as exc:
            failures += 1

            print()
            print(
                f"[FAIL] query="
                f"{row.get('id')}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    print()
    print("=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)
    print(
        f"OK   : {len(outputs)}"
    )
    print(
        f"FAIL : {failures}"
    )

    print()
    print("JSON OUTPUT:")
    print(
        json.dumps(
            outputs,
            ensure_ascii=False,
            indent=2,
        )
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()