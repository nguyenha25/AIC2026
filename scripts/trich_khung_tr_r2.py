from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Cho phép cả hai cách chạy:
#   python -m scripts.trich_khung_tr_r2
#   python scripts/trich_khung_tr_r2.py
_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Windows có thể access violation nếu open_clip được import muộn, sau khi các
# native extension khác đã khởi tạo. benchmark_tr_r2.py cũng dùng đúng thứ tự
# này. Chỉ import module; model vẫn được lazy-load khi sparse scorer chạy.
import open_clip  # noqa: E402,F401

from aic2026.frame_map import load_frame_map  # noqa: E402
from aic2026.paths import RUNS_DIR, video_file  # noqa: E402
from aic2026.trake_r2_score import select_video_by_sparse_dp  # noqa: E402
from aic2026.trake_r2_windows import (
    rank_video_candidates_rrf,
    score_video_candidates_rrf,
    windows_from_anchor_times,
)  # noqa: E402

# QUAN TRỌNG:
# Không sửa scripts/trich_khung_day.py.
# Chỉ import lại hàm trich() và thong_tin_video() có sẵn.
from scripts.trich_khung_day import trich, thong_tin_video  # noqa: E402


# ============================================================
# CONFIG
# ============================================================

ARTIFACT = Path(r"D:\aic-data\runs\tr_r1_candidates.jsonl")

VIDEO_BEAM_SIZE = 12
SPARSE_MIN_GAP = 1
SPARSE_WINDOW_PADDING_SECONDS = 5.0
CANDIDATE_TOP_K = 3
DENSE_RERANK_MARGIN = 0.015
SPAN_PENALTY_WEIGHT = 0.041
SPAN_SCALE_SECONDS = 60.0
MISSING_VIDEO_LIST = RUNS_DIR / "tr_r2_missing_source_videos.txt"


# ============================================================
# LOAD TR-R1 ARTIFACT
# ============================================================

def load_tr_r1_candidates(
    path: Path,
) -> dict[str, list[dict[str, Any]]]:
    """
    Đọc tr_r1_candidates.jsonl.

    Trả về:
        {
            query_id: [
                event_record,
                ...
            ]
        }

    Mỗi event_record chứa:
        event_id
        text
        relation
        tr_r1
        ...
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy artifact TR-R1: {path}"
        )

    queries: dict[str, list[dict[str, Any]]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON lỗi tại dòng {line_no}: {exc}"
                ) from exc

            query_id = str(record.get("query_id", "")).strip()

            if not query_id:
                raise ValueError(
                    f"Dòng {line_no} không có query_id."
                )

            queries.setdefault(query_id, []).append(record)

    if not queries:
        raise ValueError(
            f"Artifact không có record nào: {path}"
        )

    return queries


# ============================================================
# RECONSTRUCT EVENTS_REGIONS
# ============================================================

def build_events_regions(
    event_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Chuyển artifact TR-R1 về format mà TR-R2 cần:

        {
            event_id: [
                {
                    "video_id": ...,
                    "start_time": ...,
                    "end_time": ...,
                    "score": ...
                }
            ]
        }

    Chỉ lấy regions từ TR-R1.

    KHÔNG lấy video_id GT ở root record để chọn video.
    """

    events_regions: dict[str, list[dict[str, Any]]] = {}

    for event in event_records:
        event_id = str(event.get("event_id", "")).strip()

        if not event_id:
            raise ValueError(
                "Event trong artifact không có event_id."
            )

        tr_r1 = event.get("tr_r1")

        if not isinstance(tr_r1, dict):
            raise ValueError(
                f"event={event_id!r} không có tr_r1 hợp lệ."
            )

        regions = tr_r1.get("regions", [])

        if not isinstance(regions, list):
            raise ValueError(
                f"event={event_id!r} có regions không hợp lệ."
            )

        clean_regions: list[dict[str, Any]] = []

        for region in regions:
            if not isinstance(region, dict):
                continue

            video_id = region.get("video_id")

            if not video_id:
                continue

            clean_regions.append(
                {
                    "video_id": str(video_id),
                    "start_time": float(region["start_time"]),
                    "end_time": float(region["end_time"]),
                    "score": float(region.get("score", 0.0)),
                }
            )

        if not clean_regions:
            print(
                f"[WARN] event={event_id} không có TR-R1 region."
            )

        events_regions[event_id] = clean_regions

    return events_regions


def build_event_texts(
    event_records: list[dict[str, Any]],
) -> dict[str, str]:
    """Lấy event text theo đúng event_index, không đọc GT."""

    output: dict[str, str] = {}

    for event in event_records:
        event_id = str(event.get("event_id", "")).strip()
        tr_r1 = event.get("tr_r1")

        if not event_id or not isinstance(tr_r1, dict):
            raise ValueError("Artifact thiếu event_id hoặc tr_r1")

        text = str(
            tr_r1.get("text", event.get("text", ""))
        ).strip()

        if not text:
            raise ValueError(f"event={event_id!r} không có text")

        output[event_id] = text

    return output


# ============================================================
# FPS
# ============================================================

def load_video_fps(video_id: str) -> float:
    """
    Lấy FPS từ frame_map.

    Dùng cùng nguồn FPS với Task 9.
    """

    frame_map = load_frame_map()

    rows = frame_map[
        frame_map["video_id"].astype(str) == str(video_id)
    ]

    if rows.empty:
        raise ValueError(
            f"Không tìm thấy video={video_id!r} trong frame_map."
        )

    fps_values = rows["fps"].dropna().astype(float).unique()

    if len(fps_values) == 0:
        raise ValueError(
            f"Không có FPS cho video={video_id!r}."
        )

    fps = float(fps_values[0])

    if fps <= 0:
        raise ValueError(
            f"FPS không hợp lệ cho video={video_id!r}: {fps}"
        )

    return fps


# ============================================================
# TIME → FRAME
# ============================================================

def seconds_to_frame_range(
    start_time: float,
    end_time: float,
    fps: float,
) -> tuple[int, int]:
    """
    Chuyển [start_time, end_time] sang [frame_start, frame_end].

    Giữ cách quy đổi tương ứng với CLI --giay của Task 9:
        int(seconds * fps)
    """

    if end_time < start_time:
        raise ValueError(
            f"Window không hợp lệ: "
            f"start={start_time}, end={end_time}"
        )

    frame_start = int(math.floor(start_time * fps))
    frame_end = int(math.floor(end_time * fps))

    if frame_start < 0:
        frame_start = 0

    if frame_end < frame_start:
        frame_end = frame_start

    return frame_start, frame_end


# ============================================================
# EXTRACT ONE QUERY
# ============================================================

def extract_candidate(
    query_id: str,
    candidate: dict[str, Any],
    event_ids: list[str],
    *,
    candidate_rank: int,
    buoc_giay: float,
    ghi_de: bool,
    sparse_window_padding_seconds: float,
) -> None:
    """Trích dense windows cho một sparse video candidate."""

    video_id = str(candidate["video_id"])
    chosen_times = candidate.get("chosen_times")

    if not isinstance(chosen_times, dict):
        raise ValueError("Sparse candidate không trả chosen_times")

    windows = windows_from_anchor_times(
        [float(chosen_times[event_id]) for event_id in event_ids],
        padding_seconds=sparse_window_padding_seconds,
    )
    fps = load_video_fps(video_id)
    source_video = video_file(video_id)

    if not source_video.exists():
        raise FileNotFoundError(
            f"Không tìm thấy source video cho {video_id!r}: {source_video}"
        )

    video_info = thong_tin_video(source_video)
    total_frames = int(video_info["so_khung"])

    print()
    print("=" * 72)
    print(
        f"[TR-R2] query={query_id} candidate_rank={candidate_rank} "
        f"video={video_id} windows={len(windows)}"
    )
    print(
        f"[TR-R2] sparse_final_score="
        f"{float(candidate.get('final_score', candidate.get('mean_score', 0.0))):.6f}"
    )
    print(f"[TR-R2] fps={fps:.6f} total_frames={total_frames}")
    print("=" * 72)

    for i, (start_time, end_time) in enumerate(windows, start=1):
        frame_start, frame_end = seconds_to_frame_range(
            start_time,
            end_time,
            fps,
        )

        if frame_start >= total_frames:
            print(
                f"[SKIP] window={i} [{start_time:.3f}, {end_time:.3f}] "
                f"nằm ngoài video."
            )
            continue

        frame_end = min(frame_end, total_frames - 1)
        print(
            f"[TR-R2] window={i}/{len(windows)} "
            f"time=[{start_time:.3f}, {end_time:.3f}] "
            f"frame=[{frame_start}, {frame_end}]"
        )
        trich(
            video_id,
            frame_start,
            frame_end,
            buoc_giay=buoc_giay,
            ghi_de=ghi_de,
        )

    print(
        f"[DONE] query={query_id} rank={candidate_rank} "
        f"video={video_id} windows={len(windows)}"
    )

def extract_query(
    query_id: str,
    event_records: list[dict[str, Any]],
    *,
    buoc_giay: float = 0.16,
    ghi_de: bool = False,
    video_beam_size: int = VIDEO_BEAM_SIZE,
    sparse_min_gap: int = SPARSE_MIN_GAP,
    sparse_window_padding_seconds: float = (
        SPARSE_WINDOW_PADDING_SECONDS
    ),
    candidate_top_k: int = CANDIDATE_TOP_K,
    dense_rerank_margin: float = DENSE_RERANK_MARGIN,
    span_penalty_weight: float = SPAN_PENALTY_WEIGHT,
    span_scale_seconds: float = SPAN_SCALE_SECONDS,
    missing_video_ids: set[str] | None = None,
) -> None:
    """
    Chạy extraction cho một query:

        TR-R1 regions -> Top-B video beam
            ↓
        video-local sparse CLIP-L + strict DP
            ↓
        local windows quanh sparse anchors
            ↓
        seconds → frame indices
            ↓
        Task 9 trich()
    """

    ordered_records = sorted(
        event_records,
        key=lambda item: int(item.get("event_index", 0)),
    )

    events_regions = build_events_regions(ordered_records)
    event_texts = build_event_texts(ordered_records)

    # --------------------------------------------------------
    # 1. Candidate beam -> video-local sparse DP
    # --------------------------------------------------------

    candidate_video_ids = rank_video_candidates_rrf(
        events_regions,
        limit=video_beam_size,
    )

    raw_video_priors = score_video_candidates_rrf(events_regions)
    region_evidence: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for event_id, regions in events_regions.items():
        for region in regions:
            video_id = str(region["video_id"])

            if video_id not in candidate_video_ids:
                continue

            region_evidence.setdefault(video_id, {}).setdefault(
                str(event_id),
                [],
            ).append(dict(region))

    sparse_selection = select_video_by_sparse_dp(
        event_texts,
        candidate_video_ids,
        min_gap=sparse_min_gap,
        video_priors=raw_video_priors,
        region_evidence=region_evidence,
        region_prior_weight=0.0,
        video_prior_weight=0.0,
        order_gain_weight=0.0,
        span_penalty_weight=span_penalty_weight,
        span_scale_seconds=span_scale_seconds,
    )

    ranked_candidates = list(sparse_selection.get("candidate_scores", []))

    if not ranked_candidates:
        ranked_candidates = [sparse_selection]

    if candidate_top_k <= 0:
        raise ValueError("candidate_top_k phải > 0")

    # Đồng bộ với adaptive dense rerank trong pipeline: video top-2/top-3 chỉ
    # cần trích khi top-1 sparse chưa đủ chắc chắn. candidate_top_k là trần,
    # không phải số lượng bắt buộc cho mọi query.
    selected_top_k = 1
    sparse_margin = float("inf")

    if len(ranked_candidates) > 1:
        sparse_margin = (
            float(ranked_candidates[0].get("final_score", 0.0))
            - float(ranked_candidates[1].get("final_score", 0.0))
        )
        if sparse_margin < dense_rerank_margin:
            selected_top_k = min(candidate_top_k, len(ranked_candidates))

    print(
        f"[TR-R2] query={query_id} sparse_margin={sparse_margin:.6f} "
        f"dense_candidates={selected_top_k}"
    )

    extracted = 0

    for candidate_rank, candidate in enumerate(
        ranked_candidates[:selected_top_k],
        start=1,
    ):
        try:
            extract_candidate(
                query_id,
                dict(candidate),
                list(event_texts.keys()),
                candidate_rank=candidate_rank,
                buoc_giay=buoc_giay,
                ghi_de=ghi_de,
                sparse_window_padding_seconds=(
                    sparse_window_padding_seconds
                ),
            )
            extracted += 1
        except FileNotFoundError as exc:
            if missing_video_ids is not None:
                missing_video_ids.add(str(candidate["video_id"]))
            print(
                f"[WARN] query={query_id} rank={candidate_rank}: {exc}"
            )

    if extracted == 0:
        raise FileNotFoundError(
            f"query={query_id}: không candidate nào có source video"
        )

# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TR-R2 dense frame extraction từ artifact TR-R1. "
            "Không sửa Task 9 extractor."
        )
    )

    parser.add_argument(
        "--artifact",
        type=Path,
        default=ARTIFACT,
        help=(
            "Artifact TR-R1 "
            "(default: D:\\aic-data\\runs\\tr_r1_candidates.jsonl)"
        ),
    )

    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help=(
            "Query ID cần extract. "
            "Có thể truyền nhiều lần: --query 07 --query 08"
        ),
    )

    parser.add_argument(
        "--buoc-giay",
        type=float,
        default=0.16,
        help="Dense sampling step, mặc định 0.16 giây.",
    )

    parser.add_argument(
        "--ghi-de",
        action="store_true",
        help="Cho phép ghi đè frame đã tồn tại.",
    )

    parser.add_argument(
        "--video-beam-size",
        type=int,
        default=VIDEO_BEAM_SIZE,
        help="Số video RRF đưa vào sparse DP, mặc định 12.",
    )

    parser.add_argument(
        "--sparse-min-gap",
        type=int,
        default=SPARSE_MIN_GAP,
        help="Khoảng cách vị trí keyframe tối thiểu của sparse DP.",
    )

    parser.add_argument(
        "--window-padding-seconds",
        type=float,
        default=SPARSE_WINDOW_PADDING_SECONDS,
        help="Padding mỗi phía quanh sparse anchor, mặc định 5 giây.",
    )

    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=CANDIDATE_TOP_K,
        help=(
            "Trích dense frame cho bao nhiêu video đứng đầu sparse rerank; "
            "mặc định 3 để adaptive dense rerank có đủ dữ liệu."
        ),
    )

    parser.add_argument(
        "--dense-rerank-margin",
        type=float,
        default=DENSE_RERANK_MARGIN,
        help=(
            "Chỉ trích top-k khi margin top1-top2 nhỏ hơn ngưỡng này; "
            "mặc định 0.015, đồng bộ full pipeline."
        ),
    )

    parser.add_argument(
        "--span-penalty-weight",
        type=float,
        default=SPAN_PENALTY_WEIGHT,
        help=(
            "Phạt path sparse trải dài; mặc định 0.041 đã đạt 4/12 dev."
        ),
    )

    parser.add_argument(
        "--span-scale-seconds",
        type=float,
        default=SPAN_SCALE_SECONDS,
        help="Scale của log-span penalty, mặc định 60 giây.",
    )

    parser.add_argument(
        "--missing-video-list",
        type=Path,
        default=MISSING_VIDEO_LIST,
        help=(
            "Ghi danh sách source video còn thiếu để tải/copy bổ sung. "
            "Mặc định: runs/tr_r2_missing_source_videos.txt dưới DATA_ROOT."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.buoc_giay <= 0:
        raise ValueError(
            f"--buoc-giay phải > 0, nhận {args.buoc_giay}"
        )

    if args.video_beam_size <= 0:
        raise ValueError("--video-beam-size phải > 0")

    if args.sparse_min_gap < 1:
        raise ValueError("--sparse-min-gap phải >= 1")

    if args.window_padding_seconds < 0:
        raise ValueError("--window-padding-seconds phải >= 0")

    if args.candidate_top_k <= 0:
        raise ValueError("--candidate-top-k phải > 0")

    if args.dense_rerank_margin < 0:
        raise ValueError("--dense-rerank-margin phải >= 0")

    if args.span_penalty_weight < 0:
        raise ValueError("--span-penalty-weight phải >= 0")

    if args.span_scale_seconds <= 0:
        raise ValueError("--span-scale-seconds phải > 0")

    # --------------------------------------------------------
    # Load artifact
    # --------------------------------------------------------

    queries = load_tr_r1_candidates(args.artifact)

    # --------------------------------------------------------
    # Chọn query
    # --------------------------------------------------------

    if args.query:
        requested = [str(q) for q in args.query]

        missing = [
            q for q in requested
            if q not in queries
        ]

        if missing:
            raise ValueError(
                "Không tìm thấy query trong artifact: "
                + ", ".join(missing)
            )

        query_ids = requested
    else:
        # Sort để chạy deterministic.
        query_ids = sorted(
            queries.keys(),
            key=lambda x: (
                int(x) if x.isdigit() else x
            ),
        )

    print("=" * 72)
    print("TR-R2 DENSE FRAME EXTRACTION")
    print("=" * 72)
    print(f"Artifact : {args.artifact}")
    print(f"Queries  : {len(query_ids)}")
    print(f"Step     : {args.buoc_giay}s")
    print(f"Beam     : {args.video_beam_size}")
    print(f"Sparse gap: {args.sparse_min_gap}")
    print(f"Padding  : {args.window_padding_seconds}s")
    print(f"Top dense: {args.candidate_top_k}")
    print(f"Dense margin: {args.dense_rerank_margin}")
    print(f"Span penalty: {args.span_penalty_weight}")
    print(f"Span scale: {args.span_scale_seconds}s")
    print(f"Ghi đè   : {args.ghi_de}")
    print()

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    success = 0
    failed = 0
    missing_video_ids: set[str] = set()

    for query_id in query_ids:
        try:
            extract_query(
                query_id,
                queries[query_id],
                buoc_giay=args.buoc_giay,
                ghi_de=args.ghi_de,
                video_beam_size=args.video_beam_size,
                sparse_min_gap=args.sparse_min_gap,
                sparse_window_padding_seconds=(
                    args.window_padding_seconds
                ),
                candidate_top_k=args.candidate_top_k,
                dense_rerank_margin=args.dense_rerank_margin,
                span_penalty_weight=args.span_penalty_weight,
                span_scale_seconds=args.span_scale_seconds,
                missing_video_ids=missing_video_ids,
            )
            success += 1

        except Exception as exc:
            failed += 1

            print()
            print(
                f"[ERROR] query={query_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TR-R2 EXTRACTION SUMMARY")
    print("=" * 72)
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print(f"Total   : {len(query_ids)}")

    if missing_video_ids:
        args.missing_video_list.parent.mkdir(parents=True, exist_ok=True)
        args.missing_video_list.write_text(
            "\n".join(sorted(missing_video_ids)) + "\n",
            encoding="utf-8",
        )
        print(f"Missing : {len(missing_video_ids)} video")
        print(f"List    : {args.missing_video_list}")

    print("=" * 72)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
