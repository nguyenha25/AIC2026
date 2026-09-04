from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from aic2026.frame_map import load_frame_map
from aic2026.paths import video_file
from aic2026.trake_r2_windows import (
    chon_video_rrf,
    gop_cac_cua_so_theo_video,
)

# QUAN TRỌNG:
# Không sửa scripts/trich_khung_day.py.
# Chỉ import lại hàm trich() và thong_tin_video() có sẵn.
from scripts.trich_khung_day import trich, thong_tin_video


# ============================================================
# CONFIG
# ============================================================

ARTIFACT = Path(r"D:\aic-data\runs\tr_r1_candidates.jsonl")


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

def extract_query(
    query_id: str,
    event_records: list[dict[str, Any]],
    *,
    buoc_giay: float = 0.16,
    ghi_de: bool = False,
) -> None:
    """
    Chạy extraction cho một query:

        TR-R1 regions
            ↓
        RRF chọn video
            ↓
        merge windows
            ↓
        seconds → frame indices
            ↓
        Task 9 trich()
    """

    events_regions = build_events_regions(event_records)

    # --------------------------------------------------------
    # 1. Chọn video bằng TR-R1 RRF
    # --------------------------------------------------------

    video_id = chon_video_rrf(events_regions)

    # --------------------------------------------------------
    # 2. Lấy các cửa sổ thời gian TR-R2
    # --------------------------------------------------------

    windows = gop_cac_cua_so_theo_video(
        events_regions,
        video_id,
    )

    if not windows:
        raise ValueError(
            f"query={query_id} chọn video={video_id!r} "
            f"nhưng không có window."
        )

    # --------------------------------------------------------
    # 3. FPS
    # --------------------------------------------------------

    fps = load_video_fps(video_id)

    # --------------------------------------------------------
    # 4. Kiểm tra video
    # --------------------------------------------------------

    source_video = video_file(video_id)

    if not source_video.exists():
        raise FileNotFoundError(
            f"Không tìm thấy source video cho "
            f"{video_id!r}: {source_video}"
        )

    _, raw_total_frames = thong_tin_video(source_video)
    try:
        total_frames = int(raw_total_frames)
    except (ValueError, TypeError):
        total_frames = None

    # --------------------------------------------------------
    # 5. In thông tin
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        f"[TR-R2] query={query_id} "
        f"video={video_id} "
        f"windows={len(windows)}"
    )
    print(
        f"[TR-R2] fps={fps:.6f} "
        f"total_frames={total_frames}"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # 6. Extract từng window
    # --------------------------------------------------------

    for i, (start_time, end_time) in enumerate(
        windows,
        start=1,
    ):
        frame_start, frame_end = seconds_to_frame_range(
            start_time,
            end_time,
            fps,
        )

        # Không cho vượt quá video.
        if total_frames is not None:
            if frame_start >= total_frames:
                print(
                    f"[SKIP] window={i} "
                    f"[{start_time:.3f}, {end_time:.3f}] "
                    f"→ frame [{frame_start}, {frame_end}] "
                    f"nằm ngoài video."
                )
                continue

            frame_end = min(
                frame_end,
                int(total_frames) - 1,
            )

        print()
        print(
            f"[TR-R2] window={i}/{len(windows)}"
        )
        print(
            f"  time  = [{start_time:.3f}, {end_time:.3f}]"
        )
        print(
            f"  frame = [{frame_start}, {frame_end}]"
        )
        print(
            f"  dur   = {end_time - start_time:.3f}s"
        )

        # ----------------------------------------------------
        # TÁI SỬ DỤNG TASK 9
        # ----------------------------------------------------

        trich(
            video_id,
            frame_start,
            frame_end,
            buoc_giay=buoc_giay,
            ghi_de=ghi_de,
        )

    print()
    print(
        f"[DONE] query={query_id} "
        f"video={video_id} "
        f"windows={len(windows)}"
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.buoc_giay <= 0:
        raise ValueError(
            f"--buoc-giay phải > 0, nhận {args.buoc_giay}"
        )

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
    print(f"Ghi đè   : {args.ghi_de}")
    print()

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    success = 0
    failed = 0

    for query_id in query_ids:
        try:
            extract_query(
                query_id,
                queries[query_id],
                buoc_giay=args.buoc_giay,
                ghi_de=args.ghi_de,
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
    print("=" * 72)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()