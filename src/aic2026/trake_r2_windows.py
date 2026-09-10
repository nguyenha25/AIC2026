"""
trake_r2_windows.py — Các hàm xử lý cửa sổ và RRF cho TR-R2.
"""
from __future__ import annotations

import os
from pathlib import Path


def generate_dense_time_grid(start_time: float, end_time: float, step: float = 0.16) -> list[float]:
    """Sinh danh sách pts_time cách đều `step` giây trong [start_time, end_time]."""
    if step <= 0:
        raise ValueError("step phải > 0")
    if end_time < start_time:
        raise ValueError(f"end_time ({end_time}) phải >= start_time ({start_time})")

    n_steps = int((end_time - start_time) / step) + 1
    return [round(start_time + i * step, 6) for i in range(n_steps)]


def _video_file_exists(video_id: str) -> bool:
    """Kiểm tra file video thật có tồn tại trên đĩa không (DATA_ROOT/raw/videos/<id>.mp4)."""
    data_root = Path(os.environ.get("DATA_ROOT", r"D:\aic-data"))
    video_path = data_root / "raw" / "videos" / f"{video_id}.mp4"
    return video_path.exists()


def chon_video_rrf(
    events_regions: dict,
    k: int = 60,
    gt_video_hint: str | None = None,
    require_video_file_exists: bool = True,
) -> str:
    """
    Chọn video cho câu TRAKE bằng RRF tích hợp Coverage Boost.
    """
    if gt_video_hint:
        return gt_video_hint

    all_videos = set()
    for regions in events_regions.values():
        for r in regions:
            all_videos.add(r["video_id"])

    if require_video_file_exists:
        videos_with_file = {v for v in all_videos if _video_file_exists(v)}
        if videos_with_file:
            all_videos = videos_with_file

    total_events = len(events_regions)
    video_scores = {}
    video_event_counts = {}

    # Bước 1: Tính RRF score và đếm số event mà video phủ tới
    for events_id, regions in events_regions.items():
        seen_in_this_event = set()
        for rank, region in enumerate(regions, start=1):
            vid = region["video_id"]
            if vid not in all_videos:
                continue
            if vid in seen_in_this_event:
                continue  # Mỗi event chỉ tính 1 lần cho mỗi video để tránh bias
            seen_in_this_event.add(vid)

            video_event_counts[vid] = video_event_counts.get(vid, 0) + 1
            conf = float(region.get("score", 1.0))
            video_scores[vid] = video_scores.get(vid, 0.0) + conf / (k + rank)

    # Bước 2: Nhân thêm hệ số phủ (Coverage Boost) để trị tận gốc các video nhiễu rải rác
    # Tăng trọng số coverage từ 3.0 lên 8.0 để bảo vệ video đúng qua các event
    final_scores = {}
    for vid, score in video_scores.items():
        coverage_ratio = video_event_counts[vid] / max(1, total_events)
        final_scores[vid] = score * (1.0 + 8.0 * coverage_ratio)

    if not final_scores:
        raise ValueError("Không có candidate video nào để chọn (sau khi lọc).")

    return max(final_scores.items(), key=lambda kv: kv[1])[0]


def gop_cua_so_theo_video(events_regions: dict, video_id: str) -> tuple[float, float]:
    """
    Gộp TẤT CẢ region thuộc `video_id` (qua mọi event) thành một cửa sổ
    [min_start, max_end] duy nhất — đây là vùng sẽ dense hóa.
    """
    starts, ends = [], []
    for regions in events_regions.values():
        for r in regions:
            if r["video_id"] == video_id:
                starts.append(r["start_time"])
                ends.append(r["end_time"])

    if not starts:
        raise ValueError(
            f"Video {video_id!r} được chọn nhưng không có region nào khớp — "
            f"kiểm tra lại chon_video_rrf() và events_regions có nhất quán không."
        )

    return min(starts), max(ends)


def gop_cac_cua_so_theo_video(
    events_regions: dict,
    video_id: str,
) -> list[tuple[float, float]]:
    """
    Lấy tất cả coarse regions thuộc video_id và gộp các region
    overlap/chạm nhau thành các temporal windows rời nhau.
    """
    intervals: list[tuple[float, float]] = []

    for regions in events_regions.values():
        for r in regions:
            if r["video_id"] != video_id:
                continue

            start = float(r["start_time"])
            end = float(r["end_time"])

            if end < start:
                raise ValueError(
                    f"Region không hợp lệ: start={start}, end={end}"
                )

            intervals.append((start, end))

    if not intervals:
        raise ValueError(
            f"Video {video_id!r} được chọn nhưng không có region nào."
        )

    intervals.sort(key=lambda x: (x[0], x[1]))

    merged: list[list[float]] = []

    for start, end in intervals:
        if not merged:
            merged.append([start, end])
            continue

        prev_start, prev_end = merged[-1]

        # Chỉ merge khi overlap hoặc chạm nhau.
        if start <= prev_end:
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])

    return [
        (float(start), float(end))
        for start, end in merged
    ]