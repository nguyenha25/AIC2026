"""
trake_r2_windows.py — candidate video beam và local windows của TR-R2:

1. generate_dense_time_grid(): sinh lưới thời gian bước 0.16s trong 1 cửa sổ
   (đúng "trich_khung_day.py dùng bước mặc định 0,16 giây" trong handbook).

2. rank_video_candidates_rrf(): tạo beam video từ TR-R1 regions. RRF chỉ
   dùng để giữ candidate, KHÔNG còn chốt video cuối cùng.

3. windows_from_anchor_times(): tạo local windows quanh sparse anchors sau
   khi video-local CLIP-L + DP đã chọn video.
"""
from __future__ import annotations

import math
from collections.abc import Iterable


def generate_dense_time_grid(start_time: float, end_time: float, step: float = 0.16) -> list[float]:
    """Sinh danh sách pts_time cách đều `step` giây trong [start_time, end_time]."""
    if step <= 0:
        raise ValueError("step phải > 0")
    if end_time < start_time:
        raise ValueError(f"end_time ({end_time}) phải >= start_time ({start_time})")

    n_steps = int((end_time - start_time) / step) + 1
    return [round(start_time + i * step, 6) for i in range(n_steps)]


def rank_video_candidates_rrf(
    events_regions: dict,
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[str]:
    """Xếp hạng candidate video bằng RRF đã deduplicate theo event.

    Một video chỉ đóng góp best rank một lần trong mỗi event. Kết quả này
    chỉ là beam đầu vào cho video-local sparse DP; không phải quyết định
    video cuối cùng.
    """
    if k < 0:
        raise ValueError("k phải >= 0")

    if limit is not None and limit <= 0:
        raise ValueError("limit phải > 0 hoặc None")

    scores: dict[str, float] = {}

    for regions in events_regions.values():
        seen: set[str] = set()
        unique_rank = 0

        for region in regions:
            video_id = str(region["video_id"])

            if video_id in seen:
                continue

            seen.add(video_id)
            unique_rank += 1
            scores[video_id] = (
                scores.get(video_id, 0.0)
                + 1.0 / (k + unique_rank)
            )

    if not scores:
        raise ValueError("Không có video ứng viên")

    ranked = sorted(
        scores,
        key=lambda video_id: (
            -scores[video_id],
            video_id,
        ),
    )

    if limit is not None:
        ranked = ranked[:limit]

    return ranked


def chon_video_rrf(events_regions: dict, k: int = 60) -> str:
    """Compatibility wrapper: video đứng đầu beam RRF."""

    return rank_video_candidates_rrf(
        events_regions,
        k=k,
        limit=1,
    )[0]


def windows_from_anchor_times(
    anchor_times: Iterable[float],
    *,
    padding_seconds: float = 5.0,
) -> list[tuple[float, float]]:
    """Tạo và merge local windows quanh sparse anchor times.

    Anchor phải là timestamp thật của keyframe trong video đã chọn. Mỗi
    window được clamp ở 0 giây và các window overlap/chạm nhau được gộp để
    tránh encode một dense frame nhiều lần.
    """

    padding_seconds = float(padding_seconds)

    if padding_seconds < 0:
        raise ValueError("padding_seconds phải >= 0")

    times = sorted(float(value) for value in anchor_times)

    if not times:
        raise ValueError("anchor_times không được rỗng")

    if any(not math.isfinite(value) for value in times):
        raise ValueError("anchor_times phải là số hữu hạn")

    intervals = [
        (
            max(0.0, value - padding_seconds),
            value + padding_seconds,
        )
        for value in times
    ]

    merged: list[list[float]] = []

    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue

        merged[-1][1] = max(merged[-1][1], end)

    return [
        (float(start), float(end))
        for start, end in merged
    ]


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

    Không dùng GT.
    Không nối các region chỉ vì chúng nằm trong cùng một video.
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
