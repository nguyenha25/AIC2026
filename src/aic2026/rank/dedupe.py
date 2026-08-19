"""Khử trùng lặp theo cửa sổ thời gian 10 giây."""
from __future__ import annotations

from typing import Any, Sequence
from aic2026.frame_map import FrameMap


def deduplicate_temporal(
    ranked_items: Sequence[dict[str, Any]],
    frame_map: FrameMap | None = None,
    window_seconds: float = 10.0,
    fps_fallback: float = 25.0,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Giữ lại ứng viên có thứ hạng cao nhất trong mỗi cửa sổ thời gian."""
    accepted: list[dict[str, Any]] = []
    video_time_anchors: dict[str, list[float]] = {}

    for item in ranked_items:
        vid = str(item["video_id"])
        fidx = int(item["frame_idx"])

        pts_time: float | None = None
        if frame_map is not None:
            try:
                pts_time = frame_map.pts_time_of(vid, fidx)
            except Exception:
                pts_time = None

        if pts_time is None:
            pts_time = float(fidx) / fps_fallback

        anchors = video_time_anchors.setdefault(vid, [])
        if not any(abs(pts_time - anchor) < window_seconds for anchor in anchors):
            anchors.append(pts_time)
            accepted.append(dict(item))
            if len(accepted) >= limit:
                break

    return accepted