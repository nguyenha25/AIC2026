"""
TR-E2 — smoothing và latent boundary refinement cho TRAKE.

Module này KHÔNG:
- chạy retrieval;
- load CLIP;
- đọc ground truth;
- sửa DP của TR-R2.

Input:
    dense frame_idx / pts_time
    raw relevance score của một event
    anchor frame đã được TR-R2 chọn

Output:
    một latent interval [start, end]
    + representative frame
    + confidence

Boundary chỉ dùng nội bộ.
Submission TRAKE cuối cùng vẫn dùng đúng một frame_idx / event.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence


import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class BoundaryResult:
    event_id: str

    start_pos: int
    end_pos: int
    representative_pos: int

    start_frame_idx: int
    end_frame_idx: int
    representative_frame_idx: int

    start_time: float
    end_time: float
    representative_time: float

    peak_score_raw: float
    peak_score_smooth: float
    interval_mean_score: float
    confidence: float

    status: str = "ok"


def _as_1d_float_array(
    values: Sequence[float],
    *,
    name: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError(f"{name} phải là mảng 1 chiều.")

    if len(arr) == 0:
        raise ValueError(f"{name} không được rỗng.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} chứa NaN/inf.")

    return arr


def minmax_normalize(
    scores: Sequence[float],
) -> np.ndarray:
    """
    Đưa score về [0, 1].

    Nếu toàn bộ score bằng nhau thì trả vector 0,
    vì không có bằng chứng phân biệt frame nào tốt hơn.
    """

    x = _as_1d_float_array(scores, name="scores")

    lo = float(np.min(x))
    hi = float(np.max(x))

    if hi - lo <= EPS:
        return np.zeros_like(x)

    return (x - lo) / (hi - lo)


def smooth_scores(
    scores: Sequence[float],
    *,
    radius: int = 2,
) -> np.ndarray:
    """
    Moving-average đối xứng.

    radius=0:
        không smoothing

    radius=2:
        kernel 5 frame

    Dùng edge padding để không làm mất frame đầu/cuối.
    """

    x = _as_1d_float_array(scores, name="scores")

    radius = int(radius)

    if radius < 0:
        raise ValueError("radius phải >= 0.")

    if radius == 0 or len(x) == 1:
        return x.copy()

    width = 2 * radius + 1

    padded = np.pad(
        x,
        pad_width=radius,
        mode="edge",
    )

    kernel = np.ones(width, dtype=np.float64) / width

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


def nearest_position(
    values: Sequence[float],
    target: float,
) -> int:
    """
    Trả position gần target nhất.
    Tie -> position bên trái để deterministic.
    """

    x = _as_1d_float_array(values, name="values")
    target = float(target)

    return int(np.argmin(np.abs(x - target)))


def _local_peak_near_anchor(
    smooth: np.ndarray,
    anchor_pos: int,
    search_radius: int,
) -> int:
    """
    Tìm peak gần anchor, không cho TR-E2 tự nhảy
    sang một peak xa hoàn toàn do event khác/window khác.
    """

    n = len(smooth)

    left = max(0, anchor_pos - search_radius)
    right = min(n - 1, anchor_pos + search_radius)

    local = smooth[left : right + 1]

    # np.argmax deterministic: tie -> phần tử đầu tiên.
    return left + int(np.argmax(local))


def _candidate_boundaries(
    smooth: np.ndarray,
    peak_pos: int,
    *,
    relative_threshold: float,
    absolute_floor: float,
    max_radius: int,
) -> tuple[int, int]:
    """
    Mở rộng từ peak sang trái/phải đến khi score rơi dưới ngưỡng.

    threshold =
        max(absolute_floor,
            peak_score * relative_threshold)

    Đây là boundary proposal; bước objective phía sau
    sẽ chọn interval cuối.
    """

    peak = float(smooth[peak_pos])

    threshold = max(
        float(absolute_floor),
        peak * float(relative_threshold),
    )

    left_limit = max(0, peak_pos - max_radius)
    right_limit = min(len(smooth) - 1, peak_pos + max_radius)

    left = peak_pos
    while (
        left - 1 >= left_limit
        and smooth[left - 1] >= threshold
    ):
        left -= 1

    right = peak_pos
    while (
        right + 1 <= right_limit
        and smooth[right + 1] >= threshold
    ):
        right += 1

    return left, right


def _interval_objective(
    smooth: np.ndarray,
    start: int,
    end: int,
    peak_pos: int,
    *,
    length_penalty: float,
    edge_penalty: float,
) -> float:
    """
    Phiên bản MVP của objective trong handbook:

        mean(score[a:b])
        - mu * length_penalty
        - gamma * discontinuity

    Coverage chưa cộng ở TR-E2 vì score input hiện là
    một modality CLIP-L/event và coverage thuộc tầng khác.

    discontinuity được xấp xỉ bằng score ở hai biên:
    boundary tốt khi bên trong mạnh nhưng biên không quá cao.
    """

    segment = smooth[start : end + 1]

    mean_score = float(np.mean(segment))

    length = end - start + 1

    normalized_length = (
        (length - 1) / max(1, len(smooth) - 1)
    )

    left_edge = float(smooth[start])
    right_edge = float(smooth[end])

    discontinuity_penalty = (
        left_edge + right_edge
    ) / 2.0

    # Bắt buộc chứa peak.
    if not (start <= peak_pos <= end):
        return float("-inf")

    return (
        mean_score
        - length_penalty * normalized_length
        - edge_penalty * discontinuity_penalty
    )


def optimize_interval(
    smooth_scores_normalized: Sequence[float],
    peak_pos: int,
    proposal_start: int,
    proposal_end: int,
    *,
    length_penalty: float = 0.08,
    edge_penalty: float = 0.03,
) -> tuple[int, int]:
    """
    Exhaustive LOCAL search trong proposal window.

    Không phải O(F^2) toàn video:
    chỉ duyệt vùng nhỏ quanh peak do max_radius giới hạn.

    Tie-break:
        1. objective cao hơn
        2. interval ngắn hơn
        3. start nhỏ hơn

    để kết quả deterministic.
    """

    s = _as_1d_float_array(
        smooth_scores_normalized,
        name="smooth_scores_normalized",
    )

    n = len(s)

    if not 0 <= peak_pos < n:
        raise ValueError("peak_pos nằm ngoài mảng.")

    if not (
        0 <= proposal_start <= peak_pos <= proposal_end < n
    ):
        raise ValueError(
            "proposal interval không hợp lệ."
        )

    best_start = peak_pos
    best_end = peak_pos

    best_objective = _interval_objective(
        s,
        peak_pos,
        peak_pos,
        peak_pos,
        length_penalty=length_penalty,
        edge_penalty=edge_penalty,
    )

    best_length = 1

    for start in range(
        proposal_start,
        peak_pos + 1,
    ):
        for end in range(
            peak_pos,
            proposal_end + 1,
        ):
            objective = _interval_objective(
                s,
                start,
                end,
                peak_pos,
                length_penalty=length_penalty,
                edge_penalty=edge_penalty,
            )

            length = end - start + 1

            better = objective > best_objective + EPS

            tied_shorter = (
                abs(objective - best_objective) <= EPS
                and length < best_length
            )

            tied_same_length_earlier = (
                abs(objective - best_objective) <= EPS
                and length == best_length
                and start < best_start
            )

            if (
                better
                or tied_shorter
                or tied_same_length_earlier
            ):
                best_objective = objective
                best_start = start
                best_end = end
                best_length = length

    return best_start, best_end


def _confidence(
    smooth: np.ndarray,
    start: int,
    end: int,
    representative_pos: int,
) -> float:
    """
    Confidence heuristic [0,1].

    Kết hợp:
        - peak strength
        - mean interval
        - margin peak so với ngoài interval

    Đây chưa phải calibrated probability.
    """

    peak = float(smooth[representative_pos])
    mean_inside = float(
        np.mean(smooth[start : end + 1])
    )

    outside_parts = []

    if start > 0:
        outside_parts.append(smooth[:start])

    if end + 1 < len(smooth):
        outside_parts.append(smooth[end + 1 :])

    if outside_parts:
        outside = np.concatenate(outside_parts)
        second = float(np.max(outside))
    else:
        second = 0.0

    margin = max(0.0, peak - second)

    conf = (
        0.50 * peak
        + 0.30 * mean_inside
        + 0.20 * margin
    )

    return float(np.clip(conf, 0.0, 1.0))


def refine_event_boundary(
    *,
    event_id: str,
    frame_idx: Sequence[int],
    pts_time: Sequence[float],
    raw_scores: Sequence[float],
    anchor_frame_idx: int | None = None,
    anchor_time: float | None = None,
    smoothing_radius: int = 2,
    peak_search_radius: int = 12,
    boundary_max_radius: int = 20,
    relative_threshold: float = 0.55,
    absolute_floor: float = 0.20,
    length_penalty: float = 0.08,
    edge_penalty: float = 0.03,
) -> BoundaryResult:
    """
    Refine một event.

    Phải cung cấp một trong:
        anchor_frame_idx
        anchor_time

    Ưu tiên anchor_frame_idx nếu có.
    """

    frames = np.asarray(frame_idx, dtype=np.int64)
    times = _as_1d_float_array(
        pts_time,
        name="pts_time",
    )
    raw = _as_1d_float_array(
        raw_scores,
        name="raw_scores",
    )

    if frames.ndim != 1:
        raise ValueError(
            "frame_idx phải là mảng 1 chiều."
        )

    n = len(frames)

    if n == 0:
        raise ValueError(
            "frame_idx không được rỗng."
        )

    if len(times) != n or len(raw) != n:
        raise ValueError(
            "frame_idx, pts_time và raw_scores "
            "phải cùng số phần tử."
        )

    if len(np.unique(frames)) != n:
        raise ValueError(
            "frame_idx bị trùng."
        )

    if np.any(np.diff(times) < 0):
        raise ValueError(
            "pts_time phải tăng không giảm."
        )

    if np.any(np.diff(frames) <= 0):
        raise ValueError(
            "frame_idx phải tăng nghiêm ngặt."
        )

    if anchor_frame_idx is None and anchor_time is None:
        raise ValueError(
            "Phải có anchor_frame_idx hoặc anchor_time."
        )

    normalized = minmax_normalize(raw)

    smooth = smooth_scores(
        normalized,
        radius=smoothing_radius,
    )

    if anchor_frame_idx is not None:
        anchor_pos = nearest_position(
            frames.astype(np.float64),
            float(anchor_frame_idx),
        )
    else:
        assert anchor_time is not None
        anchor_pos = nearest_position(
            times,
            float(anchor_time),
        )

    peak_pos = _local_peak_near_anchor(
        smooth,
        anchor_pos,
        int(peak_search_radius),
    )

    proposal_start, proposal_end = (
        _candidate_boundaries(
            smooth,
            peak_pos,
            relative_threshold=relative_threshold,
            absolute_floor=absolute_floor,
            max_radius=int(boundary_max_radius),
        )
    )

    start, end = optimize_interval(
        smooth,
        peak_pos,
        proposal_start,
        proposal_end,
        length_penalty=length_penalty,
        edge_penalty=edge_penalty,
    )

    # Representative luôn là frame có smooth score cao nhất
    # trong latent interval.
    local = smooth[start : end + 1]
    representative_pos = (
        start + int(np.argmax(local))
    )

    confidence = _confidence(
        smooth,
        start,
        end,
        representative_pos,
    )

    return BoundaryResult(
        event_id=str(event_id),

        start_pos=int(start),
        end_pos=int(end),
        representative_pos=int(
            representative_pos
        ),

        start_frame_idx=int(frames[start]),
        end_frame_idx=int(frames[end]),
        representative_frame_idx=int(
            frames[representative_pos]
        ),

        start_time=float(times[start]),
        end_time=float(times[end]),
        representative_time=float(
            times[representative_pos]
        ),

        peak_score_raw=float(
            raw[representative_pos]
        ),
        peak_score_smooth=float(
            smooth[representative_pos]
        ),
        interval_mean_score=float(
            np.mean(smooth[start : end + 1])
        ),
        confidence=confidence,

        status="ok",
    )

def refine_from_dense_scorer(
    *,
    scorer,
    event_ids: Sequence[str],
    chosen_times: dict[str, float],
    smoothing_radius: int = 2,
    peak_search_radius: int = 12,
    boundary_max_radius: int = 20,
    relative_threshold: float = 0.55,
    absolute_floor: float = 0.20,
    length_penalty: float = 0.08,
    edge_penalty: float = 0.03,
    continuity_gap_seconds: float = 0.50,
) -> list[BoundaryResult]:
    """
    Refine boundary trực tiếp từ DenseClipLScorer của TR-R2.

    Quan trọng:
    scorer.frames có thể chứa frame từ NHIỀU dense window rời nhau.
    Vì vậy mỗi event chỉ được refine trong contiguous temporal component
    chứa anchor mà TR-R2 đã chọn.

    Nếu không giới hạn component, peak_search_radius tính theo array
    position có thể vô tình nhảy qua một khoảng thời gian rất lớn,
    ví dụ từ 361 s sang 556 s.
    """

    frames = scorer.frames
    score_matrix = scorer.score_matrix

    if not frames:
        raise ValueError(
            "Dense scorer không có frame"
        )

    if score_matrix is None:
        raise ValueError(
            "Dense scorer chưa có score_matrix"
        )

    if score_matrix.ndim != 2:
        raise ValueError(
            "score_matrix phải là ma trận 2 chiều"
        )

    if score_matrix.shape[0] != len(event_ids):
        raise ValueError(
            "Số hàng score_matrix không khớp "
            "với số event"
        )

    if score_matrix.shape[1] != len(frames):
        raise ValueError(
            "Số cột score_matrix không khớp "
            "với số dense frame"
        )

    if continuity_gap_seconds <= 0:
        raise ValueError(
            "continuity_gap_seconds phải > 0"
        )

    frame_idx = [
        int(frame.frame_idx)
        for frame in frames
    ]

    pts_time = [
        float(frame.pts_time)
        for frame in frames
    ]

    results: list[BoundaryResult] = []

    for event_pos, event_id in enumerate(
        event_ids
    ):
        if event_id not in chosen_times:
            raise ValueError(
                f"Thiếu chosen_time cho event "
                f"{event_id!r}"
            )

        anchor_time = float(
            chosen_times[event_id]
        )

        # ---------------------------------------------------------------
        # 1. Tìm dense frame thật gần anchor TR-R2 nhất.
        # ---------------------------------------------------------------

        anchor_pos = min(
            range(len(pts_time)),
            key=lambda i: (
                abs(pts_time[i] - anchor_time),
                i,
            ),
        )

        # ---------------------------------------------------------------
        # 2. Chỉ lấy contiguous temporal component chứa anchor.
        #
        # Ví dụ:
        #   361.28, 361.44, 361.60,
        #   556.20, 556.36, 556.48
        #
        # gap 194 giây sẽ tách thành hai component.
        # ---------------------------------------------------------------

        left = anchor_pos

        while left > 0:
            gap = (
                pts_time[left]
                - pts_time[left - 1]
            )

            if gap > continuity_gap_seconds:
                break

            left -= 1

        right = anchor_pos

        while right + 1 < len(pts_time):
            gap = (
                pts_time[right + 1]
                - pts_time[right]
            )

            if gap > continuity_gap_seconds:
                break

            right += 1

        local_frame_idx = (
            frame_idx[left:right + 1]
        )

        local_pts_time = (
            pts_time[left:right + 1]
        )

        local_scores = (
            score_matrix[
                event_pos,
                left:right + 1,
            ]
        )

        if len(local_frame_idx) == 0:
            raise ValueError(
                f"Event {event_id!r}: "
                "không có frame trong anchor component"
            )

        # ---------------------------------------------------------------
        # 3. Refine CHỈ bên trong component này.
        # ---------------------------------------------------------------

        local_result = refine_event_boundary(
            event_id=event_id,
            frame_idx=local_frame_idx,
            pts_time=local_pts_time,
            raw_scores=local_scores,
            anchor_time=anchor_time,
            smoothing_radius=smoothing_radius,
            peak_search_radius=peak_search_radius,
            boundary_max_radius=boundary_max_radius,
            relative_threshold=relative_threshold,
            absolute_floor=absolute_floor,
            length_penalty=length_penalty,
            edge_penalty=edge_penalty,
        )

        # ---------------------------------------------------------------
        # 4. BoundaryResult local dùng position tính từ 0 của slice.
        # Chuyển lại thành position global trong scorer.frames.
        # ---------------------------------------------------------------

        result = replace(
            local_result,
            start_pos=(
                local_result.start_pos + left
            ),
            end_pos=(
                local_result.end_pos + left
            ),
            representative_pos=(
                local_result.representative_pos
                + left
            ),
        )

        results.append(
            result
        )

    return results