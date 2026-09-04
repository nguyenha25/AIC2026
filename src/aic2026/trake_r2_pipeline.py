from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from aic2026.trake_r2_dp import solve_strict_increasing_path
from aic2026.trake_r2_score import (
    build_dense_score_fn,
    select_video_by_sparse_dp,
)
from aic2026.trake_r2_windows import (
    generate_dense_time_grid,
    chon_video_rrf,
    gop_cac_cua_so_theo_video,
    rank_video_candidates_rrf,
    windows_from_anchor_times,
)


# ============================================================================
# PURE DP / TESTABLE PATH
# ============================================================================


def _align_trake_fixed_windows(
    events_tr_r1: Mapping[str, dict],
    score_fn: Callable[[str, float], float],
    *,
    video_id: str,
    windows: Sequence[tuple[float, float]],
    step: float = 0.16,
    min_gap: int = 1,
) -> dict:
    """
    Chạy dense time-grid + DP trên nhiều temporal windows của cùng một video.

    `windows` đã được chọn từ TR-R1, không chứa GT.

    Tất cả frame times từ các window được gộp lại và sort theo thời gian
    trước khi đưa vào DP.

    Hàm này không biết CLIP, OCR, ASR hay VLM.
    Chỉ nhận score_fn(event_id, pts_time).
    """

    event_ids = list(events_tr_r1.keys())

    # ------------------------------------------------------------------
    # 1. Dense time grid cho từng window.
    # ------------------------------------------------------------------

    frame_times_set: set[float] = set()

    for window_start, window_end in windows:
        frame_times = generate_dense_time_grid(
            window_start,
            window_end,
            step=step,
        )

        frame_times_set.update(frame_times)

    # Gộp + sort toàn bộ candidate frames theo thời gian.
    frame_times = sorted(frame_times_set)

    if not frame_times:
        raise ValueError(
            f"Không có dense frame time nào cho video={video_id!r}."
        )

    # ------------------------------------------------------------------
    # 2. Score matrix:
    #
    #       event × dense frame
    #
    # Không phụ thuộc window nữa vì score_fn đã được prepare trên
    # toàn bộ candidate dense frames.
    # ------------------------------------------------------------------

    S = [
        [score_fn(event_id, pts_time) for pts_time in frame_times]
        for event_id in event_ids
    ]

    # ------------------------------------------------------------------
    # 3. Strict-increasing temporal DP.
    # ------------------------------------------------------------------

    chosen_idx, total_score = solve_strict_increasing_path(
        S,
        min_gap=min_gap,
    )

    # ------------------------------------------------------------------
    # 4. Kết quả.
    # ------------------------------------------------------------------

    return {
        "video_id": video_id,
        "windows": [
            [float(start), float(end)]
            for start, end in windows
        ],
        "dense_frame_times": frame_times,
        "chosen_times": {
            event_id: frame_times[j]
            for event_id, j in zip(event_ids, chosen_idx)
        },
        "total_score": total_score,
    }


def _align_trake_fixed_window(
    events_tr_r1: Mapping[str, dict],
    score_fn: Callable[[str, float], float],
    *,
    video_id: str,
    window_start: float,
    window_end: float,
    step: float = 0.16,
    min_gap: int = 1,
) -> dict:
    """
    Compatibility wrapper cho API cũ: một window duy nhất.
    """

    return _align_trake_fixed_windows(
        events_tr_r1,
        score_fn,
        video_id=video_id,
        windows=[(window_start, window_end)],
        step=step,
        min_gap=min_gap,
    )


def _align_prepared_dense_scorer(
    events_tr_r1: Mapping[str, dict],
    scorer: Any,
    *,
    video_id: str,
    windows: Sequence[tuple[float, float]],
    min_gap: int = 1,
) -> dict[str, Any]:
    """Chạy DP trực tiếp trên dense frames thật đã được scorer encode.

    Production không đi qua lưới timestamp giả rồi map nearest-frame nữa.
    Vì vậy strict-increasing áp dụng trực tiếp lên vị trí dense frame và
    output giữ được ``frame_idx`` thật.
    """

    event_ids = list(events_tr_r1.keys())
    frames = list(getattr(scorer, "frames", []))
    score_matrix = getattr(scorer, "score_matrix", None)

    if not frames:
        raise ValueError(
            f"Không có dense frame thật cho video={video_id!r}"
        )

    if score_matrix is None:
        raise ValueError("Dense scorer chưa có score_matrix")

    for left, right in zip(frames, frames[1:]):
        left_frame_idx = int(left.frame_idx)
        right_frame_idx = int(right.frame_idx)
        left_pts_time = float(left.pts_time)
        right_pts_time = float(right.pts_time)

        # Timestamp bằng nhau là hợp lệ khi frame map bị lượng tử/làm tròn;
        # output TRAKE chỉ yêu cầu frame_idx thật tăng nghiêm ngặt.
        if left_frame_idx >= right_frame_idx:
            raise ValueError(
                "Dense frame_idx không tăng nghiêm ngặt cho "
                f"video={video_id!r}: "
                f"{left_frame_idx} -> {right_frame_idx}"
            )

        if left_pts_time > right_pts_time:
            raise ValueError(
                "Dense pts_time bị giảm cho "
                f"video={video_id!r}: "
                f"frame_idx={left_frame_idx}, pts_time={left_pts_time} "
                f"-> frame_idx={right_frame_idx}, "
                f"pts_time={right_pts_time}"
            )

    matrix = score_matrix.tolist()
    chosen_positions, total_score = solve_strict_increasing_path(
        matrix,
        min_gap=min_gap,
    )

    chosen_frames = [
        frames[position]
        for position in chosen_positions
    ]

    chosen_frame_idx = [
        int(frame.frame_idx)
        for frame in chosen_frames
    ]

    if any(
        left >= right
        for left, right in zip(
            chosen_frame_idx,
            chosen_frame_idx[1:],
        )
    ):
        raise ValueError(
            "Dense DP vi phạm invariant frame_idx tăng nghiêm ngặt"
        )

    return {
        "video_id": str(video_id),
        "windows": [
            [float(start), float(end)]
            for start, end in windows
        ],
        "dense_frame_times": [
            float(frame.pts_time)
            for frame in frames
        ],
        "chosen_positions": [
            int(position)
            for position in chosen_positions
        ],
        "chosen_frame_idx": [
            int(frame_idx)
            for frame_idx in chosen_frame_idx
        ],
        "chosen_times": {
            event_id: float(frame.pts_time)
            for event_id, frame in zip(event_ids, chosen_frames)
        },
        "total_score": float(total_score),
    }


def align_trake_query(
    events_tr_r1: dict,
    score_fn: Callable[[str, float], float],
    *,
    step: float = 0.16,
    min_gap: int = 1,
    rrf_k: int = 60,
    window_padding_seconds: float = 0.0,
) -> dict:
    """
    Pure/testable TR-R2 alignment.

    `score_fn` được inject từ bên ngoài.
    Unit test có thể dùng fake scorer mà không cần load CLIP-L.

    TR-R2 dùng nhiều temporal windows thay vì một span
    min(start) -> max(end).
    """

    events_regions = {
        event_id: data["regions"]
        for event_id, data in events_tr_r1.items()
    }

    # ------------------------------------------------------------------
    # 1. Chọn video duy nhất bằng RRF.
    # ------------------------------------------------------------------

    video_id = chon_video_rrf(
        events_regions,
        k=rrf_k,
    )

    # ------------------------------------------------------------------
    # 2. Lấy các coarse windows của video.
    #
    # Ví dụ:
    #
    #   [97.64, 98.28]
    #   [1385.69, 1386.19]
    #
    # thay vì:
    #
    #   [97.64, 1386.19]
    # ------------------------------------------------------------------

    windows = gop_cac_cua_so_theo_video(
        events_regions,
        video_id,
    )

    # ------------------------------------------------------------------
    # 3. Padding từng window độc lập.
    # ------------------------------------------------------------------

    windows = [
        (
            max(0.0, start - window_padding_seconds),
            end + window_padding_seconds,
        )
        for start, end in windows
    ]

    # ------------------------------------------------------------------
    # 4. DP trên toàn bộ candidate times của các windows.
    # ------------------------------------------------------------------

    return _align_trake_fixed_windows(
        events_tr_r1,
        score_fn,
        video_id=video_id,
        windows=windows,
        step=step,
        min_gap=min_gap,
    )


# ============================================================================
# PRODUCTION TR-R1 -> TR-R2
# ============================================================================


def _tr_r1_results_to_events(
    tr_r1_results: Sequence,
) -> dict[str, dict[str, Any]]:
    """Adapter duy nhất từ TRR1Result sang contract nội bộ của TR-R2."""

    events: dict[str, dict[str, Any]] = {}

    for result in tr_r1_results:
        event_id = str(result.event_id)

        if event_id in events:
            raise ValueError(f"Trùng event_id trong TR-R2: {event_id!r}")

        events[event_id] = {
            "text": str(result.text),
            "relation": result.relation,
            "regions": [
                {
                    "video_id": str(region.video_id),
                    "start_time": float(region.start_time),
                    "end_time": float(region.end_time),
                    "score": float(region.score),
                    "hits": region.hits,
                }
                for region in result.regions
            ],
        }

    return events


def _select_sparse_video_and_windows(
    events_tr_r1: Mapping[str, dict[str, Any]],
    *,
    rrf_k: int,
    video_beam_size: int,
    sparse_min_gap: int,
    sparse_window_padding_seconds: float,
    window_padding_seconds: float,
    sparse_use_query_expansion: bool,
    sparse_max_query_variants: int,
) -> tuple[str, list[tuple[float, float]], dict[str, Any]]:
    """Candidate beam -> video-local sparse DP -> anchor windows."""

    if video_beam_size <= 0:
        raise ValueError("video_beam_size phải > 0")

    if sparse_min_gap < 1:
        raise ValueError("sparse_min_gap phải >= 1")

    if sparse_window_padding_seconds < 0:
        raise ValueError("sparse_window_padding_seconds phải >= 0")

    if window_padding_seconds < 0:
        raise ValueError("window_padding_seconds phải >= 0")

    events_regions = {
        event_id: data["regions"]
        for event_id, data in events_tr_r1.items()
    }
    event_texts = {
        event_id: str(data["text"])
        for event_id, data in events_tr_r1.items()
    }

    candidate_video_ids = rank_video_candidates_rrf(
        events_regions,
        k=rrf_k,
        limit=video_beam_size,
    )

    sparse_selection = select_video_by_sparse_dp(
        event_texts,
        candidate_video_ids,
        min_gap=sparse_min_gap,
        use_query_expansion=sparse_use_query_expansion,
        max_query_variants=sparse_max_query_variants,
    )

    video_id = str(sparse_selection["video_id"])
    chosen_times = sparse_selection.get("chosen_times")

    if not isinstance(chosen_times, dict):
        raise ValueError("Sparse selector không trả chosen_times dạng dict")

    missing_events = [
        event_id
        for event_id in events_tr_r1
        if event_id not in chosen_times
    ]

    if missing_events:
        raise ValueError(
            "Sparse selector thiếu event: " + ", ".join(missing_events)
        )

    windows = windows_from_anchor_times(
        [
            float(chosen_times[event_id])
            for event_id in events_tr_r1
        ],
        padding_seconds=sparse_window_padding_seconds,
    )

    if window_padding_seconds > 0:
        windows = [
            (
                max(0.0, start - window_padding_seconds),
                end + window_padding_seconds,
            )
            for start, end in windows
        ]

    sparse_selection = dict(sparse_selection)
    sparse_selection["candidate_video_ids"] = candidate_video_ids
    sparse_selection["windows"] = [
        [float(start), float(end)]
        for start, end in windows
    ]

    return video_id, windows, sparse_selection


def run_trake_r2_sparse_selection(
    tr_r1_results: Sequence,
    *,
    rrf_k: int = 60,
    video_beam_size: int = 12,
    sparse_min_gap: int = 1,
    sparse_window_padding_seconds: float = 5.0,
    window_padding_seconds: float = 0.0,
    sparse_use_query_expansion: bool = True,
    sparse_max_query_variants: int = 4,
) -> dict[str, Any]:
    """Chạy riêng video-local sparse stage, không cần dense frames."""

    if not tr_r1_results:
        raise ValueError(
            "TR-R2 sparse selection yêu cầu ít nhất một TRR1Result."
        )

    events_tr_r1 = _tr_r1_results_to_events(tr_r1_results)
    video_id, windows, selection = _select_sparse_video_and_windows(
        events_tr_r1,
        rrf_k=rrf_k,
        video_beam_size=video_beam_size,
        sparse_min_gap=sparse_min_gap,
        sparse_window_padding_seconds=sparse_window_padding_seconds,
        window_padding_seconds=window_padding_seconds,
        sparse_use_query_expansion=sparse_use_query_expansion,
        sparse_max_query_variants=sparse_max_query_variants,
    )

    return {
        "video_id": video_id,
        "event_ids": list(events_tr_r1.keys()),
        "windows": windows,
        "selection": selection,
    }


def run_trake_r2(
    tr_r1_results: Sequence,
    *,
    step: float = 0.16,
    min_gap: int = 1,
    rrf_k: int = 60,
    window_padding_seconds: float = 0.0,
    batch_size: int = 16,
    video_beam_size: int = 12,
    sparse_min_gap: int = 1,
    sparse_window_padding_seconds: float = 5.0,
    sparse_use_query_expansion: bool = True,
    sparse_max_query_variants: int = 4,
) -> dict:
    """
    Production orchestration:

        TR-R1Result
            -> Top-B video beam
            -> rescore toàn bộ sparse keyframe trong từng video
            -> strict-increasing sparse DP chọn video + anchors
            -> local dense windows quanh anchors
            -> dense CLIP-L scorer
            -> strict-increasing dense DP

    `tr_r1_results` là output trực tiếp của:
        tim_nhieu_su_kien(...)

    CLIP-L runtime được lấy từ
    `_get_trr1_clip_l_runtime()` bên trong `DenseClipLScorer`,
    nên không load model lần thứ hai.

    Quan trọng:
        - video/window selection không dùng GT
        - sparse image vectors được tái sử dụng, không encode ảnh lại
        - không dùng GT
    """

    if not tr_r1_results:
        raise ValueError(
            "TR-R2 yêu cầu ít nhất một TRR1Result."
        )

    events_tr_r1 = _tr_r1_results_to_events(tr_r1_results)

    video_id, windows, sparse_selection = (
        _select_sparse_video_and_windows(
            events_tr_r1,
            rrf_k=rrf_k,
            video_beam_size=video_beam_size,
            sparse_min_gap=sparse_min_gap,
            sparse_window_padding_seconds=(
                sparse_window_padding_seconds
            ),
            window_padding_seconds=window_padding_seconds,
            sparse_use_query_expansion=(
                sparse_use_query_expansion
            ),
            sparse_max_query_variants=(
                sparse_max_query_variants
            ),
        )
    )

    event_texts = {
        event_id: data["text"]
        for event_id, data in events_tr_r1.items()
    }

    scorer, _score_fn = build_dense_score_fn(
        video_id=video_id,
        windows=windows,
        event_texts=event_texts,
        batch_size=batch_size,
    )

    alignment = _align_prepared_dense_scorer(
        events_tr_r1,
        scorer,
        video_id=video_id,
        windows=windows,
        min_gap=min_gap,
    )

    alignment["sparse_selection"] = sparse_selection
    alignment["step"] = float(step)

    return alignment

def run_trake_r2_diagnostics(
    tr_r1_results: Sequence,
    *,
    step: float = 0.16,
    min_gap: int = 1,
    rrf_k: int = 60,
    window_padding_seconds: float = 0.0,
    batch_size: int = 16,
    video_beam_size: int = 12,
    sparse_min_gap: int = 1,
    sparse_window_padding_seconds: float = 5.0,
    sparse_use_query_expansion: bool = True,
    sparse_max_query_variants: int = 4,
) -> dict:
    """
    Phiên bản diagnostics của TR-R2 dành cho TR-E2.

    Giữ nguyên toàn bộ logic production của run_trake_r2(),
    nhưng trả thêm DenseClipLScorer để TR-E2 có thể đọc:

        - scorer.frames
        - scorer.score_matrix

    Không dùng ground truth.
    Không thay đổi contract của run_trake_r2() cũ.
    """

    if not tr_r1_results:
        raise ValueError(
            "TR-R2 diagnostics yêu cầu ít nhất một TRR1Result."
        )

    events_tr_r1 = _tr_r1_results_to_events(tr_r1_results)
    event_ids = list(events_tr_r1.keys())

    video_id, windows, sparse_selection = (
        _select_sparse_video_and_windows(
            events_tr_r1,
            rrf_k=rrf_k,
            video_beam_size=video_beam_size,
            sparse_min_gap=sparse_min_gap,
            sparse_window_padding_seconds=(
                sparse_window_padding_seconds
            ),
            window_padding_seconds=window_padding_seconds,
            sparse_use_query_expansion=(
                sparse_use_query_expansion
            ),
            sparse_max_query_variants=(
                sparse_max_query_variants
            ),
        )
    )

    event_texts = {
        event_id: data["text"]
        for event_id, data in events_tr_r1.items()
    }

    scorer, _score_fn = build_dense_score_fn(
        video_id=video_id,
        windows=windows,
        event_texts=event_texts,
        batch_size=batch_size,
    )

    alignment = _align_prepared_dense_scorer(
        events_tr_r1,
        scorer,
        video_id=video_id,
        windows=windows,
        min_gap=min_gap,
    )

    alignment["sparse_selection"] = sparse_selection
    alignment["step"] = float(step)

    # ------------------------------------------------------------------
    # 8. Trả thêm scorer cho TR-E2.
    # ------------------------------------------------------------------

    return {
        "video_id": video_id,
        "event_ids": event_ids,
        "windows": windows,
        "alignment": alignment,
        "scorer": scorer,
        "sparse_selection": sparse_selection,
    }
