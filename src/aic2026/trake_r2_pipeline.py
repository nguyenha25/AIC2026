from __future__ import annotations

from typing import Callable, Mapping, Sequence

from aic2026.trake_r2_dp import solve_strict_increasing_path
from aic2026.trake_r2_score import build_dense_score_fn
from aic2026.trake_r2_windows import (
    generate_dense_time_grid,
    chon_video_rrf,
    gop_cac_cua_so_theo_video,
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


def run_trake_r2(
    tr_r1_results: Sequence,
    *,
    step: float = 0.16,
    min_gap: int = 1,
    rrf_k: int = 60,
    window_padding_seconds: float = 0.0,
    batch_size: int = 16,
) -> dict:
    """
    Production orchestration:

        TR-R1Result
            -> normalize CoarseRegion
            -> chọn video
            -> chọn nhiều coarse windows
            -> dense CLIP-L scorer
            -> dense score matrix
            -> strict-increasing DP

    `tr_r1_results` là output trực tiếp của:
        tim_nhieu_su_kien(...)

    CLIP-L runtime được lấy từ
    `_get_trr1_clip_l_runtime()` bên trong `DenseClipLScorer`,
    nên không load model lần thứ hai.

    Quan trọng:
        - video selection chỉ dùng TR-R1
        - window selection chỉ dùng TR-R1
        - không dùng GT
    """

    if not tr_r1_results:
        raise ValueError(
            "TR-R2 yêu cầu ít nhất một TRR1Result."
        )

    # ------------------------------------------------------------------
    # 1. Adapter TR-R1Result -> contract của TR-R2.
    # ------------------------------------------------------------------

    events_tr_r1 = {
        result.event_id: {
            "text": result.text,
            "relation": result.relation,
            "regions": [
                {
                    "video_id": region.video_id,
                    "start_time": region.start_time,
                    "end_time": region.end_time,
                    "score": region.score,
                    "hits": region.hits,
                }
                for region in result.regions
            ],
        }
        for result in tr_r1_results
    }

    # ------------------------------------------------------------------
    # 2. Giữ nguyên thứ tự event của QueryPlan.
    # ------------------------------------------------------------------

    event_ids = list(events_tr_r1.keys())

    # ------------------------------------------------------------------
    # 3. Chọn video DUY NHẤT bằng RRF.
    # ------------------------------------------------------------------

    events_regions = {
        event_id: data["regions"]
        for event_id, data in events_tr_r1.items()
    }

    video_id = chon_video_rrf(
        events_regions,
        k=rrf_k,
    )

    # ------------------------------------------------------------------
    # 4. Chọn nhiều coarse windows của video.
    #
    # Đây là điểm thay đổi chính so với implementation cũ.
    # ------------------------------------------------------------------

    windows = gop_cac_cua_so_theo_video(
        events_regions,
        video_id,
    )

    # ------------------------------------------------------------------
    # 5. Padding từng window độc lập.
    # ------------------------------------------------------------------

    windows = [
        (
            max(0.0, start - window_padding_seconds),
            end + window_padding_seconds,
        )
        for start, end in windows
    ]

    # ------------------------------------------------------------------
    # 6. Event text -> dense CLIP-L scorer.
    #
    # Scorer phải encode dense frames thuộc TẤT CẢ windows,
    # nhưng chỉ load CLIP-L runtime một lần.
    # ------------------------------------------------------------------

    event_texts = {
        event_id: data["text"]
        for event_id, data in events_tr_r1.items()
    }

    _scorer, score_fn = build_dense_score_fn(
        video_id=video_id,
        windows=windows,
        event_texts=event_texts,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------
    # 7. Dense time grid + strict-increasing DP.
    # ------------------------------------------------------------------

    return _align_trake_fixed_windows(
        events_tr_r1,
        score_fn,
        video_id=video_id,
        windows=windows,
        step=step,
        min_gap=min_gap,
    )

def run_trake_r2_diagnostics(
    tr_r1_results: Sequence,
    *,
    step: float = 0.16,
    min_gap: int = 1,
    rrf_k: int = 60,
    window_padding_seconds: float = 0.0,
    batch_size: int = 16,
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

    # ------------------------------------------------------------------
    # 1. Adapter TR-R1Result -> contract của TR-R2.
    # ------------------------------------------------------------------

    events_tr_r1 = {
        result.event_id: {
            "text": result.text,
            "relation": result.relation,
            "regions": [
                {
                    "video_id": region.video_id,
                    "start_time": region.start_time,
                    "end_time": region.end_time,
                    "score": region.score,
                    "hits": region.hits,
                }
                for region in result.regions
            ],
        }
        for result in tr_r1_results
    }

    # ------------------------------------------------------------------
    # 2. Giữ nguyên thứ tự event của QueryPlan.
    # ------------------------------------------------------------------

    event_ids = list(events_tr_r1.keys())

    # ------------------------------------------------------------------
    # 3. Chọn video DUY NHẤT bằng RRF.
    # ------------------------------------------------------------------

    events_regions = {
        event_id: data["regions"]
        for event_id, data in events_tr_r1.items()
    }

    video_id = chon_video_rrf(
        events_regions,
        k=rrf_k,
    )

    # ------------------------------------------------------------------
    # 4. Chọn nhiều coarse windows của video.
    # ------------------------------------------------------------------

    windows = gop_cac_cua_so_theo_video(
        events_regions,
        video_id,
    )

    # ------------------------------------------------------------------
    # 5. Padding từng window độc lập.
    # ------------------------------------------------------------------

    windows = [
        (
            max(0.0, start - window_padding_seconds),
            end + window_padding_seconds,
        )
        for start, end in windows
    ]

    # ------------------------------------------------------------------
    # 6. Event text -> dense CLIP-L scorer.
    # ------------------------------------------------------------------

    event_texts = {
        event_id: data["text"]
        for event_id, data in events_tr_r1.items()
    }

    scorer, score_fn = build_dense_score_fn(
        video_id=video_id,
        windows=windows,
        event_texts=event_texts,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------
    # 7. Dense time grid + strict-increasing DP.
    # ------------------------------------------------------------------

    alignment = _align_trake_fixed_windows(
        events_tr_r1,
        score_fn,
        video_id=video_id,
        windows=windows,
        step=step,
        min_gap=min_gap,
    )

    # ------------------------------------------------------------------
    # 8. Trả thêm scorer cho TR-E2.
    # ------------------------------------------------------------------

    return {
        "video_id": video_id,
        "event_ids": event_ids,
        "windows": windows,
        "alignment": alignment,
        "scorer": scorer,
    }