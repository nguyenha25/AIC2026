from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

import numpy as np

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
    score_video_candidates_rrf,
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
    anchor_times: Mapping[str, float] | None = None,
    anchor_max_drift_seconds: float = 2.0,
    anchor_penalty_per_second: float = 0.01,
    anchor_forward_reward_per_second: float = 0.0,
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

    raw_matrix = np.asarray(score_matrix, dtype=np.float64)

    if raw_matrix.shape != (len(event_ids), len(frames)):
        raise ValueError(
            "Dense score_matrix sai shape: "
            f"nhận {raw_matrix.shape}, cần "
            f"({len(event_ids)}, {len(frames)})"
        )

    if anchor_max_drift_seconds <= 0:
        raise ValueError("anchor_max_drift_seconds phải > 0")

    if anchor_penalty_per_second < 0:
        raise ValueError("anchor_penalty_per_second phải >= 0")

    if anchor_forward_reward_per_second < 0:
        raise ValueError("anchor_forward_reward_per_second phải >= 0")

    # Dense chỉ là bộ tinh chỉnh cục bộ quanh sparse anchor. Nếu cho mọi
    # event nhìn toàn bộ hợp các window, các câu mô tả gần nhau dễ cùng chọn
    # một cảnh nền và DP chỉ cách nhau đúng một dense frame (DP collapse).
    # Mask theo event ngăn lỗi đó mà vẫn giữ strict-increasing toàn chuỗi.
    anchor_constraint_applied = anchor_times is not None
    anchor_constraint_relaxed = False
    anchor_distances = np.zeros_like(raw_matrix)
    adjusted_matrix = raw_matrix.copy()

    if anchor_times is not None:
        missing_anchor_events = [
            event_id
            for event_id in event_ids
            if event_id not in anchor_times
        ]
        if missing_anchor_events:
            raise ValueError(
                "Thiếu sparse anchor cho event: "
                + ", ".join(missing_anchor_events)
            )

        frame_times = np.asarray(
            [float(frame.pts_time) for frame in frames],
            dtype=np.float64,
        )
        for event_pos, event_id in enumerate(event_ids):
            anchor_time = float(anchor_times[event_id])
            distances = np.abs(frame_times - anchor_time)
            anchor_distances[event_pos] = distances
            adjusted_matrix[event_pos] -= (
                float(anchor_penalty_per_second) * distances
            )
            # Mô tả hành động thường đạt CLIP peak ở khung chuẩn bị ngay
            # trước khoảnh khắc GT. Reward có hướng cho phép ablation một
            # dịch chuyển mềm về phía trước, không dùng GT khi inference.
            adjusted_matrix[event_pos] += (
                float(anchor_forward_reward_per_second)
                * (frame_times - anchor_time)
            )
            adjusted_matrix[
                event_pos,
                distances > float(anchor_max_drift_seconds) + 1e-9,
            ] = float("-inf")

    matrix = adjusted_matrix.tolist()
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

    raw_total_score = float(
        sum(
            raw_matrix[event_pos, frame_pos]
            for event_pos, frame_pos in enumerate(chosen_positions)
        )
    )

    anchor_drift_seconds = {
        event_id: (
            abs(
                float(frame.pts_time)
                - float(anchor_times[event_id])
            )
            if anchor_times is not None
            else 0.0
        )
        for event_id, frame in zip(event_ids, chosen_frames)
    }

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
        "raw_total_score": raw_total_score,
        "anchor_constraint": {
            "applied": anchor_constraint_applied,
            "relaxed": anchor_constraint_relaxed,
            "max_drift_seconds": float(anchor_max_drift_seconds),
            "penalty_per_second": float(anchor_penalty_per_second),
            "forward_reward_per_second": float(
                anchor_forward_reward_per_second
            ),
            "chosen_drift_seconds": anchor_drift_seconds,
        },
    }


def align_dense_temporal_profiles(
    events_tr_r1: Mapping[str, dict],
    scorer: Any,
    *,
    video_id: str,
    windows: Sequence[tuple[float, float]],
    anchor_times: Mapping[str, float],
    profiles: Sequence[Mapping[str, Any]],
    min_gap: int = 1,
) -> list[dict[str, Any]]:
    """Sinh nhiều temporal path trên cùng scorer, không encode lại frame.

    Mỗi profile cần ``name``, ``max_drift_seconds``,
    ``anchor_penalty_per_second`` và ``forward_reward_per_second``.
    Hàm không dùng GT; thứ tự output giữ nguyên thứ tự profile đầu vào.
    """

    results: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for profile in profiles:
        name = str(profile.get("name", "")).strip()
        if not name:
            raise ValueError("Dense temporal profile thiếu name")
        if name in seen_names:
            raise ValueError(f"Trùng dense temporal profile name: {name!r}")
        seen_names.add(name)

        config = {
            "max_drift_seconds": float(profile["max_drift_seconds"]),
            "anchor_penalty_per_second": float(
                profile["anchor_penalty_per_second"]
            ),
            "forward_reward_per_second": float(
                profile["forward_reward_per_second"]
            ),
        }
        alignment = _align_prepared_dense_scorer(
            events_tr_r1,
            scorer,
            video_id=video_id,
            windows=windows,
            min_gap=min_gap,
            anchor_times=anchor_times,
            anchor_max_drift_seconds=config["max_drift_seconds"],
            anchor_penalty_per_second=config["anchor_penalty_per_second"],
            anchor_forward_reward_per_second=config[
                "forward_reward_per_second"
            ],
        )
        results.append(
            {
                "name": name,
                "config": config,
                "alignment": alignment,
            }
        )

    return results


def enumerate_binary_profile_hybrids(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    *,
    event_ids: Sequence[str],
    max_hybrids: int = 64,
) -> list[dict[str, Any]]:
    """Trộn event-wise hai path và chỉ giữ chuỗi frame tăng nghiêm ngặt.

    Hai path nguyên bản không được lặp lại. Hybrid được xếp theo số event lấy
    từ secondary tăng dần, nên các sửa đổi nhỏ quanh primary xuất hiện trước.
    Không dùng GT hoặc score mới và không gọi model.
    """

    if max_hybrids < 0:
        raise ValueError("max_hybrids phải >= 0")

    normalized_event_ids = [str(event_id) for event_id in event_ids]
    n_events = len(normalized_event_ids)
    if n_events == 0 or max_hybrids == 0:
        return []

    def alignment_of(profile: Mapping[str, Any]) -> Mapping[str, Any]:
        alignment = profile.get("alignment", profile)
        if not isinstance(alignment, Mapping):
            raise ValueError("Temporal profile thiếu alignment")
        return alignment

    primary_alignment = alignment_of(primary)
    secondary_alignment = alignment_of(secondary)
    primary_frames = [int(value) for value in primary_alignment["chosen_frame_idx"]]
    secondary_frames = [
        int(value) for value in secondary_alignment["chosen_frame_idx"]
    ]
    primary_times = primary_alignment["chosen_times"]
    secondary_times = secondary_alignment["chosen_times"]

    if len(primary_frames) != n_events or len(secondary_frames) != n_events:
        raise ValueError("Temporal profile không khớp số event")

    hybrids: list[dict[str, Any]] = []
    seen_paths: set[tuple[int, ...]] = {
        tuple(primary_frames),
        tuple(secondary_frames),
    }

    for secondary_count in range(1, n_events):
        for secondary_positions in combinations(range(n_events), secondary_count):
            selected = set(secondary_positions)
            frame_idx = [
                secondary_frames[pos] if pos in selected else primary_frames[pos]
                for pos in range(n_events)
            ]
            path_key = tuple(frame_idx)
            if path_key in seen_paths:
                continue
            if any(left >= right for left, right in zip(frame_idx, frame_idx[1:])):
                continue

            seen_paths.add(path_key)
            choice_names = [
                "secondary" if pos in selected else "primary"
                for pos in range(n_events)
            ]
            chosen_times = {
                event_id: float(
                    secondary_times[event_id]
                    if pos in selected
                    else primary_times[event_id]
                )
                for pos, event_id in enumerate(normalized_event_ids)
            }
            hybrids.append(
                {
                    "name": "hybrid_" + "".join(
                        "N" if choice == "secondary" else "W"
                        for choice in choice_names
                    ),
                    "chosen_frame_idx": frame_idx,
                    "chosen_times": chosen_times,
                    "profile_choices": {
                        event_id: choice
                        for event_id, choice in zip(
                            normalized_event_ids,
                            choice_names,
                        )
                    },
                    "secondary_event_count": secondary_count,
                }
            )
            if len(hybrids) >= max_hybrids:
                return hybrids

    return hybrids


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
    candidate_video_ids: Sequence[str] | None,
    sparse_min_gap: int,
    sparse_window_padding_seconds: float,
    window_padding_seconds: float,
    sparse_use_query_expansion: bool,
    sparse_max_query_variants: int,
    region_prior_weight: float,
    region_decay_seconds: float,
    video_prior_weight: float,
    order_gain_weight: float,
    span_penalty_weight: float,
    span_scale_seconds: float,
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

    if candidate_video_ids is None:
        ranked_candidate_video_ids = rank_video_candidates_rrf(
            events_regions,
            k=rrf_k,
            limit=video_beam_size,
        )
    else:
        ranked_candidate_video_ids = []
        for raw_video_id in candidate_video_ids:
            video_id = str(raw_video_id or "").strip()
            if video_id and video_id not in ranked_candidate_video_ids:
                ranked_candidate_video_ids.append(video_id)
            if len(ranked_candidate_video_ids) >= video_beam_size:
                break
        if not ranked_candidate_video_ids:
            raise ValueError("candidate_video_ids override không được rỗng")

    raw_video_priors = score_video_candidates_rrf(
        events_regions,
        k=rrf_k,
    )
    video_priors = {
        video_id: raw_video_priors[video_id]
        for video_id in ranked_candidate_video_ids
        if video_id in raw_video_priors
    }

    region_evidence: dict[
        str,
        dict[str, list[dict[str, Any]]],
    ] = {}

    for event_id, regions in events_regions.items():
        for region in regions:
            video_id = str(region["video_id"])

            if video_id not in ranked_candidate_video_ids:
                continue

            region_evidence.setdefault(video_id, {}).setdefault(
                str(event_id),
                [],
            ).append(dict(region))

    sparse_selection = select_video_by_sparse_dp(
        event_texts,
        ranked_candidate_video_ids,
        min_gap=sparse_min_gap,
        use_query_expansion=sparse_use_query_expansion,
        max_query_variants=sparse_max_query_variants,
        video_priors=video_priors,
        region_evidence=region_evidence,
        region_prior_weight=region_prior_weight,
        region_decay_seconds=region_decay_seconds,
        video_prior_weight=video_prior_weight,
        order_gain_weight=order_gain_weight,
        span_penalty_weight=span_penalty_weight,
        span_scale_seconds=span_scale_seconds,
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
    sparse_selection["candidate_video_ids"] = ranked_candidate_video_ids
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
    candidate_video_ids: Sequence[str] | None = None,
    sparse_min_gap: int = 1,
    sparse_window_padding_seconds: float = 5.0,
    window_padding_seconds: float = 0.0,
    sparse_use_query_expansion: bool = True,
    sparse_max_query_variants: int = 4,
    region_prior_weight: float = 0.0,
    region_decay_seconds: float = 8.0,
    video_prior_weight: float = 0.0,
    order_gain_weight: float = 0.0,
    span_penalty_weight: float = 0.041,
    span_scale_seconds: float = 60.0,
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
        candidate_video_ids=candidate_video_ids,
        sparse_min_gap=sparse_min_gap,
        sparse_window_padding_seconds=sparse_window_padding_seconds,
        window_padding_seconds=window_padding_seconds,
        sparse_use_query_expansion=sparse_use_query_expansion,
        sparse_max_query_variants=sparse_max_query_variants,
        region_prior_weight=region_prior_weight,
        region_decay_seconds=region_decay_seconds,
        video_prior_weight=video_prior_weight,
        order_gain_weight=order_gain_weight,
        span_penalty_weight=span_penalty_weight,
        span_scale_seconds=span_scale_seconds,
    )

    return {
        "video_id": video_id,
        "event_ids": list(events_tr_r1.keys()),
        "windows": windows,
        "selection": selection,
    }


def build_sparse_ranked_candidates(
    sparse_selection: Mapping[str, Any],
    event_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Tạo fallback submission hợp lệ khi dense frame không khả dụng.

    Candidate sparse đã chứa ``frame_idx`` thật từ frame map. Hàm chỉ giữ
    chuỗi đủ event, tăng nghiêm ngặt và không trùng ``(video, path)``. Video
    được chọn bởi sparse luôn được thử trước, sau đó mới tới phần còn lại của
    beam; không dùng tình trạng có/thiếu dense làm tín hiệu đổi video.
    """

    expected_events = len(event_ids)
    ranked: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    def add(candidate: Mapping[str, Any], *, sparse_rank: int) -> None:
        video_id = str(candidate.get("video_id", ""))
        frame_idx = [int(value) for value in candidate.get("chosen_frame_idx", [])]
        key = (video_id, tuple(frame_idx))

        if not video_id or key in seen or len(frame_idx) != expected_events:
            return
        if any(left >= right for left, right in zip(frame_idx, frame_idx[1:])):
            return

        seen.add(key)
        ranked.append(
            {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "score": float(
                    candidate.get(
                        "final_score",
                        candidate.get("mean_score", candidate.get("total_score", 0.0)),
                    )
                ),
                "source": (
                    "sparse_dp_selected_no_dense"
                    if sparse_rank == 1
                    else "sparse_dp_no_dense"
                ),
                "sparse_rank": sparse_rank,
            }
        )

    # Selection top-level là nguồn chân lý cho sparse top-1.
    add(sparse_selection, sparse_rank=1)

    for sparse_rank, candidate in enumerate(
        sparse_selection.get("candidate_scores", []),
        start=1,
    ):
        if isinstance(candidate, Mapping):
            add(candidate, sparse_rank=sparse_rank)

    return ranked


def _candidate_windows(
    candidate: Mapping[str, Any],
    event_ids: Sequence[str],
    *,
    sparse_window_padding_seconds: float,
    window_padding_seconds: float,
) -> list[tuple[float, float]]:
    chosen_times = candidate.get("chosen_times")

    if not isinstance(chosen_times, Mapping):
        raise ValueError("Sparse candidate thiếu chosen_times")

    windows = windows_from_anchor_times(
        [float(chosen_times[event_id]) for event_id in event_ids],
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

    return windows


def _run_dense_rerank(
    events_tr_r1: Mapping[str, dict[str, Any]],
    sparse_selection: Mapping[str, Any],
    *,
    step: float,
    min_gap: int,
    batch_size: int,
    sparse_window_padding_seconds: float,
    window_padding_seconds: float,
    adaptive_dense_rerank: bool,
    dense_rerank_top_k: int,
    dense_rerank_margin: float,
    dense_sparse_prior_weight: float,
    dense_anchor_max_drift_seconds: float,
    dense_anchor_penalty_per_second: float,
    dense_anchor_forward_reward_per_second: float,
) -> dict[str, Any]:
    """Dense-rerank top sparse candidates khi top-1 chưa đủ chắc chắn.

    Candidate phụ thiếu dense frame không làm hỏng luồng cũ: pipeline ghi
    diagnostic rồi giữ top-1 sparse. Candidate top-1 vẫn bắt buộc có dense
    frame vì TR-E2 cần scorer thật để refine boundary.
    """

    if dense_rerank_top_k <= 0:
        raise ValueError("dense_rerank_top_k phải > 0")

    if dense_rerank_margin < 0:
        raise ValueError("dense_rerank_margin phải >= 0")

    if dense_sparse_prior_weight < 0:
        raise ValueError("dense_sparse_prior_weight phải >= 0")

    event_ids = list(events_tr_r1.keys())
    event_texts = {
        event_id: str(data["text"])
        for event_id, data in events_tr_r1.items()
    }
    sparse_candidates = [
        dict(candidate)
        for candidate in sparse_selection.get("candidate_scores", [])
    ]

    if not sparse_candidates:
        sparse_candidates = [dict(sparse_selection)]

    top_n = 1
    sparse_margin = float("inf")

    if len(sparse_candidates) > 1:
        sparse_margin = (
            float(sparse_candidates[0].get("final_score", 0.0))
            - float(sparse_candidates[1].get("final_score", 0.0))
        )

        if adaptive_dense_rerank and sparse_margin < dense_rerank_margin:
            top_n = min(dense_rerank_top_k, len(sparse_candidates))

    dense_candidates: list[dict[str, Any]] = []

    for sparse_rank, candidate in enumerate(
        sparse_candidates[:top_n],
        start=1,
    ):
        video_id = str(candidate["video_id"])
        windows = _candidate_windows(
            candidate,
            event_ids,
            sparse_window_padding_seconds=sparse_window_padding_seconds,
            window_padding_seconds=window_padding_seconds,
        )

        try:
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
                anchor_times=candidate.get("chosen_times"),
                anchor_max_drift_seconds=dense_anchor_max_drift_seconds,
                anchor_penalty_per_second=dense_anchor_penalty_per_second,
                anchor_forward_reward_per_second=(
                    dense_anchor_forward_reward_per_second
                ),
            )
            dense_mean_score = float(alignment["total_score"]) / len(
                event_ids
            )
            rerank_score = (
                dense_mean_score
                + dense_sparse_prior_weight
                * float(candidate.get("final_score", candidate.get("mean_score", 0.0)))
            )
            dense_candidates.append(
                {
                    "video_id": video_id,
                    "sparse_rank": sparse_rank,
                    "windows": windows,
                    "sparse": candidate,
                    "scorer": scorer,
                    "alignment": alignment,
                    "dense_mean_score": dense_mean_score,
                    "rerank_score": float(rerank_score),
                    "status": "ok",
                }
            )
        except (FileNotFoundError, ValueError) as exc:
            dense_candidates.append(
                {
                    "video_id": video_id,
                    "sparse_rank": sparse_rank,
                    "windows": windows,
                    "sparse": candidate,
                    "scorer": None,
                    "alignment": None,
                    "dense_mean_score": None,
                    "rerank_score": None,
                    "status": "dense_unavailable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    top1 = dense_candidates[0]

    if top1["status"] != "ok":
        raise ValueError(
            "Candidate sparse top-1 không có dense frame: "
            + str(top1.get("error", "unknown error"))
        )

    # Chỉ rerank khi mọi candidate dự kiến đều có dense evidence. Nếu thiếu
    # candidate phụ, giữ nguyên top-1 sparse để việc thiếu file không trở thành
    # một tín hiệu chọn video giả.
    if all(candidate["status"] == "ok" for candidate in dense_candidates):
        winner = max(
            dense_candidates,
            key=lambda candidate: (
                float(candidate["rerank_score"]),
                -int(candidate["sparse_rank"]),
            ),
        )
        rerank_applied = len(dense_candidates) > 1
    else:
        winner = top1
        rerank_applied = False

    alignment = dict(winner["alignment"])
    alignment["sparse_selection"] = dict(sparse_selection)
    alignment["step"] = float(step)

    dense_summary = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"scorer", "alignment"}
        }
        | {
            "chosen_frame_idx": (
                list(candidate["alignment"].get("chosen_frame_idx", []))
                if isinstance(candidate.get("alignment"), Mapping)
                else list(candidate["sparse"].get("chosen_frame_idx", []))
            )
        }
        for candidate in dense_candidates
    ]

    return {
        "video_id": str(winner["video_id"]),
        "windows": list(winner["windows"]),
        "alignment": alignment,
        "scorer": winner["scorer"],
        "sparse_margin": sparse_margin,
        "dense_candidates_requested": top_n,
        "dense_rerank_applied": rerank_applied,
        "dense_candidates": dense_summary,
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
    candidate_video_ids: Sequence[str] | None = None,
    sparse_min_gap: int = 1,
    sparse_window_padding_seconds: float = 5.0,
    sparse_use_query_expansion: bool = True,
    sparse_max_query_variants: int = 4,
    region_prior_weight: float = 0.0,
    region_decay_seconds: float = 8.0,
    video_prior_weight: float = 0.0,
    order_gain_weight: float = 0.0,
    span_penalty_weight: float = 0.041,
    span_scale_seconds: float = 60.0,
    adaptive_dense_rerank: bool = True,
    dense_rerank_top_k: int = 3,
    dense_rerank_margin: float = 0.015,
    dense_sparse_prior_weight: float = 0.25,
    dense_anchor_max_drift_seconds: float = 2.0,
    dense_anchor_penalty_per_second: float = 0.01,
    dense_anchor_forward_reward_per_second: float = 0.0,
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
            candidate_video_ids=candidate_video_ids,
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
            region_prior_weight=region_prior_weight,
            region_decay_seconds=region_decay_seconds,
            video_prior_weight=video_prior_weight,
            order_gain_weight=order_gain_weight,
            span_penalty_weight=span_penalty_weight,
            span_scale_seconds=span_scale_seconds,
        )
    )

    dense = _run_dense_rerank(
        events_tr_r1,
        sparse_selection,
        step=step,
        min_gap=min_gap,
        batch_size=batch_size,
        sparse_window_padding_seconds=sparse_window_padding_seconds,
        window_padding_seconds=window_padding_seconds,
        adaptive_dense_rerank=adaptive_dense_rerank,
        dense_rerank_top_k=dense_rerank_top_k,
        dense_rerank_margin=dense_rerank_margin,
        dense_sparse_prior_weight=dense_sparse_prior_weight,
        dense_anchor_max_drift_seconds=dense_anchor_max_drift_seconds,
        dense_anchor_penalty_per_second=dense_anchor_penalty_per_second,
        dense_anchor_forward_reward_per_second=(
            dense_anchor_forward_reward_per_second
        ),
    )

    alignment = dict(dense["alignment"])
    alignment["dense_rerank"] = {
        "sparse_margin": dense["sparse_margin"],
        "candidates_requested": dense["dense_candidates_requested"],
        "applied": dense["dense_rerank_applied"],
        "candidates": dense["dense_candidates"],
    }
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
    candidate_video_ids: Sequence[str] | None = None,
    sparse_min_gap: int = 1,
    sparse_window_padding_seconds: float = 5.0,
    sparse_use_query_expansion: bool = True,
    sparse_max_query_variants: int = 4,
    region_prior_weight: float = 0.0,
    region_decay_seconds: float = 8.0,
    video_prior_weight: float = 0.0,
    order_gain_weight: float = 0.0,
    span_penalty_weight: float = 0.041,
    span_scale_seconds: float = 60.0,
    adaptive_dense_rerank: bool = True,
    dense_rerank_top_k: int = 3,
    dense_rerank_margin: float = 0.015,
    dense_sparse_prior_weight: float = 0.25,
    dense_anchor_max_drift_seconds: float = 2.0,
    dense_anchor_penalty_per_second: float = 0.01,
    dense_anchor_forward_reward_per_second: float = 0.0,
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
            candidate_video_ids=candidate_video_ids,
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
            region_prior_weight=region_prior_weight,
            region_decay_seconds=region_decay_seconds,
            video_prior_weight=video_prior_weight,
            order_gain_weight=order_gain_weight,
            span_penalty_weight=span_penalty_weight,
            span_scale_seconds=span_scale_seconds,
        )
    )

    dense = _run_dense_rerank(
        events_tr_r1,
        sparse_selection,
        step=step,
        min_gap=min_gap,
        batch_size=batch_size,
        sparse_window_padding_seconds=sparse_window_padding_seconds,
        window_padding_seconds=window_padding_seconds,
        adaptive_dense_rerank=adaptive_dense_rerank,
        dense_rerank_top_k=dense_rerank_top_k,
        dense_rerank_margin=dense_rerank_margin,
        dense_sparse_prior_weight=dense_sparse_prior_weight,
        dense_anchor_max_drift_seconds=dense_anchor_max_drift_seconds,
        dense_anchor_penalty_per_second=dense_anchor_penalty_per_second,
        dense_anchor_forward_reward_per_second=(
            dense_anchor_forward_reward_per_second
        ),
    )

    video_id = str(dense["video_id"])
    windows = list(dense["windows"])
    alignment = dict(dense["alignment"])
    scorer = dense["scorer"]

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
        "sparse_margin": dense["sparse_margin"],
        "dense_candidates_requested": dense["dense_candidates_requested"],
        "dense_rerank_applied": dense["dense_rerank_applied"],
        "dense_candidates": dense["dense_candidates"],
    }
