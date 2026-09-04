"""
trake_r2_score.py
=================

TR-R2 video-local sparse selector và local dense CLIP-L scorer.

Mục tiêu
--------
Module có hai tầng:

    1. Top-B video candidate
       -> rescore toàn bộ sparse CLIP-L keyframes mỗi video
       -> strict-increasing DP chọn video + anchors;

    2. Sau khi chọn video, encode dense frames trong local windows
       quanh anchors để chạy dense DP/TR-E2.

Ở tầng dense, sau khi TR-R2 đã chọn được:

    video_id
    local temporal windows

module này:

    1. tìm dense frames trong các local temporal windows;
    2. lấy CLIP-L runtime ĐÃ LOAD từ TR-R1;
    3. encode text của từng event bằng cùng runtime;
    4. encode toàn bộ dense frames theo batch;
    5. tạo score matrix:

           S[event_i, frame_j]
             = cosine(text_i, image_j)

    6. cung cấp score_fn(event_id, pts_time) cho DP.

QUAN TRỌNG
----------
Không load model CLIP-L lần thứ hai.

Runtime được lấy từ:

    aic2026.trake_retrieval._get_trr1_clip_l_runtime()

Runtime này trả:

    model, tokenizer, device, index, ids

Sparse selection dùng:

    model
    tokenizer
    device
    ids

và đọc image embeddings đã có từ ``derived/clip_l/<video_id>.npy``.
Không encode lại sparse images và không reconstruct FAISS vectors.

Dense frame dùng frame_idx thật, không dùng n.

MULTI-WINDOW
------------
TR-R2 hiện hỗ trợ nhiều local temporal windows:

    [
        (97.64, 98.28),
        (1385.69, 1386.19),
    ]

Các dense frame từ tất cả windows được:

    - discover riêng từng window;
    - deduplicate theo frame_idx;
    - sort toàn cục theo pts_time;
    - encode CLIP-L một lần;
    - đưa vào cùng score matrix / DP.

Không biến các window xa nhau thành một span lớn.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

from aic2026.paths import DATA_ROOT
from aic2026.trake_r2_dp import solve_strict_increasing_path
from aic2026.trake_retrieval import (
    _get_trr1_clip_l_runtime,
    _query_variants,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DENSE_FRAMES_ROOT = Path(DATA_ROOT) / "derived" / "frames_dense"
FRAME_MAP_PATH = Path(DATA_ROOT) / "index" / "frame_map.parquet"

DEFAULT_BATCH_SIZE = 16

# CLIP/OpenCLIP preprocessing config.
# Không tạo model mới; chỉ tạo image transform cho model đã load.
_CLIP_PREPROCESS = None
_CLIP_PREPROCESS_MODEL_ID = None


# ---------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------

TimeWindow = tuple[float, float]


# ---------------------------------------------------------------------
# Dense frame record
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DenseFrame:
    frame_idx: int
    pts_time: float
    path: Path


@dataclass(frozen=True)
class SparseKeyframe:
    """Một keyframe sparse và mapping thật dùng cho video-local DP."""

    n: int
    frame_idx: int
    pts_time: float


# ---------------------------------------------------------------------
# Frame map
# ---------------------------------------------------------------------

_FRAME_MAP_CACHE: pd.DataFrame | None = None


def _load_frame_map() -> pd.DataFrame:
    """
    Load frame_map một lần và cache trong process.
    """
    global _FRAME_MAP_CACHE

    if _FRAME_MAP_CACHE is None:
        if not FRAME_MAP_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy frame_map: {FRAME_MAP_PATH}"
            )

        df = pd.read_parquet(FRAME_MAP_PATH)

        required = {
            "video_id",
            "n",
            "pts_time",
            "fps",
            "frame_idx",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"frame_map thiếu cột: {sorted(missing)}"
            )

        _FRAME_MAP_CACHE = df

    return _FRAME_MAP_CACHE


def _fps_for_video(video_id: str) -> float:
    """
    Lấy FPS từ frame_map.
    """
    df = _load_frame_map()

    rows = df[
        df["video_id"].astype(str) == str(video_id)
    ]

    if rows.empty:
        raise ValueError(
            f"Không tìm thấy video_id={video_id!r} trong frame_map."
        )

    fps_values = (
        rows["fps"]
        .dropna()
        .astype(float)
        .unique()
    )

    if len(fps_values) == 0:
        raise ValueError(
            f"Không có fps hợp lệ cho video={video_id!r}."
        )

    fps = float(fps_values[0])

    if fps <= 0:
        raise ValueError(
            f"FPS không hợp lệ cho video={video_id!r}: {fps}"
        )

    return fps


# ---------------------------------------------------------------------
# Dense frame discovery — single window
# ---------------------------------------------------------------------

def discover_dense_frames(
    video_id: str,
    window_start: float,
    window_end: float,
) -> list[DenseFrame]:
    """
    Tìm dense frames của một video trong một local temporal window.

    Filename dense frame là frame_idx thật:

        000120.jpg
        000124.jpg
        ...

    Ưu tiên pts_time từ frame_map.

    Nếu frame_idx không có trong frame_map thì fallback:

        pts_time = frame_idx / fps

    Không scan dense frames của video khác.
    """

    window_start = float(window_start)
    window_end = float(window_end)

    if window_end < window_start:
        raise ValueError(
            f"window không hợp lệ: "
            f"start={window_start}, end={window_end}"
        )

    dense_dir = DENSE_FRAMES_ROOT / str(video_id)

    if not dense_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dense frames cho video={video_id!r}: "
            f"{dense_dir}"
        )

    fps = _fps_for_video(video_id)

    # -------------------------------------------------------------
    # Frame map lookup cho video này.
    # -------------------------------------------------------------

    frame_map = _load_frame_map()

    rows = frame_map[
        frame_map["video_id"].astype(str) == str(video_id)
    ][["frame_idx", "pts_time"]].copy()

    frame_to_pts: dict[int, float] = {}

    for frame_idx, pts_time in zip(
        rows["frame_idx"],
        rows["pts_time"],
    ):
        if pd.isna(frame_idx):
            continue

        frame_idx_int = int(frame_idx)

        if pd.notna(pts_time):
            frame_to_pts[frame_idx_int] = float(pts_time)

    # -------------------------------------------------------------
    # Đọc jpg dense.
    # -------------------------------------------------------------

    candidates: list[DenseFrame] = []

    for path in dense_dir.glob("*.jpg"):
        try:
            frame_idx = int(path.stem)
        except ValueError:
            # Bỏ qua file không mang tên frame_idx.
            continue

        pts_time = frame_to_pts.get(
            frame_idx,
            frame_idx / fps,
        )

        if window_start <= pts_time <= window_end:
            candidates.append(
                DenseFrame(
                    frame_idx=frame_idx,
                    pts_time=float(pts_time),
                    path=path,
                )
            )

    candidates.sort(
        key=lambda x: (x.pts_time, x.frame_idx)
    )

    if not candidates:
        raise ValueError(
            "Không có dense frame trong local window: "
            f"video={video_id!r}, "
            f"window=[{window_start:.6f}, {window_end:.6f}]"
        )

    return candidates


# ---------------------------------------------------------------------
# Dense frame discovery — multiple windows
# ---------------------------------------------------------------------

def discover_dense_frames_multi(
    video_id: str,
    windows: Sequence[TimeWindow],
) -> list[DenseFrame]:
    """
    Discover dense frames trong nhiều temporal windows.

    Ví dụ:

        windows = [
            (97.64, 98.28),
            (1385.69, 1386.19),
        ]

    Mỗi window được discover độc lập.

    Sau đó:
        - deduplicate theo frame_idx;
        - sort toàn bộ frame theo pts_time.

    Nhờ vậy các window cách nhau hàng trăm giây/phút
    không bị biến thành một span liên tục.

    Raises:
        ValueError nếu windows rỗng.
    """

    if not windows:
        raise ValueError(
            "windows không được rỗng."
        )

    all_frames: dict[int, DenseFrame] = {}

    for window_start, window_end in windows:
        frames = discover_dense_frames(
            video_id,
            float(window_start),
            float(window_end),
        )

        for frame in frames:
            all_frames[frame.frame_idx] = frame

    result = sorted(
        all_frames.values(),
        key=lambda frame: (
            frame.pts_time,
            frame.frame_idx,
        ),
    )

    if not result:
        raise ValueError(
            f"Không tìm thấy dense frame nào cho "
            f"video={video_id!r} trong windows={windows!r}"
        )

    return result


# ---------------------------------------------------------------------
# CLIP-L preprocessing
# ---------------------------------------------------------------------

def _get_clip_l_preprocess(model):
    """
    Tạo preprocessing transform cho model CLIP-L đã được load.

    QUAN TRỌNG:
        Không gọi create_model_and_transforms().
        Không load weights.
        Chỉ tạo transform.

    Ưu tiên lấy mean/std từ pretrained config của OpenCLIP.
    """

    global _CLIP_PREPROCESS
    global _CLIP_PREPROCESS_MODEL_ID

    model_id = id(model)

    if (
        _CLIP_PREPROCESS is not None
        and _CLIP_PREPROCESS_MODEL_ID == model_id
    ):
        return _CLIP_PREPROCESS

    import open_clip

    visual = getattr(model, "visual", None)

    if visual is None:
        raise RuntimeError(
            "CLIP-L runtime không có model.visual."
        )

    image_size = getattr(
        visual,
        "image_size",
        224,
    )

    if isinstance(image_size, (tuple, list)):
        if len(image_size) != 2:
            raise ValueError(
                f"image_size không hợp lệ: {image_size}"
            )

    # -------------------------------------------------------------
    # Runtime model hiện tại:
    #
    #   ViT-L-14
    #   laion2b_s32b_b82k
    #
    # Chỉ lấy config preprocessing.
    # Không tạo model mới.
    # -------------------------------------------------------------

    mean = None
    std = None

    try:
        cfg = open_clip.get_pretrained_cfg(
            "ViT-L-14",
            "laion2b_s32b_b82k",
        )

        if cfg:
            mean = cfg.get("mean")
            std = cfg.get("std")

    except Exception:
        # Một số version OpenCLIP có API khác.
        # image_transform sẽ dùng default nếu mean/std=None.
        pass

    preprocess = open_clip.image_transform(
        image_size=image_size,
        is_train=False,
        mean=mean,
        std=std,
    )

    _CLIP_PREPROCESS = preprocess
    _CLIP_PREPROCESS_MODEL_ID = model_id

    return preprocess


# ---------------------------------------------------------------------
# Tensor normalization
# ---------------------------------------------------------------------

def _normalize_embeddings(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    L2 normalize embedding theo hàng.
    """

    return x / x.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)


def _encode_texts_with_runtime(
    model: Any,
    tokenizer: Any,
    device: Any,
    texts: Sequence[str],
) -> torch.Tensor:
    """Encode và L2-normalize text bằng runtime CLIP-L đã load."""

    if not texts:
        raise ValueError("texts không được rỗng")

    tokens = tokenizer(list(texts))

    if hasattr(tokens, "to"):
        tokens = tokens.to(device)
    elif isinstance(tokens, dict):
        tokens = {
            key: value.to(device)
            if hasattr(value, "to")
            else value
            for key, value in tokens.items()
        }

    with torch.inference_mode():
        features = model.encode_text(tokens)

    return _normalize_embeddings(features.float())


def solve_sparse_video_paths(
    event_ids: Sequence[str],
    frames_by_video: Mapping[str, Sequence[SparseKeyframe]],
    score_matrices: Mapping[str, np.ndarray],
    *,
    min_gap: int = 1,
    candidate_order: Sequence[str] | None = None,
    anchor_time_matrices: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Chạy strict-increasing DP trên từng video và chọn path tốt nhất.

    Đây là lõi pure/testable. Mỗi ma trận có shape
    ``(num_events, num_sparse_keyframes)`` và các cột phải cùng thứ tự với
    ``frames_by_video[video_id]``.
    """

    normalized_event_ids = [str(event_id) for event_id in event_ids]

    if not normalized_event_ids:
        raise ValueError("event_ids không được rỗng")

    if min_gap < 1:
        raise ValueError("sparse min_gap phải >= 1")

    if candidate_order is None:
        ordered_video_ids = list(frames_by_video.keys())
    else:
        ordered_video_ids = [str(video_id) for video_id in candidate_order]

    if not ordered_video_ids:
        raise ValueError("Không có sparse video candidate")

    candidate_results: list[dict[str, Any]] = []

    for beam_rank, video_id in enumerate(ordered_video_ids, start=1):
        if video_id not in frames_by_video:
            continue

        if video_id not in score_matrices:
            raise ValueError(
                f"Thiếu sparse score matrix cho video={video_id!r}"
            )

        frames = list(frames_by_video[video_id])
        matrix = np.asarray(score_matrices[video_id], dtype=np.float32)

        expected_shape = (len(normalized_event_ids), len(frames))

        if matrix.shape != expected_shape:
            raise ValueError(
                f"Sparse matrix video={video_id!r} có shape "
                f"{matrix.shape}, cần {expected_shape}"
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                f"Sparse matrix video={video_id!r} chứa NaN/Inf"
            )

        if anchor_time_matrices is None:
            anchor_times = np.broadcast_to(
                np.asarray(
                    [float(frame.pts_time) for frame in frames],
                    dtype=np.float64,
                ),
                expected_shape,
            )
        else:
            if video_id not in anchor_time_matrices:
                raise ValueError(
                    "Thiếu sparse anchor-time matrix cho "
                    f"video={video_id!r}"
                )

            anchor_times = np.asarray(
                anchor_time_matrices[video_id],
                dtype=np.float64,
            )

            if anchor_times.shape != expected_shape:
                raise ValueError(
                    f"Sparse anchor-time matrix video={video_id!r} có "
                    f"shape {anchor_times.shape}, cần {expected_shape}"
                )

        if not np.isfinite(anchor_times).all():
            raise ValueError(
                "Sparse anchor-time matrix chứa NaN/Inf cho "
                f"video={video_id!r}"
            )

        if len(frames) < 1 + (len(normalized_event_ids) - 1) * min_gap:
            continue

        for left, right in zip(frames, frames[1:]):
            left_frame_idx = int(left.frame_idx)
            right_frame_idx = int(right.frame_idx)
            left_pts_time = float(left.pts_time)
            right_pts_time = float(right.pts_time)

            # TRAKE nộp frame_idx thật, vì vậy đây mới là invariant phải
            # tăng nghiêm ngặt. Hai sparse keyframe kề nhau có thể có cùng
            # pts_time do timestamp trong frame map bị lượng tử/làm tròn.
            if left_frame_idx >= right_frame_idx:
                raise ValueError(
                    "Sparse frame_idx không tăng nghiêm ngặt cho "
                    f"video={video_id!r}: "
                    f"{left_frame_idx} -> {right_frame_idx}"
                )

            if left_pts_time > right_pts_time:
                raise ValueError(
                    "Sparse pts_time bị giảm cho "
                    f"video={video_id!r}: "
                    f"frame_idx={left_frame_idx}, pts_time={left_pts_time} "
                    f"-> frame_idx={right_frame_idx}, "
                    f"pts_time={right_pts_time}"
                )

        chosen_positions, total_score = solve_strict_increasing_path(
            matrix.tolist(),
            min_gap=min_gap,
        )

        chosen_frames = [
            frames[position]
            for position in chosen_positions
        ]

        candidate_results.append(
            {
                "video_id": video_id,
                "beam_rank": int(beam_rank),
                "num_sparse_frames": len(frames),
                "chosen_positions": [
                    int(position)
                    for position in chosen_positions
                ],
                "chosen_frame_idx": [
                    int(frame.frame_idx)
                    for frame in chosen_frames
                ],
                "chosen_times": {
                    event_id: float(anchor_times[event_index, position])
                    for event_index, (event_id, position) in enumerate(
                        zip(normalized_event_ids, chosen_positions)
                    )
                },
                "total_score": float(total_score),
                "mean_score": float(total_score) / len(normalized_event_ids),
            }
        )

    if not candidate_results:
        raise ValueError(
            "Không video candidate nào có đủ sparse keyframe cho DP"
        )

    candidate_results.sort(
        key=lambda item: (
            -float(item["mean_score"]),
            int(item["beam_rank"]),
            str(item["video_id"]),
        )
    )

    winner = dict(candidate_results[0])
    winner["candidate_scores"] = [
        {
            "video_id": str(item["video_id"]),
            "beam_rank": int(item["beam_rank"]),
            "num_sparse_frames": int(item["num_sparse_frames"]),
            "total_score": float(item["total_score"]),
            "mean_score": float(item["mean_score"]),
        }
        for item in candidate_results
    ]

    return winner


def select_video_by_sparse_dp(
    event_texts: Mapping[str, str],
    video_ids: Sequence[str],
    *,
    min_gap: int = 1,
    use_query_expansion: bool = True,
    max_query_variants: int = 4,
) -> dict[str, Any]:
    """Rescore toàn bộ sparse CLIP-L keyframe trong candidate videos.

    Text được encode một lần cho cả query. Với mỗi event, điểm của một
    keyframe là cosine lớn nhất qua các query variants. Image embeddings
    được đọc từ ``derived/clip_l/<video_id>.npy``; không encode lại ảnh và
    không dùng GT.
    """

    normalized_event_texts = {
        str(event_id): str(text).strip()
        for event_id, text in event_texts.items()
    }

    if not normalized_event_texts:
        raise ValueError("event_texts không được rỗng")

    if any(not text for text in normalized_event_texts.values()):
        raise ValueError("Mọi sparse event text phải khác rỗng")

    if min_gap < 1:
        raise ValueError("sparse min_gap phải >= 1")

    if max_query_variants <= 0:
        raise ValueError("max_query_variants phải > 0")

    candidate_video_ids: list[str] = []

    for raw_video_id in video_ids:
        video_id = str(raw_video_id)
        if video_id and video_id not in candidate_video_ids:
            candidate_video_ids.append(video_id)

    if not candidate_video_ids:
        raise ValueError("video_ids không được rỗng")

    model, tokenizer, device, _index, ids = _get_trr1_clip_l_runtime()
    model.eval()

    required_columns = {"video_id", "n", "frame_idx", "pts_time"}
    missing_columns = required_columns - set(ids.columns)

    if missing_columns:
        raise ValueError(
            "clip_l_ids thiếu cột: " + ", ".join(sorted(missing_columns))
        )

    event_ids = list(normalized_event_texts.keys())
    flat_variants: list[str] = []
    variant_positions: dict[str, list[int]] = {}

    for event_id in event_ids:
        variants = _query_variants(
            normalized_event_texts[event_id],
            use_query_expansion=use_query_expansion,
            max_query_variants=max_query_variants,
        )

        positions: list[int] = []

        for variant in variants:
            positions.append(len(flat_variants))
            flat_variants.append(str(variant))

        if not positions:
            raise ValueError(
                f"Không tạo được query variant cho event={event_id!r}"
            )

        variant_positions[event_id] = positions

    text_features = _encode_texts_with_runtime(
        model,
        tokenizer,
        device,
        flat_variants,
    ).detach().cpu().numpy().astype(np.float32, copy=False)

    from aic2026.index import clip_l_index

    frames_by_video: dict[str, list[SparseKeyframe]] = {}
    score_matrices: dict[str, np.ndarray] = {}
    anchor_time_matrices: dict[str, np.ndarray] = {}

    for video_id in candidate_video_ids:
        rows = ids[
            ids["video_id"].astype(str) == video_id
        ][["n", "frame_idx", "pts_time"]].copy()

        if rows.empty:
            continue

        rows = rows.sort_values("n").reset_index(drop=True)

        if rows["n"].duplicated().any():
            raise ValueError(
                f"clip_l_ids trùng n cho video={video_id!r}"
            )

        image_features = clip_l_index.doc_dac_trung(
            video_id,
            so_hang_can=len(rows),
        ).astype(np.float32, copy=False)

        norms = np.linalg.norm(image_features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        image_features = image_features / norms

        pts_times = rows["pts_time"].astype(float).to_numpy()
        frame_indices = rows["frame_idx"].astype(int).to_numpy()

        if not np.isfinite(pts_times).all():
            raise ValueError(
                f"clip_l_ids chứa pts_time NaN/Inf cho video={video_id!r}"
            )

        temporal_order = np.lexsort((frame_indices, pts_times))

        rows = rows.iloc[temporal_order].reset_index(drop=True)
        image_features = image_features[temporal_order]
        pts_times = pts_times[temporal_order]
        frame_indices = frame_indices[temporal_order]

        variant_scores = text_features @ image_features.T

        raw_event_scores = np.stack(
            [
                np.max(
                    variant_scores[variant_positions[event_id]],
                    axis=0,
                )
                for event_id in event_ids
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        # frame_idx là đơn vị output của TRAKE. Trong frame map thực tế có
        # trường hợp hai keyframe (hai n/pts_time khác nhau) cùng quy đổi về
        # một frame_idx. Nếu để nguyên hai cột, DP có thể chọn cùng frame_idx
        # cho hai event và vi phạm strict-increasing. Gộp theo frame_idx,
        # nhưng lấy max score RIÊNG cho từng event để không làm mất keyframe
        # có tín hiệu CLIP tốt hơn.
        representative_rows: list[int] = []
        collapsed_score_columns: list[np.ndarray] = []
        collapsed_time_columns: list[np.ndarray] = []

        for frame_idx in np.unique(frame_indices):
            positions = np.flatnonzero(frame_indices == frame_idx)
            representative_rows.append(
                int(positions[np.argmin(pts_times[positions])])
            )
            scores_for_frame = raw_event_scores[:, positions]
            best_positions = np.argmax(scores_for_frame, axis=1)

            collapsed_score_columns.append(
                scores_for_frame[
                    np.arange(len(event_ids)),
                    best_positions,
                ]
            )
            collapsed_time_columns.append(
                pts_times[positions[best_positions]]
            )

        rows = rows.iloc[representative_rows].reset_index(drop=True)
        event_scores = np.stack(
            collapsed_score_columns,
            axis=1,
        ).astype(np.float32, copy=False)

        frames_by_video[video_id] = [
            SparseKeyframe(
                n=int(row.n),
                frame_idx=int(row.frame_idx),
                pts_time=float(row.pts_time),
            )
            for row in rows.itertuples(index=False)
        ]
        score_matrices[video_id] = event_scores
        anchor_time_matrices[video_id] = np.stack(
            collapsed_time_columns,
            axis=1,
        ).astype(np.float64, copy=False)

    return solve_sparse_video_paths(
        event_ids,
        frames_by_video,
        score_matrices,
        min_gap=min_gap,
        candidate_order=candidate_video_ids,
        anchor_time_matrices=anchor_time_matrices,
    )


# ---------------------------------------------------------------------
# Dense scorer
# ---------------------------------------------------------------------

class DenseClipLScorer:
    """
    CLIP-L scorer cho một video + nhiều local temporal windows.

    prepare()
        encode text events + dense images một lần.

    score(event_id, pts_time)
        trả cosine score đã cache.

    score_fn(event_id, pts_time)
        adapter đúng interface của DP.

    Compatibility:
        Có thể truyền một window cũ:

            window_start
            window_end

        hoặc API mới:

            windows=[(start, end), ...]
    """

    def __init__(
        self,
        video_id: str,
        event_texts: Mapping[str, str],
        *,
        windows: Sequence[TimeWindow] | None = None,
        window_start: float | None = None,
        window_end: float | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:

        self.video_id = str(video_id)

        # -------------------------------------------------------------
        # Normalize windows.
        #
        # API mới ưu tiên `windows`.
        #
        # API cũ:
        #     window_start + window_end
        #
        # vẫn được hỗ trợ để không phá code/test cũ.
        # -------------------------------------------------------------

        if windows is not None:
            normalized_windows = [
                (
                    float(start),
                    float(end),
                )
                for start, end in windows
            ]

        elif (
            window_start is not None
            and window_end is not None
        ):
            normalized_windows = [
                (
                    float(window_start),
                    float(window_end),
                )
            ]

        else:
            raise ValueError(
                "Phải cung cấp `windows` hoặc "
                "`window_start` + `window_end`."
            )

        if not normalized_windows:
            raise ValueError(
                "windows không được rỗng."
            )

        for start, end in normalized_windows:
            if end < start:
                raise ValueError(
                    "Window không hợp lệ: "
                    f"start={start}, end={end}"
                )

        self.windows: list[TimeWindow] = normalized_windows

        # -------------------------------------------------------------
        # Compatibility attributes cho code/debug cũ.
        #
        # Chỉ có giá trị nếu scorer có đúng một window.
        # -------------------------------------------------------------

        if len(self.windows) == 1:
            self.window_start = self.windows[0][0]
            self.window_end = self.windows[0][1]
        else:
            self.window_start = min(
                start
                for start, _end in self.windows
            )
            self.window_end = max(
                end
                for _start, end in self.windows
            )

        self.event_texts = {
            str(k): str(v)
            for k, v in event_texts.items()
        }

        self.batch_size = int(batch_size)

        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size phải > 0, nhận {batch_size}"
            )

        self.event_ids: list[str] = list(
            self.event_texts.keys()
        )

        if not self.event_ids:
            raise ValueError(
                "event_texts không được rỗng."
            )

        self.frames: list[DenseFrame] = []

        # score_matrix[event_position, frame_position]
        self.score_matrix: np.ndarray | None = None

        self.event_pos: dict[str, int] = {}

        self._times = np.empty(
            0,
            dtype=np.float64,
        )

        self._prepared = False

        # Runtime CLIP-L.
        self.model = None
        self.tokenizer = None
        self.device = None

    # -----------------------------------------------------------------
    # Runtime
    # -----------------------------------------------------------------

    def _load_runtime(self) -> None:
        """
        Dùng CHÍNH runtime CLIP-L của TR-R1.
        """

        (
            model,
            tokenizer,
            device,
            _index,
            _ids,
        ) = _get_trr1_clip_l_runtime()

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        self.model.eval()

    # -----------------------------------------------------------------
    # Text encoding
    # -----------------------------------------------------------------

    def _encode_texts(
        self,
        texts: Sequence[str],
    ) -> torch.Tensor:
        """
        Encode toàn bộ event text bằng CLIP-L runtime của TR-R1.
        """

        if self.model is None:
            raise RuntimeError(
                "CLIP-L runtime chưa được load."
            )

        if self.tokenizer is None:
            raise RuntimeError(
                "CLIP-L tokenizer chưa được load."
            )

        return _encode_texts_with_runtime(
            self.model,
            self.tokenizer,
            self.device,
            texts,
        )

    # -----------------------------------------------------------------
    # Image encoding
    # -----------------------------------------------------------------

    def _encode_images(
        self,
        frames: Sequence[DenseFrame],
    ) -> torch.Tensor:
        """
        Encode dense frames theo batch.

        Không encode từng frame trong score().
        Toàn bộ image embedding được cache trước DP.
        """

        if self.model is None:
            raise RuntimeError(
                "CLIP-L runtime chưa được load."
            )

        preprocess = _get_clip_l_preprocess(
            self.model
        )

        all_features: list[torch.Tensor] = []

        for start in range(
            0,
            len(frames),
            self.batch_size,
        ):
            batch_frames = frames[
                start : start + self.batch_size
            ]

            tensors: list[torch.Tensor] = []

            for frame in batch_frames:
                with Image.open(
                    frame.path
                ) as image:

                    image = image.convert(
                        "RGB"
                    )

                    tensor = preprocess(
                        image
                    )

                tensors.append(tensor)

            batch = torch.stack(
                tensors,
                dim=0,
            ).to(self.device)

            with torch.inference_mode():
                features = self.model.encode_image(
                    batch
                )

            features = features.float()

            features = _normalize_embeddings(
                features
            )

            all_features.append(
                features.cpu()
            )

        return torch.cat(
            all_features,
            dim=0,
        )

    # -----------------------------------------------------------------
    # Prepare
    # -----------------------------------------------------------------

    def prepare(self) -> "DenseClipLScorer":
        """
        Chuẩn bị scorer:

            dense frame discovery
            +
            text embedding
            +
            image embedding
            +
            score matrix
        """

        if self._prepared:
            return self

        # -------------------------------------------------------------
        # 1. CLIP-L runtime
        # -------------------------------------------------------------

        self._load_runtime()

        # -------------------------------------------------------------
        # 2. Dense frames từ TẤT CẢ windows.
        # -------------------------------------------------------------

        self.frames = discover_dense_frames_multi(
            self.video_id,
            self.windows,
        )

        self._times = np.asarray(
            [
                frame.pts_time
                for frame in self.frames
            ],
            dtype=np.float64,
        )

        # -------------------------------------------------------------
        # 3. Event positions
        # -------------------------------------------------------------

        self.event_pos = {
            event_id: i
            for i, event_id in enumerate(
                self.event_ids
            )
        }

        # -------------------------------------------------------------
        # 4. Text embedding
        # -------------------------------------------------------------

        text_features = self._encode_texts(
            [
                self.event_texts[event_id]
                for event_id in self.event_ids
            ]
        )

        # -------------------------------------------------------------
        # 5. Image embedding
        # -------------------------------------------------------------

        image_features = self._encode_images(
            self.frames
        )

        # -------------------------------------------------------------
        # 6. Cosine score matrix
        #
        # Vì cả hai phía đã normalize:
        #
        #     cosine = text @ image.T
        # -------------------------------------------------------------

        score_matrix = (
            text_features
            @ image_features.T
        )

        self.score_matrix = (
            score_matrix.numpy()
        )

        self._prepared = True

        return self

    # -----------------------------------------------------------------
    # Time -> nearest dense frame
    # -----------------------------------------------------------------

    def frame_idx_at_time(
        self,
        pts_time: float,
    ) -> int:
        """
        Trả frame_idx thật gần nhất với pts_time.
        """

        self._ensure_prepared()

        if len(self.frames) == 0:
            raise RuntimeError(
                "Không có dense frame."
            )

        t = float(pts_time)

        pos = int(
            np.searchsorted(
                self._times,
                t,
                side="left",
            )
        )

        if pos <= 0:
            return self.frames[0].frame_idx

        if pos >= len(self.frames):
            return self.frames[-1].frame_idx

        left = pos - 1
        right = pos

        if abs(
            self._times[left] - t
        ) <= abs(
            self._times[right] - t
        ):
            return self.frames[left].frame_idx

        return self.frames[right].frame_idx

    # -----------------------------------------------------------------
    # Time -> frame position
    # -----------------------------------------------------------------

    def _frame_position_at_time(
        self,
        pts_time: float,
    ) -> int:
        """
        Trả vị trí dense frame gần nhất.
        """

        self._ensure_prepared()

        t = float(pts_time)

        pos = int(
            np.searchsorted(
                self._times,
                t,
                side="left",
            )
        )

        if pos <= 0:
            return 0

        if pos >= len(self._times):
            return len(self._times) - 1

        left = pos - 1
        right = pos

        if abs(
            self._times[left] - t
        ) <= abs(
            self._times[right] - t
        ):
            return left

        return right

    # -----------------------------------------------------------------
    # Score
    # -----------------------------------------------------------------

    def score(
        self,
        event_id: str,
        pts_time: float,
    ) -> float:
        """
        CLIP-L score cho:

            event_id
            pts_time

        pts_time được map về dense frame gần nhất.
        """

        self._ensure_prepared()

        event_id = str(event_id)

        if event_id not in self.event_pos:
            raise KeyError(
                f"Event không tồn tại: {event_id!r}"
            )

        event_pos = self.event_pos[
            event_id
        ]

        frame_pos = (
            self._frame_position_at_time(
                pts_time
            )
        )

        assert self.score_matrix is not None

        return float(
            self.score_matrix[
                event_pos,
                frame_pos,
            ]
        )

    # -----------------------------------------------------------------
    # score_fn adapter
    # -----------------------------------------------------------------

    def score_fn(
        self,
        event_id: str,
        pts_time: float,
    ) -> float:
        """
        Adapter đúng interface:

            Callable[[str, float], float]
        """

        return self.score(
            event_id,
            pts_time,
        )

    # -----------------------------------------------------------------
    # Debug
    # -----------------------------------------------------------------

    def top_frames_for_event(
        self,
        event_id: str,
        k: int = 5,
    ) -> list[dict]:
        """
        Debug: lấy top-k dense frame của một event.

        Dùng để kiểm tra:

            "TRAKE đúng video nhưng sai mốc"

        theo handbook.
        """

        self._ensure_prepared()

        if k <= 0:
            return []

        if event_id not in self.event_pos:
            raise KeyError(
                f"Event không tồn tại: {event_id!r}"
            )

        assert self.score_matrix is not None

        event_pos = self.event_pos[
            event_id
        ]

        scores = self.score_matrix[
            event_pos
        ]

        k = min(
            k,
            len(scores),
        )

        # Stable deterministic ordering:
        # score giảm,
        # frame_idx tăng khi hòa.
        order = sorted(
            range(len(scores)),
            key=lambda i: (
                -float(scores[i]),
                self.frames[i].frame_idx,
            ),
        )[:k]

        return [
            {
                "frame_idx": self.frames[i].frame_idx,
                "pts_time": self.frames[i].pts_time,
                "score": float(scores[i]),
            }
            for i in order
        ]

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _ensure_prepared(self) -> None:
        if not self._prepared:
            self.prepare()


# ---------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------

def build_dense_score_fn(
    video_id: str,
    event_texts: Mapping[str, str],
    *,
    windows: Sequence[TimeWindow] | None = None,
    window_start: float | None = None,
    window_end: float | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[
    DenseClipLScorer,
    Callable[[str, float], float],
]:
    """
    Factory production cho TR-R2.

    API mới:

        scorer, score_fn = build_dense_score_fn(
            video_id="L25_V063",
            windows=[
                (97.640, 98.280),
                (1385.690, 1386.190),
            ],
            event_texts={
                "E1": "...",
                "E2": "...",
            },
        )

    API cũ vẫn được hỗ trợ:

        scorer, score_fn = build_dense_score_fn(
            video_id="L26_V315",
            window_start=10.0,
            window_end=20.0,
            event_texts={
                "E1": "...",
            },
        )

    Sau đó:

        score_fn("E1", 97.92)

    Scorer sẽ prepare ngay tại đây.
    """

    scorer = DenseClipLScorer(
        video_id=video_id,
        event_texts=event_texts,
        windows=windows,
        window_start=window_start,
        window_end=window_end,
        batch_size=batch_size,
    )

    # Encode ngay tại đây để pipeline không vô tình
    # gọi score() nhiều lần rồi encode lại.
    scorer.prepare()

    return (
        scorer,
        scorer.score_fn,
    )
