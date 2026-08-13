"""
Kiểm chứng ClipEncoder của Task 3.

So sánh vector do ClipEncoder tạo từ keyframe thật
với vector CLIP BTC cung cấp tương ứng.

Điều kiện đạt:
- ít nhất 30 ảnh
- dimension = 512
- cosine similarity trung bình >= 0.99
"""

import random

import numpy as np

from aic2026.index.encode.clip_encoder import ClipEncoder
from aic2026.frame_map import FrameMap, available_video_ids
from aic2026.paths import clip_features_file, keyframe_image


NUM_SAMPLES = 30
MIN_MEAN_SIMILARITY = 0.99
RANDOM_SEED = 2026


def find_keyframe_image(video_id: str, n: int):
    """
    Tìm ảnh keyframe tương ứng với video_id và n.

    paths.py hiện đang tạo tên 4 chữ số, ví dụ:
        0293.jpg

    Trong dữ liệu keyframe thực tế có thể dùng 3 chữ số:
        293.jpg

    Vì vậy thử cả hai cách, nhưng không sửa paths.py.
    """

    image_path = keyframe_image(video_id, n)

    if image_path.exists():
        return image_path

    image_path = image_path.with_name(
        f"{n:03d}.jpg"
    )

    if image_path.exists():
        return image_path

    return None


def main() -> None:
    print("[1] Loading ClipEncoder...", flush=True)

    encoder = ClipEncoder()

    print(
        f"[1] ClipEncoder loaded: "
        f"{encoder.model_name} + {encoder.pretrained}",
        flush=True,
    )

    if encoder.dimension != 512:
        raise RuntimeError(
            f"Encoder dimension = {encoder.dimension}, "
            "mong đợi 512."
        )

    print("[2] Finding videos...", flush=True)

    video_ids = available_video_ids()

    print(
        f"[2] Found {len(video_ids)} videos.",
        flush=True,
    )

    print(
        "[3] Collecting candidate frames "
        "with existing images...",
        flush=True,
    )

    candidates: list[tuple[str, int]] = []

    for video_id in video_ids:
        frame_map = FrameMap.load(video_id)

        for row in frame_map.rows:
            image_path = find_keyframe_image(
                video_id,
                row.n,
            )

            if image_path is not None:
                candidates.append(
                    (video_id, row.n)
                )

    if len(candidates) < NUM_SAMPLES:
        raise RuntimeError(
            f"Chỉ tìm được {len(candidates)} ảnh "
            f"thực tế, cần ít nhất {NUM_SAMPLES}."
        )

    rng = random.Random(RANDOM_SEED)

    samples = rng.sample(
        candidates,
        NUM_SAMPLES,
    )

    print(
        f"[3] Selected {len(samples)} samples "
        f"(seed={RANDOM_SEED}).",
        flush=True,
    )

    similarities: list[float] = []

    frame_maps: dict[str, FrameMap] = {}
    btc_features_cache: dict[str, np.ndarray] = {}

    for i, (video_id, n) in enumerate(
        samples,
        start=1,
    ):
        print(
            f"[4] Processing {i}/{len(samples)}: "
            f"{video_id} n={n}",
            flush=True,
        )

        if video_id not in frame_maps:
            frame_maps[video_id] = FrameMap.load(
                video_id
            )

        frame_map = frame_maps[video_id]

        row = frame_map.row_of(n)

        image_path = find_keyframe_image(
            video_id,
            n,
        )

        if image_path is None:
            raise FileNotFoundError(
                f"Không tìm thấy ảnh keyframe: "
                f"{video_id}, n={n}"
            )

        feature_path = clip_features_file(
            video_id
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy vector BTC: "
                f"{feature_path}"
            )

        if video_id not in btc_features_cache:
            btc_features_cache[video_id] = np.load(
                feature_path
            )

        btc_features = btc_features_cache[
            video_id
        ]

        if btc_features.ndim != 2:
            raise ValueError(
                f"{video_id}.npy có shape "
                f"{btc_features.shape}, "
                "không phải ma trận 2 chiều."
            )

        if btc_features.shape[1] != 512:
            raise ValueError(
                f"{video_id}.npy có shape "
                f"{btc_features.shape}, "
                "dimension phải là 512."
            )

        if n < 1 or n > len(btc_features):
            raise ValueError(
                f"{video_id}: n={n} không hợp lệ, "
                f"BTC có {len(btc_features)} vector."
            )

        # Vector BTC tương ứng với keyframe n.
        # n bắt đầu từ 1 nên index numpy là n - 1.
        btc = btc_features[n - 1].astype(
            np.float32,
            copy=True,
        )

        btc_norm = np.linalg.norm(btc)

        if btc_norm == 0:
            raise ValueError(
                f"Vector BTC bằng 0: "
                f"{video_id}, n={n}"
            )

        btc /= btc_norm

        # Encode ảnh bằng chính CLIP model.
        encoded = encoder.encode_image_path(
            image_path
        )

        if encoded.shape != (512,):
            raise ValueError(
                f"Encoder trả shape {encoded.shape}, "
                "mong đợi (512,)."
            )

        encoded_norm = np.linalg.norm(
            encoded
        )

        if not np.isclose(
            encoded_norm,
            1.0,
            atol=1e-5,
        ):
            raise ValueError(
                f"Encoder output chưa normalize: "
                f"norm={encoded_norm}"
            )

        # Vì cả hai vector đều L2-normalized,
        # inner product chính là cosine similarity.
        similarity = float(
            np.dot(encoded, btc)
        )

        similarities.append(similarity)

        print(
            f"    {video_id} "
            f"n={n:04d} "
            f"frame_idx={row.frame_idx} "
            f"similarity={similarity:.6f}",
            flush=True,
        )

    mean_similarity = float(
        np.mean(similarities)
    )

    min_similarity = float(
        np.min(similarities)
    )

    max_similarity = float(
        np.max(similarities)
    )

    print()
    print("=" * 60)
    print(
        "TASK 3 - CLIP SAME-SYSTEM VERIFICATION"
    )
    print("=" * 60)
    print(
        f"Samples         = {len(similarities)}"
    )
    print(
        f"Model           = {encoder.model_name}"
    )
    print(
        f"Pretrained      = {encoder.pretrained}"
    )
    print(
        f"Dimension       = {encoder.dimension}"
    )
    print(
        f"Random seed     = {RANDOM_SEED}"
    )
    print(
        f"Mean similarity = {mean_similarity:.6f}"
    )
    print(
        f"Min similarity  = {min_similarity:.6f}"
    )
    print(
        f"Max similarity  = {max_similarity:.6f}"
    )
    print(
        f"Threshold       = {MIN_MEAN_SIMILARITY:.2f}"
    )

    if mean_similarity < MIN_MEAN_SIMILARITY:
        raise RuntimeError(
            "FAIL: mean cosine similarity < 0.99. "
            "Có khả năng model/pretrained không khớp "
            "vector CLIP BTC."
        )

    print(
        "PASS: ClipEncoder khớp vector CLIP BTC."
    )


if __name__ == "__main__":
    main()