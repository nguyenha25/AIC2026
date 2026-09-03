"""
QA-R2 — Multi-Source Score Normalization
=========================================

Mục đích
--------
Chuẩn hóa điểm CLIP / OCR / ASR / Caption về cùng thang [0, 1]
trước khi thực hiện multi-source fusion.

QA-R2 yêu cầu:
    - So sánh Rank Normalization với Min-Max và Robust Scaling.
    - Kiểm tra phân phối sau normalization.
    - Kiểm tra source dominance.
    - Phát hiện trường hợp một source áp đảo chỉ vì khác thang điểm raw.

Các phương pháp:
    1. Min-Max
    2. Robust (Median + IQR + Sigmoid)
    3. Rank / Percentile

Candidate pool
--------------
Normalization được thực hiện độc lập trên candidate set của từng source.

Nếu các source không dùng cùng candidate pool, một source sparse có thể
cho percentile cao hơn tương đối. Đây là đặc tính của normalization,
không phải lỗi số học.

Candidate-pool alignment hoặc sparse-source penalty thuộc tầng retrieval /
fusion policy và không được âm thầm nhúng vào primitive normalization.

N = 1 contract
--------------
Một source chỉ có đúng một candidate không có relative ordering nội bộ.

Để tránh:
    - artificial advantage = 1.0
    - hoặc penalize = 0.0

cả ba normalization đều trả 0.5 cho N = 1.

Nếu hệ thống muốn xem sparse-source Top-1 là evidence mạnh,
đó phải là policy ở fusion/retrieval layer, KHÔNG phải primitive
normalization.

Dominance metric
----------------
Không sử dụng winner-count.

Với mỗi candidate thuộc Top-K fused:

    contribution(source, item)
        = normalized_score(source, item) * source_weight

Sau đó:

    source_share
        = tổng contribution của source trên Top-K
          -----------------------------------
          tổng contribution của mọi source

Acceptance mặc định:

    max_single_source_dominance <= 0.70


Min-Max compression diagnostic
------------------------------
Diagnostic cố gắng đo compression của các candidate có score > min.

Lý do:
    - Nếu nhiều raw score bằng 0 hoặc bằng minimum tự nhiên,
      chúng không phải bằng chứng của outlier compression.
    - Các candidate ở sàn min luôn normalize thành 0 theo Min-Max.

Do đó metric:

    compressed_fraction
        = fraction của NON-MIN candidates có normalized score <= threshold

Nếu tất cả candidate đều bằng min:
    compressed_fraction = 0.0

Metric này chỉ là diagnostic, không phải acceptance criterion độc lập.
"""


from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

EPS = 1e-8

ROBUST_CLIP = 6.0

DEFAULT_TOP_K = 10
DEFAULT_DOMINANCE_THRESHOLD = 0.70

MIN_MAX_COMPRESSION_THRESHOLD = 0.01


# ============================================================
# DATA CONTRACT
# ============================================================

class NormMethod(str, Enum):
    MIN_MAX = "min_max"
    ROBUST = "robust"
    RANK = "rank"


@dataclass(slots=True, frozen=True)
class SourceScores:
    """
    Raw scores của một query trên nhiều source.

    Candidate set giữa các source có thể sparse / khác nhau.
    """

    query_id: str
    scores: Mapping[str, Mapping[str, float]]


@dataclass(slots=True)
class NormalizationResult:
    """
    Kết quả normalization + weighted fusion của một query.
    """

    query_id: str

    normalized_scores: Dict[
        str,
        Dict[str, float],
    ]

    fused_scores: Dict[str, float]


# ============================================================
# NUMERIC HELPERS
# ============================================================

def _to_float_array(scores: Any) -> np.ndarray:
    """
    Convert input thành 1-D float64 NumPy array.

    Contract:
        - Input phải convert được sang float64.
        - Phải là mảng 1 chiều.
        - NaN / Inf bị reject rõ ràng.

    Không cố silent-cast dữ liệu rác.
    Nếu input chứa string không convert được, ValueError được raise
    với message rõ ràng.
    """

    try:
        arr = np.asarray(
            scores,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "scores phải chứa các giá trị số có thể "
            "convert sang float64"
        ) from exc

    if arr.ndim != 1:
        raise ValueError(
            f"scores phải là mảng 1 chiều, "
            f"nhận shape={arr.shape}"
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "scores chứa NaN hoặc Inf"
        )

    return arr


def _clip01(values: np.ndarray) -> np.ndarray:
    """
    Defensive clipping về [0, 1].
    """

    return np.clip(
        values,
        0.0,
        1.0,
    )


# ============================================================
# NORMALIZATION — MIN-MAX
# ============================================================

def min_max_scale(
    scores: Any,
    eps: float = EPS,
) -> np.ndarray:
    """
    Min-Max Scaling:

        norm = (x - min) / (max - min)

    Output:
        [0, 1]

    Constant input:
        0.5 cho toàn bộ.

    N = 1:
        0.5.

    N = 1 không có relative ordering nội bộ.
    Sparse-source policy phải xử lý ở tầng retrieval/fusion.
    """

    arr = _to_float_array(scores)

    n = arr.size

    if n == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    if n == 1:
        return np.array(
            [0.5],
            dtype=np.float64,
        )

    s_min = float(np.min(arr))
    s_max = float(np.max(arr))

    diff = s_max - s_min

    if diff <= eps:
        return np.full(
            n,
            0.5,
            dtype=np.float64,
        )

    normalized = (
        arr - s_min
    ) / diff

    return _clip01(normalized)


# ============================================================
# NORMALIZATION — ROBUST
# ============================================================

def robust_scale(
    scores: Any,
    eps: float = EPS,
    clip: float = ROBUST_CLIP,
) -> np.ndarray:
    """
    Robust Scaling + Sigmoid Bounding.

    Bước 1:

        z = (x - median) / IQR

    Bước 2:

        z = clip(z, -clip, clip)

    Bước 3:

        norm = sigmoid(z)

    Nếu IQR ~= 0:
        fallback sang mean/std.

    Nếu cả IQR và std đều ~= 0:
        0.5 cho toàn bộ.

    N = 1:
        0.5.
    """

    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError(
            "clip phải là finite number > 0"
        )

    arr = _to_float_array(scores)

    n = arr.size

    if n == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    if n == 1:
        return np.array(
            [0.5],
            dtype=np.float64,
        )

    median = float(
        np.median(arr)
    )

    q25, q75 = np.percentile(
        arr,
        [25.0, 75.0],
    )

    iqr = float(
        q75 - q25
    )

    if iqr > eps:
        z = (
            arr - median
        ) / iqr

    else:
        mean = float(
            np.mean(arr)
        )

        std = float(
            np.std(arr)
        )

        if std <= eps:
            return np.full(
                n,
                0.5,
                dtype=np.float64,
            )

        z = (
            arr - mean
        ) / std

    z = np.clip(
        z,
        -clip,
        clip,
    )

    normalized = 1.0 / (
        1.0 + np.exp(-z)
    )

    return _clip01(
        normalized
    )


# ============================================================
# NORMALIZATION — RANK / PERCENTILE
# ============================================================

def rank_normalize(
    scores: Any,
) -> np.ndarray:
    """
    Rank / Percentile Normalization.

    Score thấp nhất  -> 0.0
    Score cao nhất   -> 1.0

    Tie:
        average rank.

    Output luôn được map ngược về ORIGINAL INPUT ORDER.

    N = 1:
        0.5.

    Constant array:
        0.5 cho toàn bộ.
    """

    arr = _to_float_array(scores)

    n = arr.size

    if n == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    if n == 1:
        return np.array(
            [0.5],
            dtype=np.float64,
        )

    if np.all(
        arr == arr[0]
    ):
        return np.full(
            n,
            0.5,
            dtype=np.float64,
        )

    order = np.argsort(
        arr,
        kind="mergesort",
    )

    sorted_values = arr[
        order
    ]

    ranks_sorted = np.arange(
        n,
        dtype=np.float64,
    )

    start = 0

    while start < n:

        end = start + 1

        while (
            end < n
            and sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        if end - start > 1:

            average_rank = (
                float(start)
                + float(end - 1)
            ) / 2.0

            ranks_sorted[
                start:end
            ] = average_rank

        start = end

    ranks = np.empty(
        n,
        dtype=np.float64,
    )

    ranks[order] = ranks_sorted

    denom = max(
        float(n - 1),
        EPS,
    )

    normalized = (
        ranks / denom
    )

    return _clip01(
        normalized
    )


# ============================================================
# NORMALIZATION DISPATCH
# ============================================================

def normalize_scores(
    scores: Any,
    method: NormMethod,
) -> np.ndarray:
    """
    Dispatch tới normalization method.
    """

    if method == NormMethod.MIN_MAX:
        return min_max_scale(scores)

    if method == NormMethod.ROBUST:
        return robust_scale(scores)

    if method == NormMethod.RANK:
        return rank_normalize(scores)

    raise ValueError(
        f"Phương pháp normalization "
        f"không hợp lệ: {method}"
    )


# ============================================================
# MULTI-SOURCE NORMALIZER
# ============================================================

class MultiSourceScoreNormalizer:
    """
    Normalize nhiều source rồi weighted-fuse.
    """

    def __init__(
        self,
        method: NormMethod = NormMethod.RANK,
        weights: Optional[
            Mapping[str, float]
        ] = None,
    ) -> None:

        self.method = method

        default_weights = {
            "clip_b": 1.0,
            "clip_l": 0.8,
            "ocr": 0.5,
            "asr": 0.3,
            "caption": 0.6,
        }

        source_weights = (
            default_weights
            if weights is None
            else dict(weights)
        )

        self.weights: Dict[
            str,
            float,
        ] = {}

        for (
            source_name,
            weight,
        ) in source_weights.items():

            weight = float(weight)

            if not math.isfinite(weight):
                raise ValueError(
                    f"Weight không hợp lệ "
                    f"cho source "
                    f"'{source_name}': "
                    f"{weight}"
                )

            if weight < 0.0:
                raise ValueError(
                    f"Weight không được âm "
                    f"cho source "
                    f"'{source_name}'"
                )

            self.weights[
                source_name
            ] = weight

    def normalize_source_dict(
        self,
        source_raw_scores: Mapping[
            str,
            Mapping[str, float],
        ],
    ) -> Tuple[
        Dict[str, Dict[str, float]],
        Dict[str, float],
    ]:
        """
        Normalize tất cả source của một query.

        Empty source vẫn được giữ để bảo toàn schema.

        Unknown source:
            weight = 1.0.
        """

        normalized_scores: Dict[
            str,
            Dict[str, float],
        ] = {}

        fused_scores: Dict[
            str,
            float,
        ] = {}

        for (
            source_name,
            raw_map,
        ) in source_raw_scores.items():

            if not raw_map:
                normalized_scores[
                    source_name
                ] = {}

                continue

            item_ids = list(
                raw_map.keys()
            )

            try:
                raw_values = np.fromiter(
                    (
                        float(
                            raw_map[item_id]
                        )
                        for item_id in item_ids
                    ),
                    dtype=np.float64,
                    count=len(item_ids),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Source '{source_name}' "
                    "chứa raw score không hợp lệ"
                ) from exc

            norm_values = normalize_scores(
                raw_values,
                self.method,
            )

            norm_map = {
                item_id: float(norm_value)
                for (
                    item_id,
                    norm_value,
                ) in zip(
                    item_ids,
                    norm_values,
                )
            }

            normalized_scores[
                source_name
            ] = norm_map

            weight = self.weights.get(
                source_name,
                1.0,
            )

            if weight == 0.0:
                continue

            for (
                item_id,
                norm_score,
            ) in norm_map.items():

                contribution = (
                    norm_score
                    * weight
                )

                if not math.isfinite(
                    contribution
                ):
                    continue

                fused_scores[
                    item_id
                ] = (
                    fused_scores.get(
                        item_id,
                        0.0,
                    )
                    + contribution
                )

        return (
            normalized_scores,
            fused_scores,
        )

    def normalize(
        self,
        source_scores: SourceScores,
    ) -> NormalizationResult:
        """
        Normalize một SourceScores object.
        """

        (
            normalized_scores,
            fused_scores,
        ) = self.normalize_source_dict(
            source_scores.scores
        )

        return NormalizationResult(
            query_id=source_scores.query_id,
            normalized_scores=normalized_scores,
            fused_scores=fused_scores,
        )


# ============================================================
# DISTRIBUTION ANALYSIS
# ============================================================

def summarize_distribution(
    scores: Any,
) -> Dict[str, float]:
    """
    Tóm tắt phân phối.
    """

    arr = _to_float_array(scores)

    if arr.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
        }

    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def compare_normalization_distribution(
    scores: Any,
) -> Dict[
    str,
    Dict[str, float],
]:
    """
    So sánh distribution của cả 3 phương pháp.
    """

    arr = _to_float_array(scores)

    results: Dict[
        str,
        Dict[str, float],
    ] = {}

    for method in NormMethod:

        normalized = normalize_scores(
            arr,
            method,
        )

        results[
            method.value
        ] = summarize_distribution(
            normalized
        )

    return results


# ============================================================
# MIN-MAX COMPRESSION DIAGNOSTIC
# ============================================================

def analyze_min_max_compression(
    scores: Any,
    threshold: float = MIN_MAX_COMPRESSION_THRESHOLD,
) -> Dict[str, float]:
    """
    Diagnostic cho hiện tượng Min-Max signal compression.

    Raw values bằng minimum luôn normalize thành 0.
    Điều này không tự động chứng minh outlier compression.

    Vì vậy diagnostic loại các candidate có raw score == min
    khỏi denominator.

    Metric:

        compressed_fraction
            = fraction NON-MIN candidates có
              normalized score <= threshold

    Ví dụ:

        [0.1, 0.2, 0.3, 0.4, 0.5, 1000.0]

    min = 0.1

    Non-min candidates:
        0.2, 0.3, 0.4, 0.5, 1000

    Sau Min-Max:
        0.2, 0.3, 0.4, 0.5 đều < 0.01

    compressed_fraction = 4 / 5 = 0.80.

    Nếu tất cả raw scores đều bằng min:
        compressed_fraction = 0.0.

    Đây là diagnostic, không phải acceptance criterion độc lập.
    """

    if not (
        0.0 <= threshold <= 1.0
    ):
        raise ValueError(
            "threshold phải nằm trong [0, 1]"
        )

    arr = _to_float_array(scores)

    if arr.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "compressed_fraction": 0.0,
            "compression_threshold": float(
                threshold
            ),
            "excluded_min_fraction": 0.0,
            "evaluated_fraction": 0.0,
        }

    raw_min = float(np.min(arr))

    normalized = min_max_scale(arr)

    non_min_mask = arr > (
        raw_min + EPS
    )

    excluded_min_fraction = float(
        np.mean(~non_min_mask)
    )

    evaluated_count = int(
        np.count_nonzero(
            non_min_mask
        )
    )

    if evaluated_count == 0:
        compressed_fraction = 0.0
        evaluated_fraction = 0.0

    else:
        compressed_fraction = float(
            np.mean(
                normalized[
                    non_min_mask
                ] <= threshold
            )
        )

        evaluated_fraction = (
            float(evaluated_count)
            / float(arr.size)
        )

    return {
        "min": float(
            np.min(normalized)
        ),
        "max": float(
            np.max(normalized)
        ),
        "mean": float(
            np.mean(normalized)
        ),
        "compressed_fraction": (
            compressed_fraction
        ),
        "compression_threshold": float(
            threshold
        ),
        "excluded_min_fraction": (
            excluded_min_fraction
        ),
        "evaluated_fraction": (
            evaluated_fraction
        ),
    }


# ============================================================
# DOMINANCE ANALYSIS
# ============================================================

def analyze_source_dominance(
    all_raw_data: List[
        Mapping[
            str,
            Mapping[str, float],
        ]
    ],
    weights: Mapping[str, float],
    top_k: int = DEFAULT_TOP_K,
    dominance_threshold: float = (
        DEFAULT_DOMINANCE_THRESHOLD
    ),
) -> Dict[str, Any]:
    """
    Phân tích contribution share của source trên Top-K.

    KHÔNG dùng winner-count.

    Với mỗi query:

        1. Normalize từng source.
        2. Weighted fusion.
        3. Lấy Top-K fused candidates.
        4. Tính contribution của từng source.
        5. Cộng contribution theo source.
        6. Chuẩn hóa thành contribution share.

    Acceptance:

        max_single_source_dominance <= 0.70.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k phải > 0"
        )

    if not (
        0.0
        < dominance_threshold
        <= 1.0
    ):
        raise ValueError(
            "dominance_threshold "
            "phải nằm trong (0, 1]"
        )

    clean_weights: Dict[
        str,
        float,
    ] = {}

    for (
        source_name,
        weight,
    ) in weights.items():

        weight = float(weight)

        if not math.isfinite(weight):
            raise ValueError(
                f"Weight không hợp lệ "
                f"cho source "
                f"'{source_name}'"
            )

        if weight < 0.0:
            raise ValueError(
                f"Weight không được âm "
                f"cho source "
                f"'{source_name}'"
            )

        clean_weights[
            source_name
        ] = weight

    results: Dict[
        str,
        Any,
    ] = {}

    for method in NormMethod:

        normalizer = (
            MultiSourceScoreNormalizer(
                method=method,
                weights=clean_weights,
            )
        )

        source_total_contributions: Dict[
            str,
            float,
        ] = {}

        total_counted_candidates = 0
        skipped_zero_contribution = 0

        for source_raw in all_raw_data:

            if not source_raw:
                continue

            (
                normalized_scores,
                fused_scores,
            ) = normalizer.normalize_source_dict(
                source_raw
            )

            if not fused_scores:
                continue

            top_items = sorted(
                fused_scores.items(),
                key=lambda pair: (
                    -pair[1],
                    pair[0],
                ),
            )[:top_k]

            if not top_items:
                continue

            for (
                item_id,
                _,
            ) in top_items:

                candidate_total = 0.0

                for (
                    source_name,
                    norm_map,
                ) in normalized_scores.items():

                    norm_value = norm_map.get(
                        item_id
                    )

                    if norm_value is None:
                        continue

                    weight = clean_weights.get(
                        source_name,
                        1.0,
                    )

                    contribution = (
                        float(norm_value)
                        * weight
                    )

                    if not math.isfinite(
                        contribution
                    ):
                        continue

                    if contribution <= 0.0:
                        continue

                    source_total_contributions[
                        source_name
                    ] = (
                        source_total_contributions.get(
                            source_name,
                            0.0,
                        )
                        + contribution
                    )

                    candidate_total += (
                        contribution
                    )

                total_counted_candidates += 1

                if candidate_total <= 0.0:
                    skipped_zero_contribution += 1

        total_contribution = sum(
            source_total_contributions.values()
        )

        if total_contribution <= 0.0:

            contribution_shares: Dict[
                str,
                float,
            ] = {}

        else:

            contribution_shares = {
                source_name: (
                    contribution
                    / total_contribution
                )
                for (
                    source_name,
                    contribution,
                ) in (
                    source_total_contributions.items()
                )
            }

        max_single_source_dominance = (
            max(
                contribution_shares.values()
            )
            if contribution_shares
            else 0.0
        )

        results[
            method.value
        ] = {
            "top_k": top_k,
            "source_contribution_shares": (
                contribution_shares
            ),
            "max_single_source_dominance": float(
                max_single_source_dominance
            ),
            "dominance_threshold": float(
                dominance_threshold
            ),
            "is_balanced": (
                max_single_source_dominance
                <= dominance_threshold
            ),
            "total_counted_candidates": (
                total_counted_candidates
            ),
            "skipped_zero_contribution": (
                skipped_zero_contribution
            ),
        }

    return results


# ============================================================
# SELF-CHECK
# ============================================================

def _self_check() -> None:
    """
    Internal QA-R2 checks.
    """

    # --------------------------------------------------------
    # 1. Empty input
    # --------------------------------------------------------
    for method in NormMethod:

        result = normalize_scores(
            [],
            method,
        )

        assert result.size == 0
        assert result.dtype == np.float64

    # --------------------------------------------------------
    # 2. N=1 consistency
    # --------------------------------------------------------
    for method in NormMethod:

        result = normalize_scores(
            [123.456],
            method,
        )

        assert result.shape == (1,)

        assert np.allclose(
            result,
            [0.5],
        )

    # --------------------------------------------------------
    # 3. Constant input
    # --------------------------------------------------------
    for method in NormMethod:

        result = normalize_scores(
            [5.0, 5.0, 5.0],
            method,
        )

        assert np.allclose(
            result,
            [0.5, 0.5, 0.5],
        )

    # --------------------------------------------------------
    # 4. Range [0,1]
    # --------------------------------------------------------
    test_scores = np.array(
        [
            -100.0,
            -2.0,
            0.0,
            0.5,
            1.0,
            50.0,
        ],
        dtype=np.float64,
    )

    for method in NormMethod:

        result = normalize_scores(
            test_scores,
            method,
        )

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    # --------------------------------------------------------
    # 5. Monotonicity
    # --------------------------------------------------------
    monotonic_input = np.array(
        [
            0.1,
            0.3,
            0.7,
            1.0,
        ],
        dtype=np.float64,
    )

    for method in NormMethod:

        result = normalize_scores(
            monotonic_input,
            method,
        )

        assert np.all(
            np.diff(result)
            >= -EPS
        ), (
            f"{method.value}: "
            "score cao hơn phải không "
            "thấp hơn score thấp hơn"
        )

    # --------------------------------------------------------
    # 6. Unsorted Rank
    # --------------------------------------------------------
    result = rank_normalize(
        [0.8, 0.9, 0.7]
    )

    assert np.allclose(
        result,
        [0.5, 1.0, 0.0],
    )

    # --------------------------------------------------------
    # 7. Rank tie
    # --------------------------------------------------------
    result = rank_normalize(
        [0.9, 0.8, 0.8, 0.7]
    )

    assert np.allclose(
        result,
        [1.0, 0.5, 0.5, 0.0],
    )

    # --------------------------------------------------------
    # 8. Empty source preservation
    # --------------------------------------------------------
    normalizer = (
        MultiSourceScoreNormalizer(
            method=NormMethod.RANK,
            weights={
                "clip_b": 1.0,
                "ocr": 0.5,
            },
        )
    )

    normalized, fused = (
        normalizer.normalize_source_dict(
            {
                "clip_b": {
                    "f1": 0.9,
                    "f2": 0.5,
                },
                "ocr": {},
            }
        )
    )

    assert "clip_b" in normalized
    assert "ocr" in normalized
    assert normalized["ocr"] == {}

    assert "f1" in fused
    assert "f2" in fused

    # --------------------------------------------------------
    # 9. Weighted fusion
    # --------------------------------------------------------
    _, fused = (
        normalizer.normalize_source_dict(
            {
                "clip_b": {
                    "A": 1.0,
                    "B": 0.0,
                },
                "ocr": {
                    "A": 10.0,
                    "B": 0.0,
                },
            }
        )
    )

    assert np.isclose(
        fused["A"],
        1.5,
    )

    assert np.isclose(
        fused["B"],
        0.0,
    )

    # --------------------------------------------------------
    # 10. Distribution comparison
    # --------------------------------------------------------
    distribution = (
        compare_normalization_distribution(
            [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ]
        )
    )

    assert set(
        distribution.keys()
    ) == {
        "min_max",
        "robust",
        "rank",
    }

    for stats in distribution.values():

        assert (
            0.0
            <= stats["min"]
            <= 1.0
        )

        assert (
            0.0
            <= stats["max"]
            <= 1.0
        )

        assert (
            0.0
            <= stats["mean"]
            <= 1.0
        )

        assert (
            0.0
            <= stats["median"]
            <= 1.0
        )

        assert stats["std"] >= 0.0

    # --------------------------------------------------------
    # 11. Min-Max outlier diagnostic
    # --------------------------------------------------------
    outlier_scores = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        1000.0,
    ]

    compression = (
        analyze_min_max_compression(
            outlier_scores,
            threshold=0.01,
        )
    )

    # min = 0.1 bị exclude.
    # 4 / 5 non-min candidates bị nén <= 0.01.
    assert compression[
        "compressed_fraction"
    ] >= 0.80

    assert np.isclose(
        compression[
            "excluded_min_fraction"
        ],
        1.0 / 6.0,
    )

    # --------------------------------------------------------
    # 12. Zero-floor must not be mistaken for compression
    # --------------------------------------------------------
    zero_floor_scores = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        10.0,
    ]

    zero_compression = (
        analyze_min_max_compression(
            zero_floor_scores,
            threshold=0.01,
        )
    )

    # Five natural floor values are excluded.
    # Candidate 10.0 is the only evaluated non-min candidate.
    assert zero_compression[
        "compressed_fraction"
    ] == 0.0

    assert np.isclose(
        zero_compression[
            "excluded_min_fraction"
        ],
        5.0 / 6.0,
    )

    # --------------------------------------------------------
    # 13. Dominance must use contribution share
    # --------------------------------------------------------
    dominance_test_data = [
        {
            "clip_b": {
                "A": 0.9,
                "B": 0.8,
            },
            "ocr": {
                "A": 1.0,
                "B": 0.2,
            },
            "asr": {
                "A": 1.0,
                "B": 0.1,
            },
        }
    ]

    dominance = (
        analyze_source_dominance(
            dominance_test_data,
            weights={
                "clip_b": 1.0,
                "ocr": 0.5,
                "asr": 0.3,
            },
            top_k=2,
            dominance_threshold=0.70,
        )
    )

    for (
        method_name,
        result,
    ) in dominance.items():

        shares = result[
            "source_contribution_shares"
        ]

        assert len(shares) >= 2, (
            f"{method_name}: "
            "dominance metric chỉ còn "
            "một source"
        )

        assert (
            0.0
            <= result[
                "max_single_source_dominance"
            ]
            <= 1.0
        )

        assert abs(
            sum(shares.values())
            - 1.0
        ) < 1e-9

    # --------------------------------------------------------
    # 14. NaN / Inf rejection
    # --------------------------------------------------------
    for bad_value in [
        math.nan,
        math.inf,
        -math.inf,
    ]:

        try:
            normalize_scores(
                [1.0, bad_value],
                NormMethod.RANK,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"{bad_value} phải bị reject"
            )

    # --------------------------------------------------------
    # 15. Invalid numeric/string input
    # --------------------------------------------------------
    try:
        normalize_scores(
            [1.0, "not-a-number"],
            NormMethod.RANK,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "String không convert được "
            "phải bị reject"
        )

    # --------------------------------------------------------
    # 16. Invalid weight
    # --------------------------------------------------------
    try:
        MultiSourceScoreNormalizer(
            weights={
                "clip_b": -1.0,
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative weight phải bị reject"
        )

    print(
        "[SELF-CHECK] PASS — "
        "all QA-R2 normalization checks passed."
    )


# ============================================================
# DEMO
# ============================================================

def _demo() -> None:
    """
    Demo dữ liệu giả lập.

    Mục tiêu:
        chứng minh raw scale khác nhau rõ rệt,
        đặc biệt OCR có outlier lớn.
    """

    print("=" * 76)
    print(
        "QA-R2 — MULTI-SOURCE SCORE NORMALIZATION"
    )
    print("=" * 76)

    np.random.seed(42)

    items = [
        f"Item_{i}"
        for i in range(1, 101)
    ]

    mock_raw_scores = {
        "clip_b": {
            item: float(
                np.random.uniform(
                    0.20,
                    0.35,
                )
            )
            for item in items
        },

        "ocr": {
            item: float(
                np.random.exponential(
                    scale=5.0
                )
            )
            for item in items
        },

        "asr": {
            item: float(
                np.random.uniform(
                    -50.0,
                    -5.0,
                )
            )
            for item in items
        },
    }

    mock_raw_scores[
        "ocr"
    ]["Item_50"] = 45.0

    # --------------------------------------------------------
    # RAW DISTRIBUTION
    # --------------------------------------------------------
    print(
        "\n1. RAW SCORE DISTRIBUTION"
    )

    for (
        source_name,
        score_map,
    ) in mock_raw_scores.items():

        values = np.fromiter(
            score_map.values(),
            dtype=np.float64,
        )

        print(
            f"   {source_name:10s} | "
            f"min={np.min(values):9.3f} | "
            f"max={np.max(values):9.3f} | "
            f"mean={np.mean(values):9.3f} | "
            f"median={np.median(values):9.3f}"
        )

    weights = {
        "clip_b": 1.0,
        "ocr": 0.5,
        "asr": 0.3,
    }

    # --------------------------------------------------------
    # NORMALIZATION + FUSION
    # --------------------------------------------------------
    print(
        "\n2. NORMALIZATION + WEIGHTED FUSION"
    )

    for method in NormMethod:

        normalizer = (
            MultiSourceScoreNormalizer(
                method=method,
                weights=weights,
            )
        )

        _, fused = (
            normalizer.normalize_source_dict(
                mock_raw_scores
            )
        )

        top3 = sorted(
            fused.items(),
            key=lambda pair: (
                -pair[1],
                pair[0],
            ),
        )[:3]

        print(
            f"\n   [{method.value.upper()}]"
        )

        print(
            f"   Top-3: {top3}"
        )

    # --------------------------------------------------------
    # DISTRIBUTION COMPARISON
    # --------------------------------------------------------
    print(
        "\n3. DISTRIBUTION COMPARISON"
    )

    for (
        source_name,
        score_map,
    ) in mock_raw_scores.items():

        values = np.fromiter(
            score_map.values(),
            dtype=np.float64,
        )

        comparison = (
            compare_normalization_distribution(
                values
            )
        )

        print(
            f"\n   Source: {source_name}"
        )

        for (
            method_name,
            stats,
        ) in comparison.items():

            print(
                f"      {method_name:8s} | "
                f"min={stats['min']:.3f} | "
                f"max={stats['max']:.3f} | "
                f"mean={stats['mean']:.3f} | "
                f"std={stats['std']:.3f}"
            )

    # --------------------------------------------------------
    # MIN-MAX COMPRESSION
    # --------------------------------------------------------
    print(
        "\n4. MIN-MAX OUTLIER COMPRESSION"
    )

    for (
        source_name,
        score_map,
    ) in mock_raw_scores.items():

        values = np.fromiter(
            score_map.values(),
            dtype=np.float64,
        )

        compression = (
            analyze_min_max_compression(
                values
            )
        )

        print(
            f"   {source_name:10s} | "
            f"compressed_fraction="
            f"{compression['compressed_fraction']:.3f} "
            f"(threshold="
            f"{compression['compression_threshold']:.3f}) "
            f"| excluded_min="
            f"{compression['excluded_min_fraction']:.3f}"
        )

    # --------------------------------------------------------
    # DOMINANCE
    # --------------------------------------------------------
    print(
        "\n5. DOMINANCE ANALYSIS"
    )

    demo_queries = [
        mock_raw_scores
        for _ in range(10)
    ]

    dominance = (
        analyze_source_dominance(
            demo_queries,
            weights=weights,
            top_k=10,
            dominance_threshold=0.70,
        )
    )

    print(
        json.dumps(
            dominance,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\n[QA-R2] Demo completed."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    _self_check()
    _demo()