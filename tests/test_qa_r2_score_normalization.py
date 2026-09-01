"""
QA-R2 — Tests for Multi-Source Score Normalization

Module under test:
    scripts.score_normalization

Mục tiêu:
1. Kiểm tra correctness của 3 phương pháp normalization.
2. Kiểm tra edge cases: empty, single-item, constant, ties, unsorted.
3. Kiểm tra fail-safe với NaN / Inf.
4. Kiểm tra sparse candidate sources.
5. Kiểm tra weighted fusion.
6. Kiểm tra dominance analysis.
7. Kiểm tra Min-Max outlier compression.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.score_normalization import (
    NormMethod,
    MultiSourceScoreNormalizer,
    analyze_source_dominance,
    min_max_scale,
    rank_normalize,
    robust_scale,
)


# ============================================================
# 1. BASIC NORMALIZATION
# ============================================================


def test_min_max_basic():
    scores = np.array([0.0, 5.0, 10.0], dtype=np.float64)

    result = min_max_scale(scores)

    assert np.allclose(result, [0.0, 0.5, 1.0])


def test_min_max_preserves_monotonicity():
    scores = np.array([1.0, 2.0, 5.0, 10.0], dtype=np.float64)

    result = min_max_scale(scores)

    assert np.all(np.diff(result) >= 0)
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


def test_min_max_constant_scores():
    scores = np.array([5.0, 5.0, 5.0], dtype=np.float64)

    result = min_max_scale(scores)

    assert np.all(np.isfinite(result))
    assert np.allclose(result, 0.5)


def test_min_max_empty():
    result = min_max_scale(np.array([], dtype=np.float64))

    assert result.size == 0


# ============================================================
# 2. RANK NORMALIZATION
# ============================================================


def test_rank_unsorted_input():
    """
    Quan trọng:
    Hàm phải map rank ngược về đúng vị trí ban đầu.

    Input:
        [0.8, 0.9, 0.7]

    Expected:
        0.8 -> 0.5
        0.9 -> 1.0
        0.7 -> 0.0

    => [0.5, 1.0, 0.0]
    """
    scores = np.array([0.8, 0.9, 0.7], dtype=np.float64)

    result = rank_normalize(scores)

    assert np.allclose(result, [0.5, 1.0, 0.0])


def test_rank_monotonicity():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    result = rank_normalize(scores)

    assert np.all(np.diff(result) >= 0)


def test_rank_single_item():
    result = rank_normalize(np.array([42.0], dtype=np.float64))

    assert result.shape == (1,)
    assert np.isfinite(result[0])
    assert 0.0 <= result[0] <= 1.0


def test_rank_constant_scores():
    scores = np.array([5.0, 5.0, 5.0], dtype=np.float64)

    result = rank_normalize(scores)

    assert np.all(np.isfinite(result))
    assert np.allclose(result, result[0])


def test_rank_ties_are_consistent():
    """
    Ties phải nhận cùng normalized score.

    Không chấp nhận việc hai candidate có cùng raw score
    nhưng nhận rank khác nhau chỉ vì thứ tự xuất hiện.
    """
    scores = np.array([0.9, 0.8, 0.8, 0.7], dtype=np.float64)

    result = rank_normalize(scores)

    assert result[1] == pytest.approx(result[2])


def test_rank_range():
    scores = np.array(
        [0.3, 0.9, 0.1, 0.5, 0.7],
        dtype=np.float64,
    )

    result = rank_normalize(scores)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    assert np.all(np.isfinite(result))


def test_rank_empty():
    result = rank_normalize(np.array([], dtype=np.float64))

    assert result.size == 0


# ============================================================
# 3. ROBUST NORMALIZATION
# ============================================================


def test_robust_basic():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    result = robust_scale(scores)

    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_robust_monotonicity():
    scores = np.array([1.0, 2.0, 5.0, 10.0], dtype=np.float64)

    result = robust_scale(scores)

    assert np.all(np.diff(result) >= 0)


def test_robust_constant_scores():
    scores = np.array([5.0, 5.0, 5.0], dtype=np.float64)

    result = robust_scale(scores)

    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_robust_outlier_is_bounded():
    scores = np.array(
        [1.0, 1.1, 1.2, 1.3, 1000.0],
        dtype=np.float64,
    )

    result = robust_scale(scores)

    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


# ============================================================
# 4. INVALID NUMERICAL INPUT
# ============================================================


@pytest.mark.parametrize(
    "func",
    [
        min_max_scale,
        rank_normalize,
        robust_scale,
    ],
)
def test_normalizers_reject_nan(func):
    scores = np.array([0.1, np.nan, 0.3], dtype=np.float64)

    with pytest.raises((ValueError, FloatingPointError)):
        func(scores)


@pytest.mark.parametrize(
    "func",
    [
        min_max_scale,
        rank_normalize,
        robust_scale,
    ],
)
def test_normalizers_reject_inf(func):
    scores = np.array([0.1, np.inf, 0.3], dtype=np.float64)

    with pytest.raises((ValueError, FloatingPointError)):
        func(scores)


# ============================================================
# 5. MULTI-SOURCE NORMALIZER
# ============================================================


def test_normalizer_preserves_empty_source_key():
    """
    Nếu source tồn tại nhưng raw_map = {},
    schema vẫn phải giữ key source -> {}.
    """
    raw = {
        "clip_b": {
            "f1": 0.8,
            "f2": 0.7,
        },
        "ocr": {},
    }

    normalizer = MultiSourceScoreNormalizer(
        method=NormMethod.RANK,
        weights={
            "clip_b": 1.0,
            "ocr": 0.5,
        },
    )

    normalized, fused = normalizer.normalize_source_dict(raw)

    assert "clip_b" in normalized
    assert "ocr" in normalized
    assert normalized["ocr"] == {}

    assert "f1" in fused
    assert "f2" in fused


def test_normalizer_sparse_sources():
    """
    Hai source có candidate pool khác nhau.

    clip_b: f1, f2, f3
    ocr:    f3, f4, f5

    Không được crash và phải giữ đúng union candidate IDs.
    """
    raw = {
        "clip_b": {
            "f1": 0.9,
            "f2": 0.8,
            "f3": 0.7,
        },
        "ocr": {
            "f3": 10.0,
            "f4": 5.0,
            "f5": 1.0,
        },
    }

    normalizer = MultiSourceScoreNormalizer(
        method=NormMethod.RANK,
        weights={
            "clip_b": 1.0,
            "ocr": 0.5,
        },
    )

    normalized, fused = normalizer.normalize_source_dict(raw)

    assert set(normalized["clip_b"]) == {"f1", "f2", "f3"}
    assert set(normalized["ocr"]) == {"f3", "f4", "f5"}

    assert set(fused) == {"f1", "f2", "f3", "f4", "f5"}

    assert math.isfinite(fused["f3"])


def test_weighted_fusion():
    raw = {
        "clip_b": {
            "f1": 1.0,
            "f2": 0.0,
        },
        "ocr": {
            "f1": 10.0,
            "f2": 0.0,
        },
    }

    weights = {
        "clip_b": 1.0,
        "ocr": 0.5,
    }

    normalizer = MultiSourceScoreNormalizer(
        method=NormMethod.RANK,
        weights=weights,
    )

    normalized, fused = normalizer.normalize_source_dict(raw)

    assert fused["f1"] > fused["f2"]

    expected_f1 = (
        normalized["clip_b"]["f1"] * 1.0
        + normalized["ocr"]["f1"] * 0.5
    )

    assert fused["f1"] == pytest.approx(expected_f1)


def test_unknown_source_uses_default_weight():
    raw = {
        "unknown_source": {
            "f1": 0.2,
            "f2": 0.1,
        }
    }

    normalizer = MultiSourceScoreNormalizer(
        method=NormMethod.RANK,
        weights={},
    )

    _, fused = normalizer.normalize_source_dict(raw)

    assert "f1" in fused
    assert "f2" in fused
    assert fused["f1"] > fused["f2"]


# ============================================================
# 6. OUTLIER COMPRESSION — MIN-MAX
# ============================================================


def test_min_max_outlier_compression():
    """
    Demonstrate the known limitation of Min-Max.

    One extreme OCR outlier should compress most other scores
    close to zero.
    """
    scores = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 1000.0],
        dtype=np.float64,
    )

    result = min_max_scale(scores)

    compressed = result[:-1]

    assert result[-1] == pytest.approx(1.0)

    # All non-outliers should be very close to zero.
    assert np.max(compressed) < 0.001


# ============================================================
# 7. DOMINANCE ANALYSIS
# ============================================================


def test_dominance_analysis_returns_all_methods():
    raw = {
        "clip_b": {
            "f1": 0.9,
            "f2": 0.8,
            "f3": 0.7,
        },
        "ocr": {
            "f1": 10.0,
            "f2": 5.0,
            "f3": 1.0,
        },
        "asr": {
            "f1": -1.0,
            "f2": -2.0,
            "f3": -3.0,
        },
    }

    weights = {
        "clip_b": 1.0,
        "ocr": 0.5,
        "asr": 0.3,
    }

    result = analyze_source_dominance(
        [raw],
        weights=weights,
    )

    assert set(result.keys()) == {
        "min_max",
        "robust",
        "rank",
    }

    for method_name, method_result in result.items():
        assert "top_k" in method_result
        assert "source_contribution_shares" in method_result
        assert "max_single_source_dominance" in method_result
        assert "dominance_threshold" in method_result
        assert "is_balanced" in method_result

        shares = method_result["source_contribution_shares"]

        assert all(
            np.isfinite(value)
            for value in shares.values()
        )


def test_dominance_shares_sum_to_one():
    raw = {
        "clip_b": {
            "f1": 0.9,
            "f2": 0.8,
            "f3": 0.7,
        },
        "ocr": {
            "f1": 10.0,
            "f2": 5.0,
            "f3": 1.0,
        },
        "asr": {
            "f1": -1.0,
            "f2": -2.0,
            "f3": -3.0,
        },
    }

    weights = {
        "clip_b": 1.0,
        "ocr": 0.5,
        "asr": 0.3,
    }

    result = analyze_source_dominance(
        [raw],
        weights=weights,
    )

    for method_result in result.values():
        shares = method_result["source_contribution_shares"]

        if shares:
            assert sum(shares.values()) == pytest.approx(1.0)


def test_dominance_does_not_depend_on_dict_order():
    """
    Cùng dữ liệu nhưng thay đổi thứ tự source trong dict
    phải cho cùng dominance result.
    """
    raw_a = {
        "clip_b": {
            "f1": 0.9,
            "f2": 0.8,
        },
        "ocr": {
            "f1": 10.0,
            "f2": 1.0,
        },
    }

    raw_b = {
        "ocr": {
            "f1": 10.0,
            "f2": 1.0,
        },
        "clip_b": {
            "f1": 0.9,
            "f2": 0.8,
        },
    }

    weights = {
        "clip_b": 1.0,
        "ocr": 0.5,
    }

    result_a = analyze_source_dominance(
        [raw_a],
        weights=weights,
    )

    result_b = analyze_source_dominance(
        [raw_b],
        weights=weights,
    )

    assert result_a == result_b


# ============================================================
# 8. DOMINANCE ZERO-CONTRIBUTION EDGE CASE
# ============================================================


def test_dominance_handles_zero_contribution_safely():
    """
    Không được chọn source chỉ vì nó đứng đầu dictionary
    khi toàn bộ contribution đều bằng 0.
    """
    raw = {
        "clip_b": {},
        "ocr": {},
        "asr": {},
    }

    weights = {
        "clip_b": 1.0,
        "ocr": 0.5,
        "asr": 0.3,
    }

    result = analyze_source_dominance(
        [raw],
        weights=weights,
    )

    for method_result in result.values():
        assert method_result["max_single_source_dominance"] == 0.0
        assert method_result["is_balanced"] is True


# ============================================================
# 9. REGRESSION: FUSED SCORE MUST BE FINITE
# ============================================================


@pytest.mark.parametrize(
    "method",
    list(NormMethod),
)
def test_fused_scores_are_finite(method):
    raw = {
        "clip_b": {
            "f1": 0.2,
            "f2": 0.5,
            "f3": 0.9,
        },
        "ocr": {
            "f1": 0.0,
            "f2": 10.0,
            "f3": 2.0,
        },
        "asr": {
            "f1": -50.0,
            "f2": -10.0,
            "f3": -5.0,
        },
    }

    normalizer = MultiSourceScoreNormalizer(
        method=method,
        weights={
            "clip_b": 1.0,
            "ocr": 0.5,
            "asr": 0.3,
        },
    )

    _, fused = normalizer.normalize_source_dict(raw)

    assert fused
    assert all(np.isfinite(value) for value in fused.values())


# ============================================================
# 10. NORMALIZED SCORES MUST STAY IN [0, 1]
# ============================================================


@pytest.mark.parametrize(
    "method",
    list(NormMethod),
)
def test_normalized_scores_are_bounded(method):
    raw = {
        "clip_b": {
            "f1": 0.2,
            "f2": 0.5,
            "f3": 0.9,
        },
        "ocr": {
            "f1": 0.0,
            "f2": 1000.0,
            "f3": 2.0,
        },
        "asr": {
            "f1": -100.0,
            "f2": -10.0,
            "f3": -5.0,
        },
    }

    normalizer = MultiSourceScoreNormalizer(
        method=method,
        weights={
            "clip_b": 1.0,
            "ocr": 0.5,
            "asr": 0.3,
        },
    )

    normalized, _ = normalizer.normalize_source_dict(raw)

    for source_scores in normalized.values():
        for value in source_scores.values():
            assert np.isfinite(value)
            assert 0.0 <= value <= 1.0


# ============================================================
# 11. DATA CONTRACT — SOURCE NAME / CANDIDATE IDs
# ============================================================


def test_candidate_ids_are_preserved():
    raw = {
        "clip_b": {
            "frame_001": 0.91,
            "frame_999": 0.82,
        },
        "ocr": {
            "frame_001": 12.5,
        },
    }

    normalizer = MultiSourceScoreNormalizer(
        method=NormMethod.RANK,
        weights={
            "clip_b": 1.0,
            "ocr": 0.5,
        },
    )

    normalized, fused = normalizer.normalize_source_dict(raw)

    assert set(normalized["clip_b"].keys()) == {
        "frame_001",
        "frame_999",
    }

    assert set(normalized["ocr"].keys()) == {
        "frame_001",
    }

    assert set(fused.keys()) == {
        "frame_001",
        "frame_999",
    }