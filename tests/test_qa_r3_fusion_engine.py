"""
QA-R3 — Fusion Engine Contract Tests
=====================================

Khóa contract cho:
    - Semantic coverage
    - Constraint-weighted overall coverage
    - Visual similarity
    - Temporal redundancy
    - Semantic redundancy
    - Greedy selection
    - MMR selection
    - Top-K constraint
    - Edge cases
    - Determinism
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.fusion_engine import (
    CandidateMetadata,
    MultiModalFusionEngine,
    SelectionMethod,
    SemanticQuery,
    compute_marginal_semantic_coverage,
    compute_semantic_coverage,
    semantic_similarity,
    temporal_similarity,
    visual_similarity,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def query() -> SemanticQuery:
    return SemanticQuery(
        entities={"person"},
        actions={
            "run",
            "jump",
            "turn",
            "stop",
            "open",
        },
        attributes={"red"},
        relations={"person-near-car"},
    )


@pytest.fixture
def metadata() -> dict[str, CandidateMetadata]:
    return {
        "c1": CandidateMetadata(
            candidate_id="c1",
            video_id="video_1",
            frame_index=100,
            timestamp_sec=10.0,
            entities={"person"},
            feature_vector=np.asarray(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        ),
        "c2": CandidateMetadata(
            candidate_id="c2",
            video_id="video_2",
            frame_index=200,
            timestamp_sec=50.0,
            actions={
                "run",
                "jump",
                "turn",
                "stop",
            },
            feature_vector=np.asarray(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            ),
        ),
        "c3": CandidateMetadata(
            candidate_id="c3",
            video_id="video_3",
            frame_index=300,
            timestamp_sec=80.0,
            relations={
                "person-near-car",
            },
            feature_vector=np.asarray(
                [0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        ),
        "c4": CandidateMetadata(
            candidate_id="c4",
            video_id="video_1",
            frame_index=105,
            timestamp_sec=11.0,
            entities={"person"},
            feature_vector=np.asarray(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        ),
    }


@pytest.fixture
def fused_scores() -> dict[str, float]:
    return {
        "c1": 0.95,
        "c2": 0.90,
        "c3": 0.85,
        "c4": 0.94,
    }


@pytest.fixture
def engine() -> MultiModalFusionEngine:
    return MultiModalFusionEngine(
        sage_lambda=0.3,
        time_threshold_sec=3.0,
        lambda_mmr=0.7,
    )


# ============================================================================
# 1. SEMANTIC COVERAGE
# ============================================================================


def test_entity_coverage(
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
) -> None:
    coverage = compute_semantic_coverage(
        query,
        metadata["c1"],
    )

    assert coverage.entity == 1.0
    assert coverage.matched_entity == 1
    assert coverage.total_entity == 1


def test_action_coverage(
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
) -> None:
    coverage = compute_semantic_coverage(
        query,
        metadata["c2"],
    )

    assert coverage.action == pytest.approx(4.0 / 5.0)
    assert coverage.matched_action == 4
    assert coverage.total_action == 5


def test_relation_coverage(
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
) -> None:
    coverage = compute_semantic_coverage(
        query,
        metadata["c3"],
    )

    assert coverage.relation == 1.0
    assert coverage.matched_relation == 1
    assert coverage.total_relation == 1


# ============================================================================
# 2. CONSTRAINT-WEIGHTED OVERALL
# ============================================================================


def test_overall_coverage_is_constraint_weighted(
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
) -> None:
    """
    Query có:

        1 entity
        5 actions
        1 attribute
        1 relation

    Tổng = 8 constraints.

    c1 chỉ phủ entity:

        1 / 8

    Không được tính:

        (1.0 + 0 + 0 + 0) / 4 = 0.25
    """

    coverage = compute_semantic_coverage(
        query,
        metadata["c1"],
    )

    assert coverage.matched_total == 1
    assert coverage.total_constraints == 8
    assert coverage.overall == pytest.approx(1.0 / 8.0)


def test_action_rich_candidate_beats_single_entity_candidate(
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
) -> None:
    """
    Regression cho lỗi group averaging.

    Query có tổng cộng 8 constraints:

        1 entity
        5 actions
        1 attribute
        1 relation

    c1:
        1 / 8

    c2:
        4 / 8

    => c2 phải có overall coverage cao hơn c1.
    """

    c1 = compute_semantic_coverage(
        query,
        metadata["c1"],
    )

    c2 = compute_semantic_coverage(
        query,
        metadata["c2"],
    )

    assert c1.matched_total == 1
    assert c1.total_constraints == 8
    assert c1.overall == pytest.approx(1.0 / 8.0)

    assert c2.matched_total == 4
    assert c2.total_constraints == 8
    assert c2.overall == pytest.approx(4.0 / 8.0)

    assert c2.overall > c1.overall


def test_empty_query_has_zero_coverage(
    metadata: dict[str, CandidateMetadata],
) -> None:
    query = SemanticQuery()

    coverage = compute_semantic_coverage(
        query,
        metadata["c1"],
    )

    assert coverage.overall == 0.0
    assert coverage.total_constraints == 0
    assert coverage.matched_total == 0


# ============================================================================
# 3. MARGINAL SEMANTIC COVERAGE
# ============================================================================


def test_marginal_coverage_counts_only_new_constraints(
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
) -> None:
    """
    c1 đã phủ entity.

    Khi xét c4, c4 cũng chỉ có entity.

    => marginal coverage = 0.
    """

    value = compute_marginal_semantic_coverage(
        query,
        metadata["c4"],
        [metadata["c1"]],
    )

    assert value == 0.0


def test_marginal_coverage_rewards_new_actions(
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
) -> None:
    """
    Sau khi c1 phủ entity, c2 bổ sung 4 actions.

    Tổng query constraints = 8.

    Marginal gain = 4 / 8.
    """

    value = compute_marginal_semantic_coverage(
        query,
        metadata["c2"],
        [metadata["c1"]],
    )

    assert value == pytest.approx(4.0 / 8.0)


# ============================================================================
# 4. VISUAL SIMILARITY
# ============================================================================


def _candidate_with_vector(
    cid: str,
    vector: list[float],
) -> CandidateMetadata:
    return CandidateMetadata(
        candidate_id=cid,
        video_id=f"video_{cid}",
        frame_index=0,
        timestamp_sec=0.0,
        feature_vector=np.asarray(
            vector,
            dtype=np.float32,
        ),
    )


def test_visual_similarity_zero_cosine_is_zero() -> None:
    candidate = _candidate_with_vector(
        "a",
        [1.0, 0.0],
    )

    selected = _candidate_with_vector(
        "b",
        [0.0, 1.0],
    )

    value = visual_similarity(
        candidate,
        selected,
    )

    # IMPORTANT:
    # Không được biến cosine 0 thành 0.5.
    assert value == pytest.approx(0.0)


def test_visual_similarity_identical_vectors_is_one() -> None:
    candidate = _candidate_with_vector(
        "a",
        [1.0, 0.0],
    )

    selected = _candidate_with_vector(
        "b",
        [1.0, 0.0],
    )

    value = visual_similarity(
        candidate,
        selected,
    )

    assert value == pytest.approx(1.0)


def test_visual_similarity_negative_cosine_is_clipped_to_zero() -> None:
    candidate = _candidate_with_vector(
        "a",
        [1.0, 0.0],
    )

    selected = _candidate_with_vector(
        "b",
        [-1.0, 0.0],
    )

    value = visual_similarity(
        candidate,
        selected,
    )

    assert value == pytest.approx(0.0)


def test_visual_similarity_missing_vector_is_zero() -> None:
    candidate = _candidate_with_vector(
        "a",
        [1.0, 0.0],
    )

    selected = CandidateMetadata(
        candidate_id="b",
        video_id="video_b",
        frame_index=0,
        timestamp_sec=0.0,
    )

    value = visual_similarity(
        candidate,
        selected,
    )

    assert value == 0.0


# ============================================================================
# 5. TEMPORAL REDUNDANCY
# ============================================================================


def test_temporal_similarity_same_video_nearby_is_high() -> None:
    candidate = CandidateMetadata(
        candidate_id="a",
        video_id="video_1",
        frame_index=1,
        timestamp_sec=10.0,
    )

    selected = CandidateMetadata(
        candidate_id="b",
        video_id="video_1",
        frame_index=2,
        timestamp_sec=10.5,
    )

    value = temporal_similarity(
        candidate,
        selected,
        threshold_sec=3.0,
    )

    assert value == pytest.approx(
        1.0 - 0.5 / 3.0
    )


def test_temporal_similarity_same_timestamp_is_one() -> None:
    candidate = CandidateMetadata(
        candidate_id="a",
        video_id="video_1",
        frame_index=1,
        timestamp_sec=10.0,
    )

    selected = CandidateMetadata(
        candidate_id="b",
        video_id="video_1",
        frame_index=2,
        timestamp_sec=10.0,
    )

    assert temporal_similarity(
        candidate,
        selected,
        threshold_sec=3.0,
    ) == 1.0


def test_temporal_similarity_different_video_is_zero() -> None:
    candidate = CandidateMetadata(
        candidate_id="a",
        video_id="video_1",
        frame_index=1,
        timestamp_sec=10.0,
    )

    selected = CandidateMetadata(
        candidate_id="b",
        video_id="video_2",
        frame_index=2,
        timestamp_sec=10.1,
    )

    assert temporal_similarity(
        candidate,
        selected,
        threshold_sec=3.0,
    ) == 0.0


def test_temporal_similarity_outside_threshold_is_zero() -> None:
    candidate = CandidateMetadata(
        candidate_id="a",
        video_id="video_1",
        frame_index=1,
        timestamp_sec=10.0,
    )

    selected = CandidateMetadata(
        candidate_id="b",
        video_id="video_1",
        frame_index=2,
        timestamp_sec=13.0,
    )

    assert temporal_similarity(
        candidate,
        selected,
        threshold_sec=3.0,
    ) == 0.0


# ============================================================================
# 6. GREEDY SELECTION
# ============================================================================


def test_greedy_respects_top_k(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    result = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=2,
        selection_method=SelectionMethod.GREEDY,
    )

    assert len(result) == 2


def test_greedy_does_not_return_duplicate_ids(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    result = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=10,
        selection_method=SelectionMethod.GREEDY,
    )

    ids = [
        item.candidate_id
        for item in result
    ]

    assert len(ids) == len(set(ids))


def test_greedy_prefers_semantic_complement(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    """
    c1 và c4 gần như cùng cảnh.

    c2 bổ sung 4 actions.
    c3 bổ sung relation.

    Greedy không nên lấy cả c1 + c4 trước khi lấy
    các candidate bổ sung semantic coverage.
    """

    result = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=3,
        selection_method=SelectionMethod.GREEDY,
    )

    ids = [
        item.candidate_id
        for item in result
    ]

    assert "c2" in ids
    assert "c3" in ids


def test_greedy_is_deterministic(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    result_a = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=3,
        selection_method=SelectionMethod.GREEDY,
    )

    result_b = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=3,
        selection_method=SelectionMethod.GREEDY,
    )

    ids_a = [
        item.candidate_id
        for item in result_a
    ]

    ids_b = [
        item.candidate_id
        for item in result_b
    ]

    assert ids_a == ids_b


# ============================================================================
# 7. MMR
# ============================================================================


def test_mmr_respects_top_k(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    result = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=2,
        selection_method=SelectionMethod.MMR,
    )

    assert len(result) == 2


def test_mmr_returns_unique_candidates(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    result = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=10,
        selection_method=SelectionMethod.MMR,
    )

    ids = [
        item.candidate_id
        for item in result
    ]

    assert len(ids) == len(set(ids))


def test_mmr_is_deterministic(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    result_a = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=3,
        selection_method=SelectionMethod.MMR,
    )

    result_b = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=3,
        selection_method=SelectionMethod.MMR,
    )

    assert [
        x.candidate_id
        for x in result_a
    ] == [
        x.candidate_id
        for x in result_b
    ]


# ============================================================================
# 8. EDGE CASES
# ============================================================================


def test_empty_candidate_pool(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
) -> None:
    result = engine.select_top_k(
        fused_scores={},
        metadata_map={},
        query=query,
        top_k=10,
    )

    assert result == []


def test_top_k_larger_than_candidate_pool(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    result = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=1000,
    )

    assert len(result) <= len(fused_scores)


def test_zero_top_k_raises(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        engine.select_top_k(
            fused_scores=fused_scores,
            metadata_map=metadata,
            query=query,
            top_k=0,
        )


def test_negative_top_k_raises(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        engine.select_top_k(
            fused_scores=fused_scores,
            metadata_map=metadata,
            query=query,
            top_k=-1,
        )


def test_missing_metadata_does_not_crash(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
) -> None:
    fused_scores = {
        "unknown": 0.9,
    }

    result = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map={},
        query=query,
        top_k=1,
    )

    assert len(result) == 1
    assert result[0].candidate_id == "unknown"


# ============================================================================
# 9. INVALID / ZERO FEATURE VECTORS
# ============================================================================


def test_zero_vector_visual_similarity_is_zero() -> None:
    candidate = _candidate_with_vector(
        "a",
        [0.0, 0.0],
    )

    selected = _candidate_with_vector(
        "b",
        [1.0, 0.0],
    )

    assert visual_similarity(
        candidate,
        selected,
    ) == 0.0


def test_dimension_mismatch_visual_similarity_is_zero() -> None:
    candidate = _candidate_with_vector(
        "a",
        [1.0, 0.0],
    )

    selected = _candidate_with_vector(
        "b",
        [1.0, 0.0, 0.0],
    )

    assert visual_similarity(
        candidate,
        selected,
    ) == 0.0


# ============================================================================
# 10. OUTPUT CONTRACT
# ============================================================================


def test_output_rank_is_one_based(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    result = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=3,
    )

    assert [
        item.selected_rank
        for item in result
    ] == list(range(1, len(result) + 1))


def test_output_contains_coverage_breakdown(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    result = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=1,
    )

    assert len(result) == 1
    assert result[0].coverage_breakdown is not None


def test_scores_are_finite(
    engine: MultiModalFusionEngine,
    query: SemanticQuery,
    metadata: dict[str, CandidateMetadata],
    fused_scores: dict[str, float],
) -> None:
    result = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=10,
    )

    for item in result:
        assert np.isfinite(item.fused_score)
        assert np.isfinite(item.coverage_score)
        assert np.isfinite(item.sage_score)
        assert 0.0 <= item.coverage_score <= 1.0


# ============================================================================
# 11. BACKWARD COMPATIBILITY
# ============================================================================


def test_legacy_fuse_and_select_api(
    engine: MultiModalFusionEngine,
    metadata: dict[str, CandidateMetadata],
) -> None:
    raw_scores = {
        "clip": {
            "c1": 0.9,
            "c2": 0.8,
            "c3": 0.7,
        },
        "ocr": {
            "c1": 0.8,
            "c2": 0.6,
            "c3": 0.9,
        },
    }

    result = engine.fuse_and_select(
        source_raw_scores=raw_scores,
        metadata_map=metadata,
        query_entities={"person"},
        top_k=2,
        selection_method=SelectionMethod.GREEDY,
    )

    assert len(result) <= 2
    assert all(
        item.candidate_id
        in {"c1", "c2", "c3"}
        for item in result
    )