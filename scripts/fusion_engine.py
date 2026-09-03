"""
QA-R3 — Semantic Coverage + Diversity Selection
================================================

Mục đích
--------
Chọn Top-K frame từ candidate pool bằng cách kết hợp:

    1. Fused relevance score từ upstream QA-R1 / QA-R2.
    2. Semantic coverage:
       entity / action / attribute / relation.
    3. Temporal redundancy.
    4. Visual redundancy nếu có feature vector.
    5. Semantic redundancy.
    6. Greedy hoặc MMR selection.

Pipeline
--------
    candidate pool
        ↓
    base SAGE
        ↓
    precompute/vectorized redundancy
        ↓
    Greedy / MMR
        ↓
    diverse + semantically-covered Top-K

Design constraints
------------------
- Không tự ý thay đổi R1/R2 funnel.
- Không tự fusion lại raw CLIP/OCR/ASR trong R3 selector.
- Redundancy phụ thuộc selected set nên phải được cập nhật incrementally.
- Candidate pool mục tiêu: N <= 1,000.
- Selection target: K <= 50.
- Latency là benchmark target, không phải hard-coded guarantee.

Performance design
------------------
Hot path của R3 phải tránh:

    O(K * N * Python-set-work)
    O(K * N * np.linalg.norm)
    O(K * N * pair function calls)
    sorted(remaining) ở mỗi vòng

Thay vào đó:

    1. Semantic query labels được encode thành packed integer bitmask.
    2. Marginal semantic gain dùng vectorized bit operations + popcount LUT.
    3. Pair redundancy được vectorize một lần cho candidate pool.
    4. Selection dùng NumPy argmax / maximum.
    5. Redundancy matrix được cache theo candidate pool.
    6. Tie-break deterministic được xử lý ổn định.

Semantic packed mask
--------------------
Mỗi semantic group vẫn giữ contract cũ:

    entity    <= 16 labels
    action    <= 16 labels
    attribute <= 16 labels
    relation  <= 16 labels

Nhưng selection hot path không còn phải xử lý 4 mask độc lập.

Các group được pack liên tiếp vào một integer mask:

    entity    -> bits [0, 15]
    action    -> bits [16, 31]
    attribute -> bits [32, 47]
    relation  -> bits [48, 63]

Dtype được chọn theo tổng số bit cần thiết:

    <= 16 -> uint16
    <= 32 -> uint32
    <= 64 -> uint64

Do đó benchmark hiện tại với 12 constraints dùng uint16.

SAGE
-----
Base score:

    SAGE(c)
        = relevance(c)
        + sage_lambda * semantic_coverage(c)

Greedy
------
    gain(c)
        = SAGE(c)
        + marginal_coverage_weight * marginal_semantic_gain(c)
        - redundancy_penalty_weight * max_redundancy(c)

MMR
---
    MMR(c)
        = lambda_mmr * SAGE(c)
        - (1 - lambda_mmr) * max_redundancy(c)

Redundancy
----------
    pair_redundancy =
        normalized weighted combination of:

            temporal similarity
            visual cosine similarity
            semantic Jaccard similarity

For performance, pairwise redundancy is represented internally as:

    redundancy_matrix[i, j]

and:

    max_redundancy
        = max(max_redundancy, redundancy_matrix[selected])

This keeps the selection update vectorized.

Cache contract
--------------
CandidateMetadata được xem như immutable trong thời gian redundancy
cache còn hiệu lực.

CandidateMetadata là frozen dataclass, nhưng các Set / ndarray bên
trong về mặt Python vẫn có thể bị mutate. Caller không được mutate
semantic sets hoặc feature_vector của metadata object sau khi object
được đưa vào engine nếu muốn cache correctness được đảm bảo.

Nếu metadata object được thay thế bằng object mới, cache signature sẽ
được invalidated thông qua object identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

_MAX_MASK_BITS = 16
_MAX_PACKED_BITS = 64

# Fast popcount LUT.
#
# uint16 is the native representation used by the current benchmark
# (12 total semantic constraints). Larger packed masks are supported by
# viewing them as uint16 chunks and summing the LUT results.
_POPCOUNT_LUT = np.asarray(
    [int(i).bit_count() for i in range(1 << _MAX_MASK_BITS)],
    dtype=np.uint8,
)


# ============================================================================
# ENUMS
# ============================================================================


class SelectionMethod(str, Enum):
    GREEDY = "greedy"
    MMR = "mmr"


# ============================================================================
# DATA CONTRACTS
# ============================================================================


@dataclass(slots=True, frozen=True)
class SemanticQuery:
    """
    Semantic constraints của query.

    Các label phải được chuẩn hóa trước khi đưa vào engine.

    Query hiện tại được tối ưu với tối đa 16 labels / semantic group.
    """

    entities: Set[str] = field(default_factory=set)
    actions: Set[str] = field(default_factory=set)
    attributes: Set[str] = field(default_factory=set)
    relations: Set[str] = field(default_factory=set)


@dataclass(slots=True, frozen=True)
class CandidateMetadata:
    """
    Metadata phục vụ semantic coverage và diversity selection.

    Cache contract:
        Metadata object được xem như immutable trong suốt thời gian
        redundancy cache còn hiệu lực.

    Lưu ý:
        frozen=True chỉ khóa attribute assignment của dataclass.
        Các Set / ndarray bên trong vẫn mutable về mặt Python.
        Caller không nên mutate chúng sau khi đưa vào engine.
    """

    candidate_id: str
    video_id: str
    frame_index: int
    timestamp_sec: float

    entities: Set[str] = field(default_factory=set)
    actions: Set[str] = field(default_factory=set)
    attributes: Set[str] = field(default_factory=set)
    relations: Set[str] = field(default_factory=set)

    feature_vector: Optional[np.ndarray] = None

    # Compatibility với caller cũ.
    text_content: str = ""


@dataclass(slots=True, frozen=True)
class CoverageBreakdown:
    """
    Coverage theo từng semantic group.

    Overall coverage KHÔNG phải trung bình đơn giản của 4 group.

    Nó được tính theo tổng số query constraints:

        matched_total / query_constraint_total
    """

    entity: float
    action: float
    attribute: float
    relation: float

    matched_entity: int
    matched_action: int
    matched_attribute: int
    matched_relation: int

    total_entity: int
    total_action: int
    total_attribute: int
    total_relation: int

    @property
    def matched_total(self) -> int:
        return (
            self.matched_entity
            + self.matched_action
            + self.matched_attribute
            + self.matched_relation
        )

    @property
    def total_constraints(self) -> int:
        return (
            self.total_entity
            + self.total_action
            + self.total_attribute
            + self.total_relation
        )

    @property
    def overall(self) -> float:
        total = self.total_constraints

        if total <= 0:
            return 0.0

        return float(self.matched_total / total)


@dataclass(slots=True, frozen=True)
class FusedCandidate:
    """
    Output của R3 selection.

    fused_score là fused relevance đã được normalize về [0, 1].

    selection_score là gain thực tại vòng Greedy/MMR mà candidate
    được chọn. Đây là score chuyển tiếp sang QA-R4 vì nó phản ánh đúng
    semantic coverage và diversity của R3.
    """

    candidate_id: str
    video_id: str
    frame_id: int

    fused_score: float
    coverage_score: float
    sage_score: float
    selection_score: float

    selected_rank: int

    coverage_breakdown: Optional[CoverageBreakdown] = None


# ============================================================================
# BASIC HELPERS
# ============================================================================


def _safe_float(value: float) -> float:
    value = float(value)

    if not np.isfinite(value):
        return 0.0

    return value


def _normalize_score(value: float) -> float:
    """
    Fused relevance được kỳ vọng đã normalized upstream.

    Hàm này chỉ bảo vệ range [0, 1].
    """

    value = _safe_float(value)

    return float(np.clip(value, 0.0, 1.0))


def cosine_similarity(
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    """
    Cosine similarity an toàn.

    Returns:
        [-1, 1]

    Vector zero / invalid / khác dimension => 0.
    """

    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)

    if a.ndim != 1 or b.ndim != 1:
        return 0.0

    if a.shape != b.shape:
        return 0.0

    norm1 = float(np.linalg.norm(a))
    norm2 = float(np.linalg.norm(b))

    if norm1 <= 1e-8 or norm2 <= 1e-8:
        return 0.0

    value = float(np.dot(a, b) / (norm1 * norm2))

    if not np.isfinite(value):
        return 0.0

    return float(np.clip(value, -1.0, 1.0))


# ============================================================================
# SEMANTIC COVERAGE
# ============================================================================


def _coverage_ratio(
    query_labels: Set[str],
    candidate_labels: Set[str],
) -> Tuple[int, int, float]:
    """
    Returns:
        matched_count,
        total_count,
        ratio
    """

    total = len(query_labels)

    if total == 0:
        return 0, 0, 0.0

    matched = len(
        query_labels.intersection(candidate_labels)
    )

    return (
        matched,
        total,
        float(matched / total),
    )


def compute_semantic_coverage(
    query: SemanticQuery,
    candidate: CandidateMetadata,
) -> CoverageBreakdown:
    """
    Tính coverage theo:

        entity
        action
        attribute
        relation

    Overall được weighted theo số lượng constraint thực tế.
    """

    entity_matched, entity_total, entity_ratio = _coverage_ratio(
        query.entities,
        candidate.entities,
    )

    action_matched, action_total, action_ratio = _coverage_ratio(
        query.actions,
        candidate.actions,
    )

    attribute_matched, attribute_total, attribute_ratio = _coverage_ratio(
        query.attributes,
        candidate.attributes,
    )

    relation_matched, relation_total, relation_ratio = _coverage_ratio(
        query.relations,
        candidate.relations,
    )

    return CoverageBreakdown(
        entity=entity_ratio,
        action=action_ratio,
        attribute=attribute_ratio,
        relation=relation_ratio,
        matched_entity=entity_matched,
        matched_action=action_matched,
        matched_attribute=attribute_matched,
        matched_relation=relation_matched,
        total_entity=entity_total,
        total_action=action_total,
        total_attribute=attribute_total,
        total_relation=relation_total,
    )


def compute_semantic_coverage_score(
    query: SemanticQuery,
    candidate: CandidateMetadata,
) -> float:
    return compute_semantic_coverage(
        query,
        candidate,
    ).overall


# ============================================================================
# MARGINAL SEMANTIC COVERAGE
# ============================================================================


def _covered_labels(
    selected_metadata: Sequence[CandidateMetadata],
    field_name: str,
) -> Set[str]:
    """
    Compatibility helper.

    Không được dùng trong hot selection loop.
    """

    result: Set[str] = set()

    for meta in selected_metadata:
        result.update(
            getattr(meta, field_name)
        )

    return result


def _marginal_ratio(
    query_labels: Set[str],
    candidate_labels: Set[str],
    already_covered: Set[str],
) -> Tuple[int, int]:
    total = len(query_labels)

    if total == 0:
        return 0, 0

    new_labels = (
        query_labels.intersection(candidate_labels)
        - already_covered
    )

    return len(new_labels), total


def compute_marginal_semantic_coverage(
    query: SemanticQuery,
    candidate: CandidateMetadata,
    selected_metadata: Sequence[CandidateMetadata],
) -> float:
    """
    Compatibility/public helper.

    Lưu ý:
        Hàm này không được gọi trong hot loop của selector.

    Selector sử dụng packed bitmask incremental để tránh rebuild
    selected union ở từng candidate.
    """

    covered_entities = _covered_labels(
        selected_metadata,
        "entities",
    )

    covered_actions = _covered_labels(
        selected_metadata,
        "actions",
    )

    covered_attributes = _covered_labels(
        selected_metadata,
        "attributes",
    )

    covered_relations = _covered_labels(
        selected_metadata,
        "relations",
    )

    new_matched = 0
    total_constraints = 0

    matched, total = _marginal_ratio(
        query.entities,
        candidate.entities,
        covered_entities,
    )
    new_matched += matched
    total_constraints += total

    matched, total = _marginal_ratio(
        query.actions,
        candidate.actions,
        covered_actions,
    )
    new_matched += matched
    total_constraints += total

    matched, total = _marginal_ratio(
        query.attributes,
        candidate.attributes,
        covered_attributes,
    )
    new_matched += matched
    total_constraints += total

    matched, total = _marginal_ratio(
        query.relations,
        candidate.relations,
        covered_relations,
    )
    new_matched += matched
    total_constraints += total

    if total_constraints <= 0:
        return 0.0

    return float(
        new_matched / total_constraints
    )


# ============================================================================
# REDUNDANCY SIGNALS
# ============================================================================


def temporal_similarity(
    candidate: CandidateMetadata,
    selected: CandidateMetadata,
    threshold_sec: float,
) -> float:
    """
    Temporal proximity trong [0, 1].

        khác video                     -> 0
        cùng video, >= threshold       -> 0
        cùng timestamp                 -> 1
    """

    if candidate.video_id != selected.video_id:
        return 0.0

    threshold = float(threshold_sec)

    if threshold <= 0.0:
        return (
            1.0
            if candidate.timestamp_sec == selected.timestamp_sec
            else 0.0
        )

    dt = abs(
        float(candidate.timestamp_sec)
        - float(selected.timestamp_sec)
    )

    if dt >= threshold:
        return 0.0

    return float(
        np.clip(
            1.0 - (dt / threshold),
            0.0,
            1.0,
        )
    )


def semantic_similarity(
    candidate: CandidateMetadata,
    selected: CandidateMetadata,
) -> float:
    """
    Jaccard similarity trên semantic labels.
    """

    candidate_labels = (
        set(candidate.entities)
        | set(candidate.actions)
        | set(candidate.attributes)
        | set(candidate.relations)
    )

    selected_labels = (
        set(selected.entities)
        | set(selected.actions)
        | set(selected.attributes)
        | set(selected.relations)
    )

    union = candidate_labels | selected_labels

    if not union:
        return 0.0

    intersection = (
        candidate_labels
        .intersection(selected_labels)
    )

    return float(
        len(intersection) / len(union)
    )


def visual_similarity(
    candidate: CandidateMetadata,
    selected: CandidateMetadata,
) -> float:
    """
    Visual similarity cho feature embedding.

    Cosine similarity clip về [0, 1].
    """

    if candidate.feature_vector is None:
        return 0.0

    if selected.feature_vector is None:
        return 0.0

    similarity = cosine_similarity(
        candidate.feature_vector,
        selected.feature_vector,
    )

    return float(
        np.clip(
            similarity,
            0.0,
            1.0,
        )
    )


def pair_redundancy(
    candidate: CandidateMetadata,
    selected: CandidateMetadata,
    *,
    time_threshold_sec: float,
    temporal_weight: float,
    visual_weight: float,
    semantic_weight: float,
) -> float:
    """
    Redundancy giữa một pair candidate-selected.

    Combine:

        temporal
        visual
        semantic

    Returns [0, 1].
    """

    weights = np.asarray(
        [
            max(0.0, float(temporal_weight)),
            max(0.0, float(visual_weight)),
            max(0.0, float(semantic_weight)),
        ],
        dtype=np.float64,
    )

    total = float(weights.sum())

    if total <= 1e-12:
        weights = np.asarray(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        )
        total = 1.0

    weights /= total

    temporal = temporal_similarity(
        candidate,
        selected,
        time_threshold_sec,
    )

    visual = visual_similarity(
        candidate,
        selected,
    )

    semantic = semantic_similarity(
        candidate,
        selected,
    )

    value = (
        float(weights[0]) * temporal
        + float(weights[1]) * visual
        + float(weights[2]) * semantic
    )

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


# ============================================================================
# FUSION ENGINE
# ============================================================================


class MultiModalFusionEngine:
    """
    QA-R3 Semantic Coverage + Diversity Selector.

    Input relevance phải đến từ upstream R1/R2.

    Performance:
        - semantic coverage dùng uint16/uint32/uint64 packed bitmask
        - marginal coverage dùng vectorized popcount
        - redundancy được vectorize theo candidate pool và cache
        - selection chỉ dùng NumPy arrays

    Scoring:

        SAGE(c)
            = relevance(c)
            + sage_lambda * coverage(c)

        GREEDY gain(c)
            = SAGE(c)
            + marginal_coverage_weight * marginal_gain(c)
            - redundancy_penalty_weight * max_redundancy(c)

        MMR(c)
            = lambda_mmr * SAGE(c)
            - (1 - lambda_mmr) * max_redundancy(c)
    """

    def __init__(
        self,
        *,
        sage_lambda: float = 0.3,
        marginal_coverage_weight: float = 0.3,
        sage_eta: float = 0.0,
        time_threshold_sec: float = 2.0,
        lambda_mmr: float = 0.7,
        temporal_redundancy_weight: float = 0.5,
        visual_redundancy_weight: float = 0.3,
        semantic_redundancy_weight: float = 0.2,
        redundancy_penalty_weight: float = 0.10,
    ) -> None:

        if sage_lambda < 0.0:
            raise ValueError(
                "sage_lambda phải >= 0"
            )

        if marginal_coverage_weight < 0.0:
            raise ValueError(
                "marginal_coverage_weight phải >= 0"
            )

        if sage_eta < 0.0:
            raise ValueError(
                "sage_eta phải >= 0"
            )

        if time_threshold_sec < 0.0:
            raise ValueError(
                "time_threshold_sec phải >= 0"
            )

        if not 0.0 <= lambda_mmr <= 1.0:
            raise ValueError(
                "lambda_mmr phải nằm trong [0, 1]"
            )

        if redundancy_penalty_weight < 0.0:
            raise ValueError(
                "redundancy_penalty_weight phải >= 0"
            )

        self.sage_lambda = float(
            sage_lambda
        )

        self.marginal_coverage_weight = float(
            marginal_coverage_weight
        )

        self.sage_eta = float(
            sage_eta
        )

        self.time_threshold_sec = float(
            time_threshold_sec
        )

        self.lambda_mmr = float(
            lambda_mmr
        )

        self.temporal_redundancy_weight = float(
            temporal_redundancy_weight
        )

        self.visual_redundancy_weight = float(
            visual_redundancy_weight
        )

        self.semantic_redundancy_weight = float(
            semantic_redundancy_weight
        )

        self.redundancy_penalty_weight = float(
            redundancy_penalty_weight
        )

        # ------------------------------------------------------------------
        # Redundancy cache.
        #
        # CandidateMetadata objects are expected to remain immutable while
        # this cache is valid. Object identity is used to invalidate the
        # cache when caller replaces metadata objects.
        # ------------------------------------------------------------------

        self._redundancy_cache_key: Optional[
            Tuple[object, ...]
        ] = None

        self._redundancy_matrix: Optional[
            np.ndarray
        ] = None

    # ----------------------------------------------------------------------
    # BASE SAGE — PUBLIC COMPATIBILITY API
    # ----------------------------------------------------------------------

    def compute_sage_scores(
        self,
        fused_scores: Mapping[str, float],
        metadata_map: Mapping[str, CandidateMetadata],
        query: SemanticQuery,
    ) -> Dict[
        str,
        Tuple[
            float,
            float,
            CoverageBreakdown,
        ],
    ]:
        """
        Public compatibility API.

        Base SAGE:

            relevance
            + sage_lambda * semantic coverage

        sage_eta giữ compatibility/tuning sau này.
        Temporal redundancy được xử lý ở selection layer.
        """

        result: Dict[
            str,
            Tuple[
                float,
                float,
                CoverageBreakdown,
            ],
        ] = {}

        total_entity = len(query.entities)
        total_action = len(query.actions)
        total_attribute = len(query.attributes)
        total_relation = len(query.relations)

        missing_breakdown = CoverageBreakdown(
            entity=0.0,
            action=0.0,
            attribute=0.0,
            relation=0.0,
            matched_entity=0,
            matched_action=0,
            matched_attribute=0,
            matched_relation=0,
            total_entity=total_entity,
            total_action=total_action,
            total_attribute=total_attribute,
            total_relation=total_relation,
        )

        for cid, raw_score in fused_scores.items():

            relevance = _normalize_score(
                raw_score
            )

            meta = metadata_map.get(cid)

            if meta is None:
                breakdown = missing_breakdown
                coverage = 0.0

            else:
                breakdown = compute_semantic_coverage(
                    query,
                    meta,
                )

                coverage = breakdown.overall

            temporal_bonus = 0.0

            sage = (
                relevance
                + self.sage_lambda * coverage
                + self.sage_eta * temporal_bonus
            )

            result[cid] = (
                float(sage),
                float(coverage),
                breakdown,
            )

        return result

    # ----------------------------------------------------------------------
    # FAST SEMANTIC MASK PREPARATION
    # ----------------------------------------------------------------------

    @staticmethod
    def _build_label_index(
        labels: Set[str],
    ) -> Dict[str, int]:
        """
        Stable label -> bit index.

        Mỗi semantic group có tối đa 16 labels để encode bằng uint16
        group mask.

        Nếu vượt quá 16 labels, raise rõ ràng.
        """

        ordered = sorted(labels)

        if len(ordered) > _MAX_MASK_BITS:
            raise ValueError(
                "Mỗi semantic query group chỉ hỗ trợ tối đa "
                f"{_MAX_MASK_BITS} labels trong optimized R3 "
                f"(got {len(ordered)})."
            )

        return {
            label: idx
            for idx, label in enumerate(ordered)
        }

    @staticmethod
    def _labels_to_mask(
        labels: Set[str],
        label_index: Mapping[str, int],
    ) -> np.uint16:
        """
        Convert candidate labels thành group uint16 mask.

        Chỉ query labels được encode.
        """

        mask = 0

        for label in labels:
            bit = label_index.get(label)

            if bit is not None:
                mask |= 1 << bit

        return np.uint16(mask)

    @staticmethod
    def _packed_mask_dtype(
        total_constraints: int,
    ) -> np.dtype:
        """
        Chọn dtype cho packed semantic mask.

        Mỗi group vẫn có tối đa 16 labels, nên tổng tối đa là 64.

            <= 16 -> uint16
            <= 32 -> uint32
            <= 64 -> uint64
        """

        if total_constraints <= 16:
            return np.dtype(np.uint16)

        if total_constraints <= 32:
            return np.dtype(np.uint32)

        if total_constraints <= _MAX_PACKED_BITS:
            return np.dtype(np.uint64)

        raise ValueError(
            "Tổng số semantic constraints tối đa là "
            f"{_MAX_PACKED_BITS}; got {total_constraints}."
        )

    @classmethod
    def _build_semantic_masks(
        cls,
        candidate_ids: Sequence[str],
        metadata_map: Mapping[str, CandidateMetadata],
        query: SemanticQuery,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Build four candidate semantic bitmask arrays.

        Returns:

            entity_masks
            action_masks
            attribute_masks
            relation_masks

        Shape:

            (N,)

        dtype:

            uint16
        """

        entity_index = cls._build_label_index(
            query.entities
        )

        action_index = cls._build_label_index(
            query.actions
        )

        attribute_index = cls._build_label_index(
            query.attributes
        )

        relation_index = cls._build_label_index(
            query.relations
        )

        n = len(candidate_ids)

        entity_masks = np.zeros(
            n,
            dtype=np.uint16,
        )

        action_masks = np.zeros(
            n,
            dtype=np.uint16,
        )

        attribute_masks = np.zeros(
            n,
            dtype=np.uint16,
        )

        relation_masks = np.zeros(
            n,
            dtype=np.uint16,
        )

        for idx, cid in enumerate(candidate_ids):

            meta = metadata_map.get(cid)

            if meta is None:
                continue

            entity_masks[idx] = cls._labels_to_mask(
                meta.entities,
                entity_index,
            )

            action_masks[idx] = cls._labels_to_mask(
                meta.actions,
                action_index,
            )

            attribute_masks[idx] = cls._labels_to_mask(
                meta.attributes,
                attribute_index,
            )

            relation_masks[idx] = cls._labels_to_mask(
                meta.relations,
                relation_index,
            )

        return (
            entity_masks,
            action_masks,
            attribute_masks,
            relation_masks,
        )

    @classmethod
    def _build_packed_semantic_masks(
        cls,
        *,
        entity_masks: np.ndarray,
        action_masks: np.ndarray,
        attribute_masks: np.ndarray,
        relation_masks: np.ndarray,
        query: SemanticQuery,
    ) -> np.ndarray:
        """
        Pack four semantic group masks into one integer mask.

        Layout:

            entity    -> bits [0, 15]
            action    -> bits [16, 31]
            attribute -> bits [32, 47]
            relation  -> bits [48, 63]

        Chỉ dùng số bit thực tế cần thiết theo query, nhưng offset của
        từng group vẫn cố định 16 bit để giữ representation đơn giản
        và deterministic.

        Với benchmark hiện tại (12 total constraints), dtype là uint16
        vì tất cả semantic constraints nằm trong 16 bit đầu.

        Lưu ý:
            Mặc dù action/attribute/relation được đặt ở các offset group,
            nếu total constraints <= 16 thì query layout vẫn cần tổng
            số bit <= 16. Vì vậy trường hợp benchmark 3+5+2+2 = 12
            vẫn hoàn toàn nằm trong uint16.
        """

        total_constraints = (
            len(query.entities)
            + len(query.actions)
            + len(query.attributes)
            + len(query.relations)
        )

        dtype = cls._packed_mask_dtype(
            total_constraints
        )

        n = entity_masks.shape[0]

        if total_constraints <= 16:
            # Compact layout: groups được pack sát nhau theo số label
            # thực tế, thay vì cố định offset 16 bit.
            entity_bits = len(query.entities)
            action_bits = len(query.actions)
            attribute_bits = len(query.attributes)

            packed = np.asarray(
                entity_masks,
                dtype=dtype,
            )

            if action_bits > 0:
                packed = (
                    packed
                    | (
                        np.asarray(
                            action_masks,
                            dtype=dtype,
                        )
                        << np.uint16(entity_bits)
                    )
                )

            action_offset = entity_bits + action_bits

            if attribute_bits > 0:
                packed = (
                    packed
                    | (
                        np.asarray(
                            attribute_masks,
                            dtype=dtype,
                        )
                        << np.uint16(action_offset)
                    )
                )

            relation_offset = (
                entity_bits
                + action_bits
                + attribute_bits
            )

            if len(query.relations) > 0:
                packed = (
                    packed
                    | (
                        np.asarray(
                            relation_masks,
                            dtype=dtype,
                        )
                        << np.uint16(relation_offset)
                    )
                )

            return packed.astype(
                dtype,
                copy=False,
            )

        # ------------------------------------------------------------------
        # >16 constraints:
        #
        # Use fixed 16-bit slots. This preserves every group independently
        # and fits exactly into uint32 / uint64.
        # ------------------------------------------------------------------

        packed = np.asarray(
            entity_masks,
            dtype=dtype,
        )

        packed = (
            packed
            | (
                np.asarray(
                    action_masks,
                    dtype=dtype,
                )
                << 16
            )
        )

        packed = (
            packed
            | (
                np.asarray(
                    attribute_masks,
                    dtype=dtype,
                )
                << 32
            )
        )

        if dtype == np.dtype(np.uint64):
            packed = (
                packed
                | (
                    np.asarray(
                        relation_masks,
                        dtype=dtype,
                    )
                    << 48
                )
            )

        else:
            # This branch can only happen for uint32, where total
            # constraints are <=32. Relation labels cannot all fit after
            # the first 32 bits if fixed 16-bit slots are used.
            #
            # Therefore for uint32 we use compact packing instead.
            entity_bits = len(query.entities)
            action_bits = len(query.actions)
            attribute_bits = len(query.attributes)

            packed = np.asarray(
                entity_masks,
                dtype=dtype,
            )

            if action_bits > 0:
                packed |= (
                    np.asarray(
                        action_masks,
                        dtype=dtype,
                    )
                    << np.uint32(entity_bits)
                )

            action_offset = (
                entity_bits
                + action_bits
            )

            if attribute_bits > 0:
                packed |= (
                    np.asarray(
                        attribute_masks,
                        dtype=dtype,
                    )
                    << np.uint32(action_offset)
                )

            relation_offset = (
                entity_bits
                + action_bits
                + attribute_bits
            )

            if len(query.relations) > 0:
                packed |= (
                    np.asarray(
                        relation_masks,
                        dtype=dtype,
                    )
                    << np.uint32(relation_offset)
                )

        return packed.astype(
            dtype,
            copy=False,
        )

    @staticmethod
    def _mask_bit_count(
        masks: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized popcount for uint16.

        Không dùng np.unpackbits vì LUT uint16 nhỏ hơn và tránh tạo
        temporary matrix lớn.

        Input:
            uint16 array

        Output:
            uint8 array
        """

        return _POPCOUNT_LUT[
            masks.astype(
                np.uint16,
                copy=False,
            )
        ]

    @classmethod
    def _packed_popcount(
        cls,
        masks: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized popcount cho packed uint16 / uint32 / uint64.

        uint16:
            direct LUT lookup.

        uint32:
            2 x uint16 chunks.

        uint64:
            4 x uint16 chunks.

        Output:
            uint16 array

        Không dùng Python loop theo candidate.
        """

        dtype = masks.dtype

        if dtype == np.dtype(np.uint16):
            return _POPCOUNT_LUT[
                masks
            ].astype(
                np.uint16,
                copy=False,
            )

        if dtype == np.dtype(np.uint32):
            chunks = masks.view(
                np.uint16
            ).reshape(
                -1,
                2,
            )

        elif dtype == np.dtype(np.uint64):
            chunks = masks.view(
                np.uint16
            ).reshape(
                -1,
                4,
            )

        else:
            raise TypeError(
                "Packed semantic mask phải có dtype "
                "uint16, uint32 hoặc uint64."
            )

        return _POPCOUNT_LUT[
            chunks
        ].sum(
            axis=1,
            dtype=np.uint16,
        )

    @classmethod
    def _compute_fast_semantic_arrays(
        cls,
        *,
        entity_masks: np.ndarray,
        action_masks: np.ndarray,
        attribute_masks: np.ndarray,
        relation_masks: np.ndarray,
        query: SemanticQuery,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Tính:

            matched_entity
            matched_action
            matched_attribute
            matched_relation

        bằng bitmask + LUT.

        Không có Python per-candidate semantic intersection.
        """

        matched_entity = cls._mask_bit_count(
            entity_masks
        )

        matched_action = cls._mask_bit_count(
            action_masks
        )

        matched_attribute = cls._mask_bit_count(
            attribute_masks
        )

        matched_relation = cls._mask_bit_count(
            relation_masks
        )

        return (
            matched_entity,
            matched_action,
            matched_attribute,
            matched_relation,
        )

    @classmethod
    def _compute_fast_sage_arrays(
        cls,
        *,
        candidate_ids: Sequence[str],
        fused_scores: Mapping[str, float],
        metadata_map: Mapping[str, CandidateMetadata],
        query: SemanticQuery,
        sage_lambda: float,
        sage_eta: float,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        np.ndarray,
    ]:
        """
        Fast path cho select_top_k().

        Trả về:

            sage
            fused
            coverage
            matched_entity
            matched_action
            matched_attribute
            matched_relation
            semantic masks
            packed semantic mask

        Semantic coverage được tính bằng NumPy sau bước mask
        construction.

        packed semantic mask được thêm riêng cho Greedy hot path.
        """

        (
            entity_masks,
            action_masks,
            attribute_masks,
            relation_masks,
        ) = cls._build_semantic_masks(
            candidate_ids,
            metadata_map,
            query,
        )

        (
            matched_entity,
            matched_action,
            matched_attribute,
            matched_relation,
        ) = cls._compute_fast_semantic_arrays(
            entity_masks=entity_masks,
            action_masks=action_masks,
            attribute_masks=attribute_masks,
            relation_masks=relation_masks,
            query=query,
        )

        packed_masks = cls._build_packed_semantic_masks(
            entity_masks=entity_masks,
            action_masks=action_masks,
            attribute_masks=attribute_masks,
            relation_masks=relation_masks,
            query=query,
        )

        total_constraints = (
            len(query.entities)
            + len(query.actions)
            + len(query.attributes)
            + len(query.relations)
        )

        n = len(candidate_ids)

        fused = np.empty(
            n,
            dtype=np.float32,
        )

        for idx, cid in enumerate(candidate_ids):
            fused[idx] = _normalize_score(
                fused_scores.get(
                    cid,
                    0.0,
                )
            )

        matched_total = (
            matched_entity.astype(np.int16)
            + matched_action.astype(np.int16)
            + matched_attribute.astype(np.int16)
            + matched_relation.astype(np.int16)
        )

        if total_constraints > 0:
            coverage = (
                matched_total.astype(
                    np.float32
                )
                / float(total_constraints)
            )
        else:
            coverage = np.zeros(
                n,
                dtype=np.float32,
            )

        sage = (
            fused
            + float(sage_lambda) * coverage
        )

        if sage_eta != 0.0:
            # Compatibility only.
            sage = (
                sage
                + float(sage_eta) * 0.0
            )

        return (
            sage.astype(
                np.float32,
                copy=False,
            ),
            fused,
            coverage.astype(
                np.float32,
                copy=False,
            ),
            matched_entity,
            matched_action,
            matched_attribute,
            matched_relation,
            (
                entity_masks,
                action_masks,
                attribute_masks,
                relation_masks,
            ),
            packed_masks,
        )

    # ----------------------------------------------------------------------
    # VECTOR BUILDERS
    # ----------------------------------------------------------------------

    @staticmethod
    def _build_temporal_matrix(
        metadata: Sequence[Optional[CandidateMetadata]],
        threshold_sec: float,
    ) -> np.ndarray:
        """
        Vectorized temporal redundancy matrix.

        Chỉ cùng video mới có temporal similarity.
        """

        n = len(metadata)

        result = np.zeros(
            (n, n),
            dtype=np.float32,
        )

        threshold = float(threshold_sec)

        if n == 0 or threshold <= 0.0:
            return result

        groups: Dict[str, List[int]] = {}

        for idx, meta in enumerate(metadata):
            if meta is None:
                continue

            groups.setdefault(
                meta.video_id,
                [],
            ).append(idx)

        for indices in groups.values():

            if len(indices) <= 1:
                continue

            idx = np.asarray(
                indices,
                dtype=np.int32,
            )

            times = np.asarray(
                [
                    float(metadata[i].timestamp_sec)
                    for i in indices
                ],
                dtype=np.float32,
            )

            dt = np.abs(
                times[:, None]
                - times[None, :]
            )

            similarity = (
                1.0
                - dt / threshold
            )

            similarity = np.clip(
                similarity,
                0.0,
                1.0,
            ).astype(
                np.float32,
                copy=False,
            )

            result[
                np.ix_(idx, idx)
            ] = similarity

        np.fill_diagonal(
            result,
            0.0,
        )

        return result

    @staticmethod
    def _build_semantic_matrix(
        metadata: Sequence[Optional[CandidateMetadata]],
    ) -> np.ndarray:
        """
        Vectorized semantic Jaccard matrix.

        Semantic labels của candidate được gom thành một vocabulary.
        Ma trận membership sau đó dùng matrix multiplication để tính
        pairwise intersection.

        Complexity:
            O(N^2 * L) trong native NumPy/BLAS,

        thay vì:
            O(N^2) Python set construction/function calls.
        """

        n = len(metadata)

        result = np.zeros(
            (n, n),
            dtype=np.float32,
        )

        if n == 0:
            return result

        label_sets: List[Set[str]] = []

        vocabulary: Set[str] = set()

        for meta in metadata:

            if meta is None:
                labels: Set[str] = set()

            else:
                labels = (
                    set(meta.entities)
                    | set(meta.actions)
                    | set(meta.attributes)
                    | set(meta.relations)
                )

            label_sets.append(labels)
            vocabulary.update(labels)

        if not vocabulary:
            return result

        labels_sorted = sorted(
            vocabulary
        )

        label_to_index = {
            label: idx
            for idx, label in enumerate(labels_sorted)
        }

        membership = np.zeros(
            (n, len(labels_sorted)),
            dtype=np.float32,
        )

        for row, labels in enumerate(label_sets):

            for label in labels:
                membership[
                    row,
                    label_to_index[label]
                ] = 1.0

        intersection = (
            membership
            @ membership.T
        )

        counts = membership.sum(
            axis=1,
            dtype=np.float32,
        )

        union = (
            counts[:, None]
            + counts[None, :]
            - intersection
        )

        np.divide(
            intersection,
            union,
            out=result,
            where=union > 0.0,
        )

        np.fill_diagonal(
            result,
            0.0,
        )

        return result.astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _build_visual_matrix(
        metadata: Sequence[Optional[CandidateMetadata]],
    ) -> np.ndarray:
        """
        Vectorized visual cosine similarity.

        Contract:
            Tất cả feature_vector hợp lệ trong candidate pool phải có
            cùng dimension.

        Nếu feature dimensions không đồng nhất, raise ValueError thay vì
        silently disable toàn bộ visual redundancy.

        Candidate không có feature_vector vẫn được phép và tạo zero row.
        """

        n = len(metadata)

        result = np.zeros(
            (n, n),
            dtype=np.float32,
        )

        if n == 0:
            return result

        dimension: Optional[int] = None

        for idx, meta in enumerate(metadata):

            if meta is None:
                continue

            feature = meta.feature_vector

            if feature is None:
                continue

            arr = np.asarray(
                feature,
                dtype=np.float32,
            )

            if arr.ndim != 1:
                raise ValueError(
                    "feature_vector của candidate "
                    f"index={idx} phải là 1-D."
                )

            if arr.size == 0:
                raise ValueError(
                    "feature_vector của candidate "
                    f"index={idx} không được rỗng."
                )

            if not np.all(
                np.isfinite(arr)
            ):
                raise ValueError(
                    "feature_vector của candidate "
                    f"index={idx} chứa NaN/Inf."
                )

            current_dimension = int(
                arr.size
            )

            if dimension is None:
                dimension = current_dimension

            elif current_dimension != dimension:
                raise ValueError(
                    "Visual feature dimensions không đồng nhất "
                    f"trong candidate pool: expected {dimension}, "
                    f"got {current_dimension} tại index={idx}."
                )

        if dimension is None:
            return result

        features = np.zeros(
            (n, dimension),
            dtype=np.float32,
        )

        valid = np.zeros(
            n,
            dtype=bool,
        )

        for idx, meta in enumerate(metadata):

            if meta is None:
                continue

            feature = meta.feature_vector

            if feature is None:
                continue

            arr = np.asarray(
                feature,
                dtype=np.float32,
            )

            norm = float(
                np.linalg.norm(arr)
            )

            if norm <= 1e-8:
                continue

            features[idx] = (
                arr / norm
            )

            valid[idx] = True

        if not np.any(valid):
            return result

        result = (
            features
            @ features.T
        )

        np.clip(
            result,
            0.0,
            1.0,
            out=result,
        )

        invalid = ~valid

        if np.any(invalid):
            result[invalid, :] = 0.0
            result[:, invalid] = 0.0

        np.fill_diagonal(
            result,
            0.0,
        )

        return result.astype(
            np.float32,
            copy=False,
        )

    # ----------------------------------------------------------------------
    # REDUNDANCY MATRIX
    # ----------------------------------------------------------------------

    def _redundancy_cache_signature(
        self,
        candidate_ids: Sequence[str],
        metadata_map: Mapping[str, CandidateMetadata],
    ) -> Tuple[object, ...]:
        """
        Signature cho redundancy cache.

        Candidate metadata object identity được đưa vào signature để
        tránh reuse matrix nếu caller thay metadata objects.

        Pool order cũng được giữ nguyên.

        Contract:
            Nếu mutate nội dung bên trong cùng một CandidateMetadata object,
            cache không tự phát hiện được. Caller phải coi metadata là
            immutable trong thời gian cache còn hiệu lực.
        """

        ids = tuple(candidate_ids)

        metadata_identity = tuple(
            id(metadata_map.get(cid))
            for cid in ids
        )

        return (
            ids,
            metadata_identity,
            float(self.time_threshold_sec),
            float(self.temporal_redundancy_weight),
            float(self.visual_redundancy_weight),
            float(self.semantic_redundancy_weight),
        )

    def _build_redundancy_matrix(
        self,
        candidate_ids: Sequence[str],
        metadata_map: Mapping[str, CandidateMetadata],
    ) -> np.ndarray:
        """
        Build pairwise redundancy matrix.

        Đây là phần nặng nhất nhưng chỉ chạy khi candidate pool thay đổi
        hoặc configuration redundancy thay đổi.
        """

        metadata: List[
            Optional[CandidateMetadata]
        ] = [
            metadata_map.get(cid)
            for cid in candidate_ids
        ]

        temporal = self._build_temporal_matrix(
            metadata,
            self.time_threshold_sec,
        )

        visual = self._build_visual_matrix(
            metadata,
        )

        semantic = self._build_semantic_matrix(
            metadata,
        )

        weights = np.asarray(
            [
                max(
                    0.0,
                    float(
                        self.temporal_redundancy_weight
                    ),
                ),
                max(
                    0.0,
                    float(
                        self.visual_redundancy_weight
                    ),
                ),
                max(
                    0.0,
                    float(
                        self.semantic_redundancy_weight
                    ),
                ),
            ],
            dtype=np.float32,
        )

        total = float(
            weights.sum()
        )

        if total <= 1e-12:
            weights[:] = 0.0
            weights[0] = 1.0
            total = 1.0

        weights /= total

        result = (
            weights[0] * temporal
            + weights[1] * visual
            + weights[2] * semantic
        )

        np.clip(
            result,
            0.0,
            1.0,
            out=result,
        )

        np.fill_diagonal(
            result,
            0.0,
        )

        return result.astype(
            np.float32,
            copy=False,
        )

    def _get_redundancy_matrix(
        self,
        candidate_ids: Sequence[str],
        metadata_map: Mapping[str, CandidateMetadata],
    ) -> np.ndarray:
        """
        Get cached redundancy matrix.
        """

        signature = (
            self._redundancy_cache_signature(
                candidate_ids,
                metadata_map,
            )
        )

        if (
            self._redundancy_cache_key == signature
            and self._redundancy_matrix is not None
        ):
            return self._redundancy_matrix

        matrix = self._build_redundancy_matrix(
            candidate_ids,
            metadata_map,
        )

        self._redundancy_cache_key = signature
        self._redundancy_matrix = matrix

        return matrix

    # ----------------------------------------------------------------------
    # COMPATIBILITY REDUNDANCY CACHE UPDATE
    # ----------------------------------------------------------------------

    def _update_redundancy_cache(
        self,
        *,
        selected_id: str,
        remaining_ids: Set[str],
        metadata_map: Mapping[str, CandidateMetadata],
        max_redundancy: Dict[str, float],
    ) -> None:
        """
        Compatibility implementation.

        Selector mới KHÔNG dùng function này.

        Giữ lại để tránh break caller cũ.
        """

        selected_meta = metadata_map.get(
            selected_id
        )

        if selected_meta is None:
            return

        for cid in remaining_ids:

            candidate_meta = metadata_map.get(
                cid
            )

            if candidate_meta is None:
                continue

            similarity = pair_redundancy(
                candidate_meta,
                selected_meta,
                time_threshold_sec=self.time_threshold_sec,
                temporal_weight=self.temporal_redundancy_weight,
                visual_weight=self.visual_redundancy_weight,
                semantic_weight=self.semantic_redundancy_weight,
            )

            if similarity > max_redundancy.get(
                cid,
                0.0,
            ):
                max_redundancy[cid] = similarity

    # ----------------------------------------------------------------------
    # LEGACY MARGINAL COVERAGE ARRAYS
    # ----------------------------------------------------------------------

    @staticmethod
    def _initial_candidate_labels(
        candidate_ids: Sequence[str],
        metadata_map: Mapping[str, CandidateMetadata],
        query: SemanticQuery,
    ) -> Tuple[
        List[Set[str]],
        List[Set[str]],
        List[Set[str]],
        List[Set[str]],
    ]:
        """
        Compatibility helper.

        Không được dùng trong hot selection loop.
        """

        query_entities = query.entities
        query_actions = query.actions
        query_attributes = query.attributes
        query_relations = query.relations

        entity_labels: List[Set[str]] = []
        action_labels: List[Set[str]] = []
        attribute_labels: List[Set[str]] = []
        relation_labels: List[Set[str]] = []

        for cid in candidate_ids:

            meta = metadata_map.get(cid)

            if meta is None:
                entity_labels.append(set())
                action_labels.append(set())
                attribute_labels.append(set())
                relation_labels.append(set())
                continue

            entity_labels.append(
                query_entities.intersection(
                    meta.entities
                )
            )

            action_labels.append(
                query_actions.intersection(
                    meta.actions
                )
            )

            attribute_labels.append(
                query_attributes.intersection(
                    meta.attributes
                )
            )

            relation_labels.append(
                query_relations.intersection(
                    meta.relations
                )
            )

        return (
            entity_labels,
            action_labels,
            attribute_labels,
            relation_labels,
        )

    @staticmethod
    def _marginal_gain_array(
        *,
        entity_labels: Sequence[Set[str]],
        action_labels: Sequence[Set[str]],
        attribute_labels: Sequence[Set[str]],
        relation_labels: Sequence[Set[str]],
        covered_entities: Set[str],
        covered_actions: Set[str],
        covered_attributes: Set[str],
        covered_relations: Set[str],
        total_constraints: int,
    ) -> np.ndarray:
        """
        Legacy compatibility implementation.

        Selector mới KHÔNG dùng function này.
        """

        n = len(entity_labels)

        result = np.zeros(
            n,
            dtype=np.float32,
        )

        if total_constraints <= 0:
            return result

        inv_total = 1.0 / float(
            total_constraints
        )

        for idx in range(n):

            new_count = 0

            if entity_labels[idx]:
                new_count += len(
                    entity_labels[idx]
                    - covered_entities
                )

            if action_labels[idx]:
                new_count += len(
                    action_labels[idx]
                    - covered_actions
                )

            if attribute_labels[idx]:
                new_count += len(
                    attribute_labels[idx]
                    - covered_attributes
                )

            if relation_labels[idx]:
                new_count += len(
                    relation_labels[idx]
                    - covered_relations
                )

            result[idx] = (
                float(new_count)
                * inv_total
            )

        return result

    # ----------------------------------------------------------------------
    # FAST MARGINAL COVERAGE
    # ----------------------------------------------------------------------

    @staticmethod
    def _fast_marginal_gain_array(
        *,
        entity_masks: np.ndarray,
        action_masks: np.ndarray,
        attribute_masks: np.ndarray,
        relation_masks: np.ndarray,
        covered_entity_mask: int,
        covered_action_mask: int,
        covered_attribute_mask: int,
        covered_relation_mask: int,
        total_constraints: int,
    ) -> np.ndarray:
        """
        Compatibility optimized implementation.

        Fully vectorized marginal semantic gain using four group masks.

        Selector GREEDY hiện tại dùng _fast_packed_marginal_gain_array()
        để giảm số temporary arrays trong K*N hot path.
        """

        n = entity_masks.shape[0]

        if total_constraints <= 0:
            return np.zeros(
                n,
                dtype=np.float32,
            )

        entity_new = (
            entity_masks
            & np.uint16(
                (~covered_entity_mask)
                & 0xFFFF
            )
        )

        action_new = (
            action_masks
            & np.uint16(
                (~covered_action_mask)
                & 0xFFFF
            )
        )

        attribute_new = (
            attribute_masks
            & np.uint16(
                (~covered_attribute_mask)
                & 0xFFFF
            )
        )

        relation_new = (
            relation_masks
            & np.uint16(
                (~covered_relation_mask)
                & 0xFFFF
            )
        )

        new_count = (
            _POPCOUNT_LUT[entity_new].astype(
                np.uint16,
                copy=False,
            )
            + _POPCOUNT_LUT[action_new].astype(
                np.uint16,
                copy=False,
            )
            + _POPCOUNT_LUT[attribute_new].astype(
                np.uint16,
                copy=False,
            )
            + _POPCOUNT_LUT[relation_new].astype(
                np.uint16,
                copy=False,
            )
        )

        return (
            new_count.astype(
                np.float32,
                copy=False,
            )
            / float(total_constraints)
        )

    @classmethod
    def _fast_packed_marginal_gain_array(
        cls,
        *,
        packed_masks: np.ndarray,
        covered_mask: int,
        total_constraints: int,
    ) -> np.ndarray:
        """
        Hot-path marginal semantic gain.

        Thay vì xử lý 4 group độc lập:

            4 AND
            4 LUT lookup
            nhiều astype
            nhiều temporary arrays

        ta xử lý một packed mask:

            new_bits = candidate_mask & ~covered_mask
            popcount(new_bits)

        Với benchmark hiện tại total_constraints=12 nên packed_masks
        là uint16 và operation này chỉ cần một LUT lookup + một division.
        """

        n = packed_masks.shape[0]

        if total_constraints <= 0:
            return np.zeros(
                n,
                dtype=np.float32,
            )

        dtype = packed_masks.dtype

        if dtype == np.dtype(np.uint16):
            covered = np.uint16(
                covered_mask & 0xFFFF
            )

        elif dtype == np.dtype(np.uint32):
            covered = np.uint32(
                covered_mask & 0xFFFFFFFF
            )

        elif dtype == np.dtype(np.uint64):
            covered = np.uint64(
                covered_mask & 0xFFFFFFFFFFFFFFFF
            )

        else:
            raise TypeError(
                "Packed semantic mask phải có dtype "
                "uint16, uint32 hoặc uint64."
            )

        new_bits = (
            packed_masks
            & np.bitwise_not(
                covered
            )
        )

        new_count = cls._packed_popcount(
            new_bits
        )

        return (
            new_count.astype(
                np.float32,
                copy=False,
            )
            / float(total_constraints)
        )

    # ----------------------------------------------------------------------
    # COVERAGE BREAKDOWN
    # ----------------------------------------------------------------------

    @staticmethod
    def _make_coverage_breakdown(
        *,
        matched_entity: int,
        matched_action: int,
        matched_attribute: int,
        matched_relation: int,
        query: SemanticQuery,
    ) -> CoverageBreakdown:
        total_entity = len(query.entities)
        total_action = len(query.actions)
        total_attribute = len(query.attributes)
        total_relation = len(query.relations)

        return CoverageBreakdown(
            entity=(
                float(matched_entity / total_entity)
                if total_entity > 0
                else 0.0
            ),
            action=(
                float(matched_action / total_action)
                if total_action > 0
                else 0.0
            ),
            attribute=(
                float(matched_attribute / total_attribute)
                if total_attribute > 0
                else 0.0
            ),
            relation=(
                float(matched_relation / total_relation)
                if total_relation > 0
                else 0.0
            ),
            matched_entity=int(matched_entity),
            matched_action=int(matched_action),
            matched_attribute=int(matched_attribute),
            matched_relation=int(matched_relation),
            total_entity=total_entity,
            total_action=total_action,
            total_attribute=total_attribute,
            total_relation=total_relation,
        )

    # ----------------------------------------------------------------------
    # GREEDY
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # FAST SAGE-ONLY PATH
    # ----------------------------------------------------------------------

    @staticmethod
    def _compute_fast_sage_only(
        *,
        candidate_ids: Sequence[str],
        fused_scores: Mapping[str, float],
        metadata_map: Mapping[str, CandidateMetadata],
        query: SemanticQuery,
        sage_lambda: float,
        sage_eta: float,
    ) -> np.ndarray:
        """
        Fast path dành riêng cho MMR.

        MMR chỉ cần:

            SAGE(c)
                = relevance(c)
                + sage_lambda * semantic_coverage(c)

        Nó KHÔNG cần:
            - semantic bitmask
            - marginal coverage
            - packed mask
            - CoverageBreakdown cho toàn bộ candidate pool

        Vì vậy tránh toàn bộ overhead không cần thiết của
        _compute_fast_sage_arrays().

        Complexity:
            O(N)

        Returns:
            sage: float32 array shape (N,)
        """

        n = len(candidate_ids)

        if n == 0:
            return np.empty(
                0,
                dtype=np.float32,
            )

        sage = np.empty(
            n,
            dtype=np.float32,
        )

        sage_lambda = float(sage_lambda)

        # Query constraint counts.
        total_constraints = (
            len(query.entities)
            + len(query.actions)
            + len(query.attributes)
            + len(query.relations)
        )

        if total_constraints <= 0:
            # Empty semantic query:
            # SAGE == normalized relevance.
            for idx, cid in enumerate(candidate_ids):
                sage[idx] = _normalize_score(
                    fused_scores.get(cid, 0.0)
                )

            return sage

        inv_total = 1.0 / float(
            total_constraints
        )

        query_entities = query.entities
        query_actions = query.actions
        query_attributes = query.attributes
        query_relations = query.relations

        for idx, cid in enumerate(candidate_ids):

            relevance = _normalize_score(
                fused_scores.get(cid, 0.0)
            )

            meta = metadata_map.get(cid)

            if meta is None:
                coverage = 0.0

            else:
                matched = 0

                if query_entities:
                    matched += len(
                        query_entities.intersection(
                            meta.entities
                        )
                    )

                if query_actions:
                    matched += len(
                        query_actions.intersection(
                            meta.actions
                        )
                    )

                if query_attributes:
                    matched += len(
                        query_attributes.intersection(
                            meta.attributes
                        )
                    )

                if query_relations:
                    matched += len(
                        query_relations.intersection(
                            meta.relations
                        )
                    )

                coverage = float(
                    matched * inv_total
                )

            sage[idx] = (
                relevance
                + sage_lambda * coverage
            )

        # sage_eta is intentionally compatibility-only.
        if sage_eta != 0.0:
            sage += np.float32(
                float(sage_eta) * 0.0
            )

        return sage

    def _select_greedy(
        self,
        candidate_ids: Sequence[str],
        fused_scores: Mapping[str, float],
        metadata_map: Mapping[str, CandidateMetadata],
        query: SemanticQuery,
        top_k: int,
        *,
        precomputed: Optional[
            Tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                Tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                ],
                np.ndarray,
            ]
        ] = None,
    ) -> List[Tuple[str, float]]:
        """
        Greedy semantic coverage + diversity.

        Optimized hot path:

            O(K*N)

        but all semantic work is native NumPy.

        Hot-path semantic optimization:

            packed_masks
                ↓
            one bitwise AND
                ↓
            one popcount
                ↓
            marginal gain

        Không còn:

            compute_marginal_semantic_coverage()
            Python set difference per candidate
            pair_redundancy()
            sorted(remaining)
            4 independent semantic marginal arrays
        """

        n = len(candidate_ids)

        if top_k <= 0 or n == 0:
            return []

        k = min(
            int(top_k),
            n,
        )

        # ------------------------------------------------------------------
        # Fast static arrays.
        # ------------------------------------------------------------------

        if precomputed is None:

            (
                sage,
                fused,
                coverage,
                matched_entity,
                matched_action,
                matched_attribute,
                matched_relation,
                masks,
                packed_masks,
            ) = self._compute_fast_sage_arrays(
                candidate_ids=candidate_ids,
                fused_scores=fused_scores,
                metadata_map=metadata_map,
                query=query,
                sage_lambda=self.sage_lambda,
                sage_eta=self.sage_eta,
            )

        else:

            (
                sage,
                fused,
                coverage,
                matched_entity,
                matched_action,
                matched_attribute,
                matched_relation,
                masks,
                packed_masks,
            ) = precomputed

        redundancy_matrix = (
            self._get_redundancy_matrix(
                candidate_ids,
                metadata_map,
            )
        )

        active = np.ones(
            n,
            dtype=bool,
        )

        max_redundancy = np.zeros(
            n,
            dtype=np.float32,
        )

        # Packed semantic coverage state.
        covered_mask = 0

        total_constraints = (
            len(query.entities)
            + len(query.actions)
            + len(query.attributes)
            + len(query.relations)
        )

        selected: List[Tuple[str, float]] = []

        # list() once outside the hot loop.
        candidate_id_array = list(
            candidate_ids
        )

        penalty_weight = float(
            self.redundancy_penalty_weight
        )

        marginal_weight = float(
            self.marginal_coverage_weight
        )

        for _ in range(k):

            # --------------------------------------------------------------
            # Vectorized marginal coverage.
            #
            # This is the main optimization:
            #
            # OLD:
            #     4 masks -> 4 AND -> 4 LUT -> several temporaries
            #
            # NEW:
            #     packed mask -> 1 AND -> popcount
            # --------------------------------------------------------------

            marginal = self._fast_packed_marginal_gain_array(
                packed_masks=packed_masks,
                covered_mask=covered_mask,
                total_constraints=total_constraints,
            )

            # --------------------------------------------------------------
            # Gain.
            # --------------------------------------------------------------

            gain = (
                sage
                + marginal_weight * marginal
                - penalty_weight * max_redundancy
            )

            gain[~active] = -np.inf

            best_idx = int(
                np.argmax(gain)
            )

            best_value = float(
                gain[best_idx]
            )

            if not np.isfinite(best_value):
                break

            # --------------------------------------------------------------
            # Deterministic tie-break.
            #
            # Fast path:
            #   np.argmax already returns the first maximum.
            #
            # Only perform the more expensive tie logic if there is
            # actually more than one exact winner.
            # --------------------------------------------------------------

            tie_count = int(
                np.count_nonzero(
                    gain == best_value
                )
            )

            if tie_count > 1:

                tied = np.flatnonzero(
                    gain == best_value
                )

                tied_marginal = marginal[
                    tied
                ]

                best_marginal = float(
                    np.max(
                        tied_marginal
                    )
                )

                tied = tied[
                    tied_marginal
                    == best_marginal
                ]

                if tied.size > 1:

                    tied_fused = fused[
                        tied
                    ]

                    best_fused = float(
                        np.max(
                            tied_fused
                        )
                    )

                    tied = tied[
                        tied_fused
                        == best_fused
                    ]

                    if tied.size > 1:

                        best_idx = min(
                            (
                                int(idx)
                                for idx in tied
                            ),
                            key=lambda idx:
                                candidate_id_array[idx],
                        )

                    else:
                        best_idx = int(
                            tied[0]
                        )

                else:
                    best_idx = int(
                        tied[0]
                    )

            best_cid = candidate_id_array[
                best_idx
            ]

            selected.append(
                (
                    best_cid,
                    float(gain[best_idx]),
                )
            )

            active[best_idx] = False

            # --------------------------------------------------------------
            # Increment packed coverage mask.
            # --------------------------------------------------------------

            covered_mask |= int(
                packed_masks[best_idx]
            )

            # --------------------------------------------------------------
            # Increment max redundancy.
            #
            # Native NumPy vector operation.
            # --------------------------------------------------------------

            np.maximum(
                max_redundancy,
                redundancy_matrix[
                    best_idx
                ],
                out=max_redundancy,
            )

            max_redundancy[best_idx] = 0.0

        return selected

    # ----------------------------------------------------------------------
    # MMR
    # ----------------------------------------------------------------------

    def _select_mmr(
        self,
        candidate_ids: Sequence[str],
        sage: np.ndarray,
        metadata_map: Mapping[str, CandidateMetadata],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Optimized MMR selector.

        MMR:

            lambda * SAGE
            - (1 - lambda) * max_redundancy

        Important performance properties:

            - SAGE được tính một lần trước selection.
            - Không build semantic masks.
            - Không build marginal coverage arrays.
            - Không tạo CoverageBreakdown cho toàn bộ N.
            - Redundancy được cache.
            - max_redundancy được update incremental.
            - Không sorted(remaining).
            - Selection sử dụng NumPy argmax.

        Tie behavior:
            np.argmax giữ deterministic first-index behavior.
            Exact MMR ties được resolve theo candidate input order.

        Đây là behavior deterministic như candidate pool hiện tại.
        """

        n = len(candidate_ids)

        if top_k <= 0 or n == 0:
            return []

        k = min(
            int(top_k),
            n,
        )

        if sage.shape[0] != n:
            raise ValueError(
                "sage phải có cùng số phần tử với candidate_ids."
            )

        redundancy_matrix = (
            self._get_redundancy_matrix(
                candidate_ids,
                metadata_map,
            )
        )

        active = np.ones(
            n,
            dtype=bool,
        )

        max_redundancy = np.zeros(
            n,
            dtype=np.float32,
        )

        selected: List[Tuple[str, float]] = []

        candidate_id_array = list(
            candidate_ids
        )

        lambda_mmr = np.float32(
            self.lambda_mmr
        )

        one_minus_lambda = np.float32(
            1.0 - self.lambda_mmr
        )

        # --------------------------------------------------------------
        # Reuse one output buffer.
        #
        # Tránh tạo một ndarray mới cho mỗi vòng selection.
        # --------------------------------------------------------------

        mmr = np.empty(
            n,
            dtype=np.float32,
        )

        for _ in range(k):

            # ----------------------------------------------------------
            # MMR score.
            #
            # Reuse preallocated buffer.
            # ----------------------------------------------------------

            np.multiply(
                sage,
                lambda_mmr,
                out=mmr,
            )

            if one_minus_lambda != 0.0:
                mmr -= (
                    one_minus_lambda
                    * max_redundancy
                )

            # ----------------------------------------------------------
            # Disable already-selected candidates.
            # ----------------------------------------------------------

            mmr[~active] = -np.inf

            # ----------------------------------------------------------
            # np.argmax is deterministic and substantially cheaper than
            # constructing tie arrays on every iteration.
            #
            # Exact ties resolve by original candidate order.
            # ----------------------------------------------------------

            best_idx = int(
                np.argmax(mmr)
            )

            best_value = float(
                mmr[best_idx]
            )

            if not np.isfinite(best_value):
                break

            best_cid = candidate_id_array[
                best_idx
            ]

            selected.append(
                (
                    best_cid,
                    float(mmr[best_idx]),
                )
            )

            active[best_idx] = False

            # ----------------------------------------------------------
            # Incremental max redundancy.
            #
            # Only one vector operation per selected candidate.
            # ----------------------------------------------------------

            np.maximum(
                max_redundancy,
                redundancy_matrix[best_idx],
                out=max_redundancy,
            )

            max_redundancy[best_idx] = 0.0

        return selected

    # ----------------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------------

    def select_top_k(
        self,
        fused_scores: Mapping[str, float],
        metadata_map: Mapping[str, CandidateMetadata],
        query: SemanticQuery,
        *,
        top_k: int = 50,
        selection_method: SelectionMethod = SelectionMethod.GREEDY,
    ) -> List[FusedCandidate]:
        """
        Public QA-R3 API.

        Input:
            fused_scores:
                candidate_id -> upstream fused relevance.

            metadata_map:
                candidate_id -> semantic / temporal / visual metadata.

            query:
                semantic constraints.

        Output:
            diversity-aware Top-K candidates.

        Contract:
            fused_scores được xem là relevance từ upstream QA-R1/R2.
            R3 không tự fusion raw CLIP/OCR/ASR.

            Output fused_score luôn normalized về [0, 1].
        """

        if top_k <= 0:
            raise ValueError(
                "top_k phải lớn hơn 0"
            )

        if not fused_scores:
            return []

        candidate_ids = list(
            fused_scores.keys()
        )

        # ==================================================================
        # MMR FAST PATH
        # ==================================================================
        #
        # MMR chỉ cần:
        #
        #     SAGE
        #     redundancy matrix
        #     max redundancy
        #
        # Nó không cần:
        #
        #     semantic masks
        #     packed masks
        #     marginal coverage
        #     CoverageBreakdown cho N candidates
        #
        # Đây là critical optimization cho N=1000, K=50.
        # ==================================================================

        if selection_method == SelectionMethod.MMR:

            sage = self._compute_fast_sage_only(
                candidate_ids=candidate_ids,
                fused_scores=fused_scores,
                metadata_map=metadata_map,
                query=query,
                sage_lambda=self.sage_lambda,
                sage_eta=self.sage_eta,
            )

            selected_items = self._select_mmr(
                candidate_ids=candidate_ids,
                sage=sage,
                metadata_map=metadata_map,
                top_k=top_k,
            )

            # --------------------------------------------------------------
            # Chỉ tạo CoverageBreakdown cho selected K candidates.
            #
            # Trước đây:
            #     CoverageBreakdown × N
            #
            # Bây giờ:
            #     CoverageBreakdown × K
            # --------------------------------------------------------------

            results: List[
                FusedCandidate
            ] = []

            candidate_index = {
                cid: idx
                for idx, cid in enumerate(candidate_ids)
            }

            for rank, (cid, selection_score) in enumerate(
                selected_items,
                start=1,
            ):

                meta = metadata_map.get(cid)

                video_id = (
                    meta.video_id
                    if meta is not None
                    else "unknown"
                )

                frame_id = (
                    int(meta.frame_index)
                    if meta is not None
                    else -1
                )

                if meta is None:
                    breakdown = CoverageBreakdown(
                        entity=0.0,
                        action=0.0,
                        attribute=0.0,
                        relation=0.0,
                        matched_entity=0,
                        matched_action=0,
                        matched_attribute=0,
                        matched_relation=0,
                        total_entity=len(query.entities),
                        total_action=len(query.actions),
                        total_attribute=len(query.attributes),
                        total_relation=len(query.relations),
                    )

                    coverage_score = 0.0

                else:
                    breakdown = compute_semantic_coverage(
                        query,
                        meta,
                    )

                    coverage_score = breakdown.overall

                normalized_fused = _normalize_score(
                    fused_scores.get(
                        cid,
                        0.0,
                    )
                )

                results.append(
                    FusedCandidate(
                        candidate_id=cid,
                        video_id=video_id,
                        frame_id=frame_id,
                        fused_score=float(
                            normalized_fused
                        ),
                        coverage_score=float(
                            coverage_score
                        ),
                        sage_score=float(
                            sage[
                                candidate_index[cid]
                            ]
                        ),
                        selection_score=float(
                            selection_score
                        ),
                        selected_rank=rank,
                        coverage_breakdown=breakdown,
                    )
                )

            return results

        # ==================================================================
        # GREEDY PATH
        # ==================================================================
        #
        # Giữ nguyên optimized semantic path hiện tại vì Greedy đã đạt:
        #
        #     N=1000 K=50
        #     P95 = 9.409 ms
        #
        # Không cần mạo hiểm thay đổi.
        # ==================================================================

        if selection_method != SelectionMethod.GREEDY:
            raise ValueError(
                "Selection method không hợp lệ: "
                f"{selection_method}"
            )

        precomputed = (
            self._compute_fast_sage_arrays(
                candidate_ids=candidate_ids,
                fused_scores=fused_scores,
                metadata_map=metadata_map,
                query=query,
                sage_lambda=self.sage_lambda,
                sage_eta=self.sage_eta,
            )
        )

        (
            sage,
            fused,
            coverage,
            matched_entity,
            matched_action,
            matched_attribute,
            matched_relation,
            masks,
            packed_masks,
        ) = precomputed

        selected_items = self._select_greedy(
            candidate_ids=candidate_ids,
            fused_scores=fused_scores,
            metadata_map=metadata_map,
            query=query,
            top_k=top_k,
            precomputed=precomputed,
        )

        results: List[
            FusedCandidate
        ] = []

        candidate_index = {
            cid: idx
            for idx, cid in enumerate(candidate_ids)
        }

        for rank, (cid, selection_score) in enumerate(
            selected_items,
            start=1,
        ):

            idx = candidate_index[cid]

            breakdown = self._make_coverage_breakdown(
                matched_entity=int(
                    matched_entity[idx]
                ),
                matched_action=int(
                    matched_action[idx]
                ),
                matched_attribute=int(
                    matched_attribute[idx]
                ),
                matched_relation=int(
                    matched_relation[idx]
                ),
                query=query,
            )

            sage_score = float(sage[idx])
            coverage_score = float(coverage[idx])

            meta = metadata_map.get(cid)

            video_id = (
                meta.video_id
                if meta is not None
                else "unknown"
            )

            frame_id = (
                int(meta.frame_index)
                if meta is not None
                else -1
            )

            normalized_fused = float(fused[idx])

            results.append(
                FusedCandidate(
                    candidate_id=cid,
                    video_id=video_id,
                    frame_id=frame_id,
                    fused_score=float(
                        normalized_fused
                    ),
                    coverage_score=float(
                        coverage_score
                    ),
                    sage_score=float(
                        sage_score
                    ),
                    selection_score=float(
                        selection_score
                    ),
                    selected_rank=rank,
                    coverage_breakdown=breakdown,
                )
            )

        return results

    # ----------------------------------------------------------------------
    # QA-R4 HANDOFF
    # ----------------------------------------------------------------------

    def select_for_r4(
        self,
        query_id: str,
        fused_scores: Mapping[str, float],
        metadata_map: Mapping[str, CandidateMetadata],
        query: SemanticQuery,
        *,
        top_k: int = 50,
        selection_method: SelectionMethod = SelectionMethod.GREEDY,
    ) -> Dict[str, object]:
        """
        Chạy R3 và đóng gói trực tiếp contract đầu vào cho QA-R4.

        Output:
            {
                "query_id": "...",
                "candidates": [
                    {
                        "video_id": "...",
                        "frame_id": 123,
                        "score": 0.91,
                    }
                ],
            }
        """

        selected = self.select_top_k(
            fused_scores=fused_scores,
            metadata_map=metadata_map,
            query=query,
            top_k=top_k,
            selection_method=selection_method,
        )

        return build_r4_payload(
            query_id=query_id,
            candidates=selected,
        )

    # ----------------------------------------------------------------------
    # BACKWARD COMPATIBILITY
    # ----------------------------------------------------------------------

    def fuse_and_select(
        self,
        source_raw_scores: Mapping[
            str,
            Mapping[str, float],
        ],
        metadata_map: Mapping[
            str,
            CandidateMetadata,
        ],
        query_entities: Set[str],
        top_k: int = 10,
        selection_method: SelectionMethod = (
            SelectionMethod.GREEDY
        ),
    ) -> List[FusedCandidate]:
        """
        DEPRECATED COMPATIBILITY API.

        Không dùng cho QA-R3 production path mới.

        QA-R3 production nên dùng:

            select_top_k()

        với fused_scores đã được tạo bởi upstream QA-R1/R2.

        API legacy này chỉ có entity query nên không thể biểu diễn
        action / attribute / relation.

        Lưu ý:
            API này giữ behavior cũ: normalize từng source score rồi
            average các source có candidate.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k phải lớn hơn 0"
            )

        if not source_raw_scores:
            return []

        candidate_ids: Set[str] = set()

        for scores in source_raw_scores.values():
            candidate_ids.update(
                scores.keys()
            )

        fused_scores: Dict[str, float] = {}

        for cid in candidate_ids:

            values = []

            for scores in source_raw_scores.values():

                if cid in scores:
                    values.append(
                        _normalize_score(
                            scores[cid]
                        )
                    )

            if values:
                fused_scores[cid] = float(
                    sum(values)
                    / len(values)
                )

        query = SemanticQuery(
            entities=set(query_entities)
        )

        return self.select_top_k(
            fused_scores=fused_scores,
            metadata_map=metadata_map,
            query=query,
            top_k=top_k,
            selection_method=selection_method,
        )


# ============================================================================
# QA-R4 OUTPUT CONTRACT
# ============================================================================


def build_r4_payload(
    query_id: str,
    candidates: Sequence[FusedCandidate],
) -> Dict[str, object]:
    """
    Serialize kết quả R3 thành input tối thiểu của QA-R4.

    `score` là selection_score thực của Greedy/MMR. Danh sách luôn được
    chuẩn hóa theo selected_rank để R4 nhận đúng thứ tự selection của R3.

    Candidate thiếu metadata bị reject rõ ràng thay vì phát sinh
    video_id="unknown" hoặc frame_id=-1 trong payload production.
    """

    normalized_query_id = str(query_id).strip()

    if not normalized_query_id:
        raise ValueError(
            "query_id không được rỗng khi xuất payload cho QA-R4."
        )

    ordered = sorted(
        candidates,
        key=lambda item: item.selected_rank,
    )

    expected_ranks = list(
        range(1, len(ordered) + 1)
    )
    actual_ranks = [
        int(item.selected_rank)
        for item in ordered
    ]

    if actual_ranks != expected_ranks:
        raise ValueError(
            "selected_rank phải liên tiếp từ 1 khi xuất payload QA-R4."
        )

    seen_candidate_ids: Set[str] = set()
    payload_candidates: List[Dict[str, object]] = []

    for item in ordered:
        if item.candidate_id in seen_candidate_ids:
            raise ValueError(
                "R3 chứa candidate_id trùng khi xuất payload QA-R4: "
                f"{item.candidate_id}"
            )

        seen_candidate_ids.add(item.candidate_id)

        video_id = str(item.video_id).strip()
        frame_id = int(item.frame_id)
        score = float(item.selection_score)

        if (
            not video_id
            or video_id == "unknown"
            or frame_id < 0
        ):
            raise ValueError(
                "Candidate thiếu metadata video_id/frame_id "
                "hợp lệ cho QA-R4: "
                f"{item.candidate_id}"
            )

        if not np.isfinite(score):
            raise ValueError(
                "Candidate có selection_score không hữu hạn cho QA-R4: "
                f"{item.candidate_id}"
            )

        payload_candidates.append(
            {
                "video_id": video_id,
                "frame_id": frame_id,
                "score": score,
            }
        )

    return {
        "query_id": normalized_query_id,
        "candidates": payload_candidates,
    }


# ============================================================================
# SELF CHECK
# ============================================================================


def _self_check() -> None:
    """
    Self-check QA-R3.

    Kiểm tra:

        1. Entity coverage.
        2. Action coverage.
        3. Relation coverage.
        4. Constraint-weighted overall coverage.
        5. Marginal semantic coverage.
        6. Visual similarity mapping.
        7. Temporal redundancy.
        8. Greedy diversity.
        9. MMR.
        10. Determinism.
        11. Constructor compatibility.
        12. Redundancy matrix symmetry.
        13. Fast bitmask semantic coverage.
        14. Fast marginal semantic coverage.
        15. Packed marginal semantic coverage.
        16. Packed mask dtype selection.
        17. Invalid marginal coverage weight.
        18. Mixed visual feature dimensions.
        19. Normalized fused output.
    """

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    query = SemanticQuery(
        entities={"person"},
        actions={
            "run",
            "jump",
            "turn",
            "stop",
            "open",
        },
        relations={
            "person-near-car",
        },
    )

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    metadata = {

        "c1": CandidateMetadata(
            candidate_id="c1",
            video_id="video_1",
            frame_index=100,
            timestamp_sec=10.0,
            entities={"person"},
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
        ),

        "c3": CandidateMetadata(
            candidate_id="c3",
            video_id="video_3",
            frame_index=300,
            timestamp_sec=80.0,
            relations={
                "person-near-car",
            },
        ),

        "c4": CandidateMetadata(
            candidate_id="c4",
            video_id="video_1",
            frame_index=105,
            timestamp_sec=11.0,
            entities={"person"},
        ),
    }

    fused_scores = {
        "c1": 0.95,
        "c2": 0.90,
        "c3": 0.85,
        "c4": 0.94,
    }

    engine = MultiModalFusionEngine(
        sage_lambda=0.3,
        marginal_coverage_weight=0.3,
        sage_eta=0.0,
        time_threshold_sec=3.0,
        lambda_mmr=0.7,
        temporal_redundancy_weight=0.5,
        visual_redundancy_weight=0.3,
        semantic_redundancy_weight=0.2,
        redundancy_penalty_weight=0.10,
    )

    # ------------------------------------------------------------------
    # Coverage correctness
    # ------------------------------------------------------------------

    c1_cov = compute_semantic_coverage(
        query,
        metadata["c1"],
    )

    assert c1_cov.matched_entity == 1
    assert c1_cov.matched_action == 0
    assert c1_cov.matched_relation == 0

    assert c1_cov.matched_total == 1
    assert c1_cov.total_constraints == 7

    assert abs(
        c1_cov.overall - (1.0 / 7.0)
    ) < 1e-9

    c2_cov = compute_semantic_coverage(
        query,
        metadata["c2"],
    )

    assert c2_cov.matched_total == 4

    assert abs(
        c2_cov.overall - (4.0 / 7.0)
    ) < 1e-9

    c3_cov = compute_semantic_coverage(
        query,
        metadata["c3"],
    )

    assert c3_cov.matched_relation == 1

    assert abs(
        c3_cov.overall - (1.0 / 7.0)
    ) < 1e-9

    # ------------------------------------------------------------------
    # Marginal coverage correctness
    # ------------------------------------------------------------------

    marginal_first = compute_marginal_semantic_coverage(
        query,
        metadata["c1"],
        [],
    )

    assert abs(
        marginal_first - (1.0 / 7.0)
    ) < 1e-9

    marginal_after_c1 = compute_marginal_semantic_coverage(
        query,
        metadata["c2"],
        [metadata["c1"]],
    )

    assert abs(
        marginal_after_c1 - (4.0 / 7.0)
    ) < 1e-9

    # ------------------------------------------------------------------
    # Fast semantic masks
    # ------------------------------------------------------------------

    (
        entity_masks,
        action_masks,
        attribute_masks,
        relation_masks,
    ) = engine._build_semantic_masks(
        list(fused_scores.keys()),
        metadata,
        query,
    )

    assert entity_masks.dtype == np.uint16
    assert action_masks.dtype == np.uint16
    assert attribute_masks.dtype == np.uint16
    assert relation_masks.dtype == np.uint16

    # c1 entity => 1
    assert int(
        _POPCOUNT_LUT[
            entity_masks[0]
        ]
    ) == 1

    # c2 actions => 4
    assert int(
        _POPCOUNT_LUT[
            action_masks[1]
        ]
    ) == 4

    # c3 relation => 1
    assert int(
        _POPCOUNT_LUT[
            relation_masks[2]
        ]
    ) == 1

    # ------------------------------------------------------------------
    # Packed semantic masks
    # ------------------------------------------------------------------

    packed_masks = engine._build_packed_semantic_masks(
        entity_masks=entity_masks,
        action_masks=action_masks,
        attribute_masks=attribute_masks,
        relation_masks=relation_masks,
        query=query,
    )

    # 1 + 5 + 0 + 1 = 7 constraints -> uint16.
    assert packed_masks.dtype == np.uint16

    # c1 has exactly one packed bit.
    assert int(
        engine._packed_popcount(
            packed_masks[0:1]
        )[0]
    ) == 1

    # c2 has exactly four packed bits.
    assert int(
        engine._packed_popcount(
            packed_masks[1:2]
        )[0]
    ) == 4

    # c3 has exactly one packed bit.
    assert int(
        engine._packed_popcount(
            packed_masks[2:3]
        )[0]
    ) == 1

    # ------------------------------------------------------------------
    # Packed marginal correctness
    # ------------------------------------------------------------------

    packed_first = engine._fast_packed_marginal_gain_array(
        packed_masks=packed_masks,
        covered_mask=0,
        total_constraints=7,
    )

    assert abs(
        float(packed_first[0]) - (1.0 / 7.0)
    ) < 1e-6

    covered_c1 = int(
        packed_masks[0]
    )

    packed_after_c1 = engine._fast_packed_marginal_gain_array(
        packed_masks=packed_masks,
        covered_mask=covered_c1,
        total_constraints=7,
    )

    assert abs(
        float(packed_after_c1[1]) - (4.0 / 7.0)
    ) < 1e-6

    # ------------------------------------------------------------------
    # Packed dtype selection
    # ------------------------------------------------------------------

    assert (
        engine._packed_mask_dtype(16)
        == np.dtype(np.uint16)
    )

    assert (
        engine._packed_mask_dtype(32)
        == np.dtype(np.uint32)
    )

    assert (
        engine._packed_mask_dtype(64)
        == np.dtype(np.uint64)
    )

    try:
        engine._packed_mask_dtype(65)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "More than 64 packed semantic bits "
            "must raise ValueError."
        )

    # ------------------------------------------------------------------
    # Visual similarity
    # ------------------------------------------------------------------

    v_a = np.asarray(
        [1.0, 0.0],
        dtype=np.float32,
    )

    v_b = np.asarray(
        [0.0, 1.0],
        dtype=np.float32,
    )

    visual_a_b = visual_similarity(
        CandidateMetadata(
            "va",
            "video_a",
            0,
            0.0,
            feature_vector=v_a,
        ),
        CandidateMetadata(
            "vb",
            "video_b",
            0,
            0.0,
            feature_vector=v_b,
        ),
    )

    assert abs(
        visual_a_b - 0.0
    ) < 1e-9

    # ------------------------------------------------------------------
    # Temporal redundancy
    # ------------------------------------------------------------------

    temporal = temporal_similarity(
        metadata["c1"],
        metadata["c4"],
        threshold_sec=3.0,
    )

    assert temporal > 0.0

    temporal_far = temporal_similarity(
        metadata["c1"],
        metadata["c3"],
        threshold_sec=3.0,
    )

    assert temporal_far == 0.0

    # ------------------------------------------------------------------
    # Pair redundancy correctness
    # ------------------------------------------------------------------

    redundancy_c1_c4 = pair_redundancy(
        metadata["c1"],
        metadata["c4"],
        time_threshold_sec=3.0,
        temporal_weight=0.5,
        visual_weight=0.3,
        semantic_weight=0.2,
    )

    assert redundancy_c1_c4 > 0.0

    # ------------------------------------------------------------------
    # Redundancy matrix
    # ------------------------------------------------------------------

    matrix = engine._get_redundancy_matrix(
        list(fused_scores.keys()),
        metadata,
    )

    assert matrix.shape == (4, 4)

    assert np.allclose(
        matrix,
        matrix.T,
        atol=1e-6,
    )

    assert np.allclose(
        np.diag(matrix),
        0.0,
        atol=1e-7,
    )

    # ------------------------------------------------------------------
    # Greedy
    # ------------------------------------------------------------------

    greedy = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=3,
        selection_method=SelectionMethod.GREEDY,
    )

    greedy_ids = [
        item.candidate_id
        for item in greedy
    ]

    assert len(greedy_ids) == 3
    assert len(set(greedy_ids)) == 3

    assert "c2" in greedy_ids
    assert "c3" in greedy_ids

    # ------------------------------------------------------------------
    # MMR
    # ------------------------------------------------------------------

    mmr = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=3,
        selection_method=SelectionMethod.MMR,
    )

    mmr_ids = [
        item.candidate_id
        for item in mmr
    ]

    assert len(mmr_ids) == 3
    assert len(set(mmr_ids)) == 3

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    greedy_again = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=3,
        selection_method=SelectionMethod.GREEDY,
    )

    assert [
        item.candidate_id
        for item in greedy_again
    ] == greedy_ids

    mmr_again = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=3,
        selection_method=SelectionMethod.MMR,
    )

    assert [
        item.candidate_id
        for item in mmr_again
    ] == mmr_ids

    # ------------------------------------------------------------------
    # Top-K > N
    # ------------------------------------------------------------------

    oversized = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=query,
        top_k=100,
        selection_method=SelectionMethod.GREEDY,
    )

    assert len(oversized) == len(
        fused_scores
    )

    # ------------------------------------------------------------------
    # Missing metadata
    # ------------------------------------------------------------------

    missing_metadata_result = engine.select_top_k(
        fused_scores={
            "x1": 0.8,
            "x2": 0.7,
        },
        metadata_map={},
        query=query,
        top_k=2,
        selection_method=SelectionMethod.GREEDY,
    )

    assert len(
        missing_metadata_result
    ) == 2

    assert all(
        item.video_id == "unknown"
        for item in missing_metadata_result
    )

    # ------------------------------------------------------------------
    # Empty semantic query
    # ------------------------------------------------------------------

    empty_query_result = engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata,
        query=SemanticQuery(),
        top_k=2,
        selection_method=SelectionMethod.GREEDY,
    )

    assert len(
        empty_query_result
    ) == 2

    # ------------------------------------------------------------------
    # Normalized fused output
    # ------------------------------------------------------------------

    normalized_result = engine.select_top_k(
        fused_scores={
            "n1": 7.5,
        },
        metadata_map={},
        query=SemanticQuery(),
        top_k=1,
        selection_method=SelectionMethod.GREEDY,
    )

    assert len(normalized_result) == 1
    assert normalized_result[0].fused_score == 1.0

    # ------------------------------------------------------------------
    # Invalid redundancy penalty
    # ------------------------------------------------------------------

    try:
        MultiModalFusionEngine(
            redundancy_penalty_weight=-1.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative redundancy_penalty_weight "
            "must raise ValueError"
        )

    # ------------------------------------------------------------------
    # Invalid marginal coverage weight
    # ------------------------------------------------------------------

    try:
        MultiModalFusionEngine(
            marginal_coverage_weight=-1.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative marginal_coverage_weight "
            "must raise ValueError"
        )

    # ------------------------------------------------------------------
    # Too many labels per group
    # ------------------------------------------------------------------

    try:
        MultiModalFusionEngine().select_top_k(
            fused_scores={"x": 0.5},
            metadata_map={
                "x": CandidateMetadata(
                    "x",
                    "v",
                    0,
                    0.0,
                )
            },
            query=SemanticQuery(
                entities={
                    f"entity_{i}"
                    for i in range(17)
                }
            ),
            top_k=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "More than 16 labels in one semantic group "
            "must raise ValueError."
        )

    # ------------------------------------------------------------------
    # Mixed visual dimensions must fail explicitly.
    # ------------------------------------------------------------------

    mixed_visual_metadata = {
        "v1": CandidateMetadata(
            candidate_id="v1",
            video_id="video_v1",
            frame_index=0,
            timestamp_sec=0.0,
            feature_vector=np.ones(
                4,
                dtype=np.float32,
            ),
        ),
        "v2": CandidateMetadata(
            candidate_id="v2",
            video_id="video_v2",
            frame_index=0,
            timestamp_sec=0.0,
            feature_vector=np.ones(
                8,
                dtype=np.float32,
            ),
        ),
    }

    try:
        MultiModalFusionEngine().select_top_k(
            fused_scores={
                "v1": 0.8,
                "v2": 0.7,
            },
            metadata_map=mixed_visual_metadata,
            query=SemanticQuery(),
            top_k=2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Mixed visual feature dimensions "
            "must raise ValueError."
        )

    print(
        "[SELF-CHECK QA-R3] PASS — "
        "constraint-weighted coverage + "
        "uint16 semantic bitmask + "
        "packed semantic marginal gain + "
        "separate marginal coverage weight + "
        "vectorized redundancy + "
        "incremental max-redundancy + "
        "strict visual dimension contract + "
        "normalized fused output + "
        "greedy + MMR + "
        "deterministic selection."
    )


if __name__ == "__main__":
    _self_check()
