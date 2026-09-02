"""
QA-R4 — Adaptive K / Semantic Funnel Tests
==========================================

Test contract
-------------
QA-R4 nhận đúng 2 input JSONL:

    S4:
        query_id / id
        semantic_k_hint

    R3:
        query_id
        candidates

Pipeline:

    S4 JSONL ─┐
              ├── streaming pairwise ──> QA-R4 ──> output JSONL
    R3 JSONL ─┘

R4 KHÔNG:
    - tính uncertainty
    - tính entropy
    - tính normalized margin
    - rerank candidates
    - thay đổi thứ tự R3
    - load toàn bộ R3 vào RAM

R4 CHỈ:
    candidates[:semantic_k_hint]

Test groups
-----------
1. AdaptiveKConfig
2. K selection
3. R3 ordering preservation
4. S4 query-plan compatibility
5. Invalid K / clamping
6. Empty / short candidate pool
7. JSON serialization
8. Two-input JSONL integration
9. Input synchronization / desync
10. Streaming behavior
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from scripts.adaptive_k import (
    AdaptiveKConfig,
    AdaptiveKEngine,
    AdaptiveKResult,
    InputDesyncError,
    R4JSONEncoder,
    _normalize_for_json,
    run_jsonl_funnel,
)


# ============================================================================
# Helpers
# ============================================================================


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write JSONL fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    cls=R4JSONEncoder,
                    allow_nan=False,
                )
                + "\n"
            )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL output for assertions."""
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def make_candidates(n: int) -> List[Dict[str, Any]]:
    """
    Create deterministic fake R3 candidates.

    The rank field is deliberately used to verify that R4 preserves
    R3 ordering exactly.
    """
    return [
        {
            "rank": i,
            "frame_id": f"frame_{i:04d}",
            "score": 1.0 - (i * 0.0001),
        }
        for i in range(n)
    ]


def make_s4_records(
    query_ids: List[str],
    k: int = 100,
) -> List[Dict[str, Any]]:
    return [
        {
            "id": query_id,
            "semantic_k_hint": k,
            "uncertainty": 0.5,
            "preferred_modalities": ["visual"],
        }
        for query_id in query_ids
    ]


def make_r3_records(
    query_ids: List[str],
    candidate_count: int = 500,
) -> List[Dict[str, Any]]:
    return [
        {
            "query_id": query_id,
            "candidates": make_candidates(candidate_count),
        }
        for query_id in query_ids
    ]


# ============================================================================
# Configuration
# ============================================================================


class TestAdaptiveKConfig:
    def test_default_allowed_k(self):
        config = AdaptiveKConfig()

        assert config.allowed_k == (100, 300, 500)
        assert config.strict is True

    def test_custom_allowed_k_is_normalized(self):
        config = AdaptiveKConfig(
            allowed_k=(500, 100, 300, 300),
        )

        assert config.allowed_k == (100, 300, 500)

    def test_empty_allowed_k_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveKConfig(allowed_k=())

    def test_non_positive_k_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveKConfig(allowed_k=(0, 100))

    def test_negative_k_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveKConfig(allowed_k=(-100, 100))


# ============================================================================
# Core selection
# ============================================================================


class TestAdaptiveKSelection:
    @pytest.fixture
    def engine(self):
        return AdaptiveKEngine()

    @pytest.fixture
    def candidates(self):
        return make_candidates(500)

    def test_k_100(self, engine, candidates):
        result = engine.select(
            query_id="q100",
            candidates=candidates,
            semantic_k_hint=100,
        )

        assert isinstance(result, AdaptiveKResult)
        assert result.query_id == "q100"
        assert result.k_requested == 100
        assert result.k_effective == 100
        assert result.k_available == 500
        assert len(result.selected_candidates) == 100
        assert result.status == "ok"
        assert result.fallback_reason is None

    def test_k_300(self, engine, candidates):
        result = engine.select(
            query_id="q300",
            candidates=candidates,
            semantic_k_hint=300,
        )

        assert result.k_requested == 300
        assert result.k_effective == 300
        assert result.k_available == 500
        assert len(result.selected_candidates) == 300
        assert result.status == "ok"

    def test_k_500(self, engine, candidates):
        result = engine.select(
            query_id="q500",
            candidates=candidates,
            semantic_k_hint=500,
        )

        assert result.k_requested == 500
        assert result.k_effective == 500
        assert result.k_available == 500
        assert len(result.selected_candidates) == 500
        assert result.status == "ok"

    def test_r4_is_exact_prefix_slice(self, engine, candidates):
        result = engine.select(
            query_id="prefix",
            candidates=candidates,
            semantic_k_hint=100,
        )

        assert result.selected_candidates == tuple(
            candidates[:100]
        )

    def test_r4_does_not_rerank(self, engine):
        candidates = [
            {"rank": 0, "score": 0.1},
            {"rank": 1, "score": 0.9},
            {"rank": 2, "score": 0.8},
        ]

        result = engine.select(
            query_id="no-rerank",
            candidates=candidates,
            semantic_k_hint=100,
        )

        assert result.selected_candidates == tuple(candidates)

    def test_r4_preserves_duplicate_candidate_payloads(self, engine):
        candidates = [
            {"frame_id": "same", "rank": 0},
            {"frame_id": "same", "rank": 1},
            {"frame_id": "different", "rank": 2},
        ]

        result = engine.select(
            query_id="duplicates",
            candidates=candidates,
            semantic_k_hint=100,
        )

        assert result.selected_candidates == tuple(candidates)


# ============================================================================
# S4 contract
# ============================================================================


class TestS4Contract:
    def test_accepts_query_id(self):
        engine = AdaptiveKEngine()

        result = engine.select_from_query_plan(
            {
                "query_id": "q-query-id",
                "semantic_k_hint": 100,
            },
            make_candidates(100),
        )

        assert result.query_id == "q-query-id"

    def test_accepts_s4_id(self):
        engine = AdaptiveKEngine()

        result = engine.select_from_query_plan(
            {
                "id": "q-id",
                "semantic_k_hint": 100,
            },
            make_candidates(100),
        )

        assert result.query_id == "q-id"

    def test_query_id_takes_precedence(self):
        engine = AdaptiveKEngine()

        result = engine.select_from_query_plan(
            {
                "id": "old-id",
                "query_id": "query-id",
                "semantic_k_hint": 100,
            },
            make_candidates(100),
        )

        assert result.query_id == "query-id"

    def test_missing_query_id_rejected(self):
        engine = AdaptiveKEngine()

        with pytest.raises(KeyError):
            engine.select_from_query_plan(
                {
                    "semantic_k_hint": 100,
                },
                make_candidates(100),
            )

    def test_missing_semantic_k_hint_rejected(self):
        engine = AdaptiveKEngine()

        with pytest.raises(KeyError):
            engine.select_from_query_plan(
                {
                    "id": "q1",
                },
                make_candidates(100),
            )

    def test_uncertainty_is_not_used(self):
        engine = AdaptiveKEngine()

        candidates = make_candidates(500)

        easy = engine.select_from_query_plan(
            {
                "id": "q1",
                "semantic_k_hint": 300,
                "uncertainty": 0.01,
                "preferred_modalities": ["ocr"],
            },
            candidates,
        )

        hard = engine.select_from_query_plan(
            {
                "id": "q1",
                "semantic_k_hint": 300,
                "uncertainty": 0.99,
                "preferred_modalities": [
                    "visual",
                    "ocr",
                    "asr",
                ],
            },
            candidates,
        )

        assert easy.k_requested == hard.k_requested
        assert easy.k_effective == hard.k_effective
        assert (
            easy.selected_candidates
            == hard.selected_candidates
        )

    def test_normalized_margin_is_not_required(self):
        engine = AdaptiveKEngine()

        result = engine.select_from_query_plan(
            {
                "id": "q1",
                "semantic_k_hint": 100,
            },
            make_candidates(100),
        )

        assert result.k_effective == 100
        assert not hasattr(result, "normalized_margin")


# ============================================================================
# Invalid K
# ============================================================================


class TestInvalidK:
    def test_invalid_k_strict_mode_raises(self):
        engine = AdaptiveKEngine(
            AdaptiveKConfig(
                allowed_k=(100, 300, 500),
                strict=True,
            )
        )

        with pytest.raises(ValueError):
            engine.select(
                query_id="invalid",
                candidates=make_candidates(500),
                semantic_k_hint=250,
            )

    def test_invalid_k_non_strict_clamps(self):
        engine = AdaptiveKEngine(
            AdaptiveKConfig(
                allowed_k=(100, 300, 500),
                strict=False,
            )
        )

        result = engine.select(
            query_id="clamped",
            candidates=make_candidates(500),
            semantic_k_hint=250,
        )

        assert result.k_requested == 250
        assert result.k_effective == 300
        assert result.status == "ok_with_warning"
        assert (
            result.fallback_reason
            == "clamped_invalid_semantic_k"
        )

    def test_clamp_to_lower_bound(self):
        engine = AdaptiveKEngine(
            AdaptiveKConfig(
                allowed_k=(100, 300, 500),
                strict=False,
            )
        )

        result = engine.select(
            query_id="low",
            candidates=make_candidates(500),
            semantic_k_hint=1,
        )

        assert result.k_effective == 100
        assert result.status == "ok_with_warning"
        assert (
            result.fallback_reason
            == "clamped_invalid_semantic_k"
        )

    def test_clamp_to_upper_bound(self):
        engine = AdaptiveKEngine(
            AdaptiveKConfig(
                allowed_k=(100, 300, 500),
                strict=False,
            )
        )

        result = engine.select(
            query_id="high",
            candidates=make_candidates(500),
            semantic_k_hint=9999,
        )

        assert result.k_effective == 500
        assert result.status == "ok_with_warning"
        assert (
            result.fallback_reason
            == "clamped_invalid_semantic_k"
        )

    def test_boolean_k_is_rejected(self):
        engine = AdaptiveKEngine()

        with pytest.raises(ValueError):
            engine.select(
                query_id="bool-k",
                candidates=make_candidates(500),
                semantic_k_hint=True,
            )

    def test_string_integer_k_is_accepted(self):
        engine = AdaptiveKEngine()

        result = engine.select(
            query_id="string-k",
            candidates=make_candidates(500),
            semantic_k_hint="100",
        )

        assert result.k_requested == 100
        assert result.k_effective == 100


# ============================================================================
# Candidate pool edge cases
# ============================================================================


class TestCandidatePool:
    def test_short_candidate_pool(self):
        engine = AdaptiveKEngine()

        candidates = make_candidates(20)

        result = engine.select(
            query_id="short",
            candidates=candidates,
            semantic_k_hint=100,
        )

        assert result.k_requested == 100
        assert result.k_effective == 20
        assert result.k_available == 20
        assert len(result.selected_candidates) == 20
        assert result.status == "ok_with_warning"
        assert (
            result.fallback_reason
            == "candidate_pool_below_requested_k"
        )

    def test_empty_candidate_pool(self):
        engine = AdaptiveKEngine()

        result = engine.select(
            query_id="empty",
            candidates=[],
            semantic_k_hint=300,
        )

        assert result.k_requested == 300
        assert result.k_effective == 0
        assert result.k_available == 0
        assert result.selected_candidates == ()
        assert result.status == "ok_with_warning"
        assert (
            result.fallback_reason
            == "empty_candidate_pool"
        )

    def test_none_candidate_pool_rejected(self):
        engine = AdaptiveKEngine()

        with pytest.raises(ValueError):
            engine.select(
                query_id="none",
                candidates=None,
                semantic_k_hint=100,
            )

    def test_k_must_be_positive(self):
        engine = AdaptiveKEngine()

        with pytest.raises(ValueError):
            engine.select(
                query_id="zero",
                candidates=make_candidates(10),
                semantic_k_hint=0,
            )


# ============================================================================
# Serialization
# ============================================================================


class TestSerialization:
    def test_dataclass_serialization(self):
        engine = AdaptiveKEngine()

        result = engine.select(
            query_id="serialize",
            candidates=make_candidates(100),
            semantic_k_hint=100,
        )

        normalized = _normalize_for_json(result)

        assert normalized["query_id"] == "serialize"
        assert normalized["k_requested"] == 100
        assert normalized["k_effective"] == 100
        assert len(normalized["selected_candidates"]) == 100

    def test_json_encoder_serializes_result(self):
        engine = AdaptiveKEngine()

        result = engine.select(
            query_id="json",
            candidates=make_candidates(100),
            semantic_k_hint=100,
        )

        encoded = json.dumps(
            result,
            cls=R4JSONEncoder,
            ensure_ascii=False,
            allow_nan=False,
        )

        decoded = json.loads(encoded)

        assert decoded["query_id"] == "json"
        assert len(decoded["selected_candidates"]) == 100

    def test_fake_numpy_scalar(self):
        class FakeNumpyScalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        value = FakeNumpyScalar(0.123)

        assert _normalize_for_json(value) == 0.123

    def test_fake_numpy_array(self):
        class FakeNumpyArray:
            def tolist(self):
                return [1, 2, 3]

        value = FakeNumpyArray()

        assert _normalize_for_json(value) == [1, 2, 3]

    def test_fake_torch_tensor(self):
        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [0.1, 0.2, 0.3]

        value = FakeTensor()

        assert _normalize_for_json(value) == [
            0.1,
            0.2,
            0.3,
        ]

    def test_non_finite_float_rejected(self):
        with pytest.raises(ValueError):
            _normalize_for_json(float("nan"))

        with pytest.raises(ValueError):
            _normalize_for_json(float("inf"))

        with pytest.raises(ValueError):
            _normalize_for_json(float("-inf"))


# ============================================================================
# Two-input JSONL integration
# ============================================================================


class TestJSONLFunnel:
    def test_basic_two_input_funnel(self, tmp_path):
        s4_path = tmp_path / "s4.jsonl"
        r3_path = tmp_path / "r3.jsonl"
        output_path = tmp_path / "r4.jsonl"

        query_ids = ["q01", "q02", "q03"]

        write_jsonl(
            s4_path,
            make_s4_records(
                query_ids=query_ids,
                k=100,
            ),
        )

        write_jsonl(
            r3_path,
            make_r3_records(
                query_ids=query_ids,
                candidate_count=500,
            ),
        )

        yielded = list(
            run_jsonl_funnel(
                s4_path=s4_path,
                r3_path=r3_path,
                output_path=output_path,
            )
        )

        output = read_jsonl(output_path)

        assert len(yielded) == 3
        assert len(output) == 3

        assert [
            result.query_id
            for result in yielded
        ] == query_ids

        assert [
            record["query_id"]
            for record in output
        ] == query_ids

    def test_output_contains_selected_candidates(self, tmp_path):
        s4_path = tmp_path / "s4.jsonl"
        r3_path = tmp_path / "r3.jsonl"
        output_path = tmp_path / "r4.jsonl"

        query_ids = ["q01"]

        write_jsonl(
            s4_path,
            make_s4_records(
                query_ids=query_ids,
                k=100,
            ),
        )

        write_jsonl(
            r3_path,
            make_r3_records(
                query_ids=query_ids,
                candidate_count=500,
            ),
        )

        list(
            run_jsonl_funnel(
                s4_path,
                r3_path,
                output_path,
            )
        )

        output = read_jsonl(output_path)

        assert len(output) == 1

        selected = output[0]["selected_candidates"]

        assert len(selected) == 100

        assert [
            candidate["rank"]
            for candidate in selected
        ] == list(range(100))

    def test_k_distribution_100_300_500(self, tmp_path):
        s4_path = tmp_path / "s4.jsonl"
        r3_path = tmp_path / "r3.jsonl"
        output_path = tmp_path / "r4.jsonl"

        s4_records = [
            {
                "id": "q100",
                "semantic_k_hint": 100,
            },
            {
                "id": "q300",
                "semantic_k_hint": 300,
            },
            {
                "id": "q500",
                "semantic_k_hint": 500,
            },
        ]

        r3_records = [
            {
                "query_id": "q100",
                "candidates": make_candidates(500),
            },
            {
                "query_id": "q300",
                "candidates": make_candidates(500),
            },
            {
                "query_id": "q500",
                "candidates": make_candidates(500),
            },
        ]

        write_jsonl(s4_path, s4_records)
        write_jsonl(r3_path, r3_records)

        yielded = list(
            run_jsonl_funnel(
                s4_path,
                r3_path,
                output_path,
            )
        )

        assert [
            result.k_requested
            for result in yielded
        ] == [100, 300, 500]

        assert [
            result.k_effective
            for result in yielded
        ] == [100, 300, 500]

        output = read_jsonl(output_path)

        assert [
            len(record["selected_candidates"])
            for record in output
        ] == [100, 300, 500]

    def test_r3_order_is_preserved(self, tmp_path):
        s4_path = tmp_path / "s4.jsonl"
        r3_path = tmp_path / "r3.jsonl"
        output_path = tmp_path / "r4.jsonl"

        candidates = [
            {"rank": 17, "frame_id": "A"},
            {"rank": 3, "frame_id": "B"},
            {"rank": 99, "frame_id": "C"},
            {"rank": 1, "frame_id": "D"},
        ]

        write_jsonl(
            s4_path,
            [
                {
                    "id": "q1",
                    "semantic_k_hint": 300,
                }
            ],
        )

        write_jsonl(
            r3_path,
            [
                {
                    "query_id": "q1",
                    "candidates": candidates,
                }
            ],
        )

        list(
            run_jsonl_funnel(
                s4_path,
                r3_path,
                output_path,
            )
        )

        output = read_jsonl(output_path)

        assert output[0]["selected_candidates"] == candidates

    def test_streaming_generator_interface(self, tmp_path):
        s4_path = tmp_path / "s4.jsonl"
        r3_path = tmp_path / "r3.jsonl"
        output_path = tmp_path / "r4.jsonl"

        query_ids = [
            "q01",
            "q02",
            "q03",
        ]

        write_jsonl(
            s4_path,
            make_s4_records(query_ids, k=100),
        )

        write_jsonl(
            r3_path,
            make_r3_records(query_ids, 100),
        )

        iterator = run_jsonl_funnel(
            s4_path,
            r3_path,
            output_path,
        )

        # The function must be a generator/iterator, not a returned list.
        assert hasattr(iterator, "__iter__")
        assert hasattr(iterator, "__next__")

        first = next(iterator)

        assert first.query_id == "q01"
        assert first.k_effective == 100

        second = next(iterator)

        assert second.query_id == "q02"

        third = next(iterator)

        assert third.query_id == "q03"

        with pytest.raises(StopIteration):
            next(iterator)

        del iterator
        gc.collect()

        output = read_jsonl(output_path)

        assert len(output) == 3


# ============================================================================
# Desynchronization
# ============================================================================


class TestInputSynchronization:
    def test_query_order_mismatch_raises(self, tmp_path):
        s4_path = tmp_path / "s4.jsonl"
        r3_path = tmp_path / "r3.jsonl"
        output_path = tmp_path / "r4.jsonl"

        write_jsonl(
            s4_path,
            [
                {
                    "id": "q01",
                    "semantic_k_hint": 100,
                },
                {
                    "id": "q02",
                    "semantic_k_hint": 100,
                },
            ],
        )

        write_jsonl(
            r3_path,
            [
                {
                    "query_id": "q01",
                    "candidates": make_candidates(100),
                },
                {
                    "query_id": "q03",
                    "candidates": make_candidates(100),
                },
            ],
        )

        with pytest.raises(InputDesyncError):
            list(
                run_jsonl_funnel(
                    s4_path,
                    r3_path,
                    output_path,
                )
            )

    def test_r3_ends_first_raises(self, tmp_path):
        s4_path = tmp_path / "s4.jsonl"
        r3_path = tmp_path / "r3.jsonl"
        output_path = tmp_path / "r4.jsonl"

        write_jsonl(
            s4_path,
            [
                {
                    "id": "q01",
                    "semantic_k_hint": 100,
                },
                {
                    "id": "q02",
                    "semantic_k_hint": 100,
                },
            ],
        )

        write_jsonl(
            r3_path,
            [
                {
                    "query_id": "q01",
                    "candidates": make_candidates(100),
                },
            ],
        )

        with pytest.raises(InputDesyncError):
            list(
                run_jsonl_funnel(
                    s4_path,
                    r3_path,
                    output_path,
                )
            )

    def test_s4_ends_first_raises(self, tmp_path):
        s4_path = tmp_path / "s4.jsonl"
        r3_path = tmp_path / "r3.jsonl"
        output_path = tmp_path / "r4.jsonl"

        write_jsonl(
            s4_path,
            [
                {
                    "id": "q01",
                    "semantic_k_hint": 100,
                },
            ],
        )

        write_jsonl(
            r3_path,
            [
                {
                    "query_id": "q01",
                    "candidates": make_candidates(100),
                },
                {
                    "query_id": "q02",
                    "candidates": make_candidates(100),
                },
            ],
        )

        with pytest.raises(InputDesyncError):
            list(
                run_jsonl_funnel(
                    s4_path,
                    r3_path,
                    output_path,
                )
            )

    def test_empty_both_streams_is_valid(self, tmp_path):
        s4_path = tmp_path / "s4.jsonl"
        r3_path = tmp_path / "r3.jsonl"
        output_path = tmp_path / "r4.jsonl"

        write_jsonl(s4_path, [])
        write_jsonl(r3_path, [])

        yielded = list(
            run_jsonl_funnel(
                s4_path,
                r3_path,
                output_path,
            )
        )

        assert yielded == []

        output = read_jsonl(output_path)

        assert output == []

    def test_r3_missing_candidates_raises(self, tmp_path):
        s4_path = tmp_path / "s4.jsonl"
        r3_path = tmp_path / "r3.jsonl"
        output_path = tmp_path / "r4.jsonl"

        write_jsonl(
            s4_path,
            [
                {
                    "id": "q01",
                    "semantic_k_hint": 100,
                }
            ],
        )

        write_jsonl(
            r3_path,
            [
                {
                    "query_id": "q01",
                }
            ],
        )

        with pytest.raises(KeyError):
            list(
                run_jsonl_funnel(
                    s4_path,
                    r3_path,
                    output_path,
                )
            )

    def test_s4_missing_semantic_k_hint_raises(self, tmp_path):
        s4_path = tmp_path / "s4.jsonl"
        r3_path = tmp_path / "r3.jsonl"
        output_path = tmp_path / "r4.jsonl"

        write_jsonl(
            s4_path,
            [
                {
                    "id": "q01",
                }
            ],
        )

        write_jsonl(
            r3_path,
            [
                {
                    "query_id": "q01",
                    "candidates": make_candidates(100),
                }
            ],
        )

        with pytest.raises(KeyError):
            list(
                run_jsonl_funnel(
                    s4_path,
                    r3_path,
                    output_path,
                )
            )


# ============================================================================
# End-to-end payload integrity
# ============================================================================


class TestEndToEndIntegrity:
    def test_payload_integrity_for_multiple_queries(self, tmp_path):
        s4_path = tmp_path / "s4.jsonl"
        r3_path = tmp_path / "r3.jsonl"
        output_path = tmp_path / "r4.jsonl"

        query_ids = [
            "q01",
            "q02",
            "q03",
            "q04",
        ]

        s4_records = [
            {
                "id": "q01",
                "semantic_k_hint": 100,
            },
            {
                "id": "q02",
                "semantic_k_hint": 300,
            },
            {
                "id": "q03",
                "semantic_k_hint": 500,
            },
            {
                "id": "q04",
                "semantic_k_hint": 100,
            },
        ]

        r3_records = []

        for index, query_id in enumerate(query_ids):
            candidates = [
                {
                    "rank": rank,
                    "video_id": f"{query_id}_video",
                    "frame_id": (
                        f"{query_id}_frame_{rank}"
                    ),
                    "score": 1000 - rank - index,
                }
                for rank in range(500)
            ]

            r3_records.append(
                {
                    "query_id": query_id,
                    "candidates": candidates,
                }
            )

        write_jsonl(s4_path, s4_records)
        write_jsonl(r3_path, r3_records)

        yielded = list(
            run_jsonl_funnel(
                s4_path,
                r3_path,
                output_path,
            )
        )

        output = read_jsonl(output_path)

        assert len(yielded) == 4
        assert len(output) == 4

        expected_k = [100, 300, 500, 100]

        assert [
            result.k_effective
            for result in yielded
        ] == expected_k

        assert [
            len(record["selected_candidates"])
            for record in output
        ] == expected_k

        for record, expected in zip(
            output,
            expected_k,
        ):
            selected = record["selected_candidates"]

            assert [
                candidate["rank"]
                for candidate in selected
            ] == list(range(expected))


# ============================================================================
# Public smoke test
# ============================================================================


def test_module_smoke():
    """
    Basic import / object construction smoke test.

    This catches accidental API breakage before the integration tests.
    """
    config = AdaptiveKConfig()
    engine = AdaptiveKEngine(config)

    assert engine.config == config
    assert config.allowed_k == (100, 300, 500)