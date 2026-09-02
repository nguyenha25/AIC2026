"""
QA-R4 — Adaptive Candidate Funnel
=================================

Mục đích
--------
R4 là một thin funnel đứng sau:

    QA-S4 semantic parsing
            +
    QA-R3 fused candidate pool
            |
            v
        QA-R4 funnel
            |
            v
    adaptive_candidates

Contract hiện tại
-----------------
S4 input:
    D:/aic-data/runs/n01_semantic_parsing.jsonl

Mỗi S4 record cần:
    - id hoặc query_id
    - semantic_k_hint

semantic_k_hint được S4 cung cấp sẵn và hiện tại có các giá trị:
    - 100
    - 300
    - 500

R3 input:
    Output từ fusion_engine, mỗi record có:
    - query_id
    - candidates

R4 KHÔNG:
    - tính lại uncertainty
    - đọc entropy
    - tính score margin
    - rerank candidates
    - thay đổi thứ tự candidates của R3
    - sửa QA-S4
    - sửa QA-R3

R4 CHỈ:
    1. Match S4 query với R3 query.
    2. Lấy semantic_k_hint từ S4.
    3. Chọn candidates[:semantic_k_hint].
    4. Serialize kết quả ra JSONL.

Memory / Streaming
------------------
Toàn bộ pipeline được thiết kế streaming:

    S4 line
        +
    R3 line
        |
        v
    process 1 query
        |
        v
    write JSONL
        |
        v
    yield result
        |
        v
    query tiếp theo

Không load toàn bộ S4/R3 JSONL vào RAM.
Không tích lũy AdaptiveKResult của tất cả query vào một list.

run_jsonl_funnel() là generator và yield từng result.

Input Synchronization
---------------------
Hai file S4 và R3 phải có:
    - cùng số record
    - cùng thứ tự query_id

Nếu lệch:
    -> raise InputDesyncError

Điều này tránh silent data drop.

Serialization
-------------
R3 candidates có thể chứa:
    - numpy.float32
    - numpy.int64
    - numpy.ndarray
    - torch.Tensor
    - các scalar tương tự

R4JSONEncoder / _normalize_for_json xử lý các trường hợp này
trước khi json.dumps().

Invalid semantic_k_hint
-----------------------
strict=True:
    -> raise ValueError

strict=False:
    -> clamp về K hợp lệ gần nhất
    -> status = "ok_with_warning"
    -> fallback_reason = "clamped_invalid_semantic_k"

Candidate pool ngắn hơn K
-------------------------
Không fabricate candidate.

Nếu R3 có ít hơn K candidate:
    - lấy toàn bộ candidate hiện có
    - status = "ok_with_warning"
    - fallback_reason = "candidate_pool_below_requested_k"

Empty candidate pool:
    - selected_candidates = ()
    - status = "ok_with_warning"
    - fallback_reason = "empty_candidate_pool"
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple


# ============================================================================
# Exceptions
# ============================================================================


class InputDesyncError(ValueError):
    """Raised when S4 and R3 JSONL streams are not aligned."""


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class AdaptiveKConfig:
    """
    Configuration for QA-R4 candidate funnel.

    allowed_k
        Valid semantic_k_hint values produced by the frozen S4 stage.

    strict
        If True, invalid semantic_k_hint raises immediately.

        If False, invalid values are clamped to the nearest allowed K and
        the result is explicitly marked as a warning.
    """

    allowed_k: Tuple[int, ...] = (100, 300, 500)
    strict: bool = True

    def __post_init__(self) -> None:
        if not self.allowed_k:
            raise ValueError("allowed_k must not be empty")

        normalized = tuple(sorted(set(int(k) for k in self.allowed_k)))

        if any(k <= 0 for k in normalized):
            raise ValueError("all allowed_k values must be positive")

        object.__setattr__(self, "allowed_k", normalized)


# ============================================================================
# Result
# ============================================================================


@dataclass(frozen=True)
class AdaptiveKResult:
    """
    Result metadata for one query.

    selected_candidates
        Full selected candidate payload.

        Important:
        This object is yielded immediately by run_jsonl_funnel().
        It is NOT accumulated internally.

    k_requested
        K requested by S4 semantic_k_hint.

    k_effective
        Actual number of candidates selected.

    k_available
        Number of candidates available from R3 before slicing.

    status
        One of:
            - "ok"
            - "ok_with_warning"
            - "error"

    fallback_reason
        Optional machine-readable reason.
    """

    schema_version: str
    task: str
    query_id: str

    k_requested: int
    k_effective: int
    k_available: int

    selected_candidates: Tuple[Any, ...]

    status: str
    error: Optional[str]
    fallback_reason: Optional[str]


# ============================================================================
# Semantic K helpers
# ============================================================================


def validate_semantic_k_hint(
    semantic_k_hint: Any,
    config: AdaptiveKConfig,
) -> int:
    """
    Validate semantic_k_hint against the configured allowed K values.

    strict=True:
        invalid K -> ValueError

    strict=False:
        invalid K -> nearest allowed K
    """
    if isinstance(semantic_k_hint, bool):
        raise ValueError("semantic_k_hint must be an integer, not bool")

    try:
        k = int(semantic_k_hint)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"semantic_k_hint must be an integer, got {semantic_k_hint!r}"
        ) from exc

    if k in config.allowed_k:
        return k

    if config.strict:
        raise ValueError(
            f"Unsupported semantic_k_hint={k}; "
            f"allowed values are {config.allowed_k}"
        )

    return min(
        config.allowed_k,
        key=lambda allowed: (abs(allowed - k), allowed),
    )


def _resolve_semantic_k(
    semantic_k_hint: Any,
    config: AdaptiveKConfig,
) -> Tuple[int, bool]:
    """
    Return:
        (effective_k, was_clamped)
    """
    if isinstance(semantic_k_hint, bool):
        raise ValueError("semantic_k_hint must be an integer, not bool")

    try:
        requested = int(semantic_k_hint)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"semantic_k_hint must be an integer, got {semantic_k_hint!r}"
        ) from exc

    if requested in config.allowed_k:
        return requested, False

    if config.strict:
        raise ValueError(
            f"Unsupported semantic_k_hint={requested}; "
            f"allowed values are {config.allowed_k}"
        )

    effective = min(
        config.allowed_k,
        key=lambda allowed: (abs(allowed - requested), allowed),
    )

    return effective, True


# ============================================================================
# Candidate selection
# ============================================================================


def select_candidates(
    candidates: Sequence[Any],
    k: int,
) -> Tuple[Any, ...]:
    """
    Select the first K candidates from R3.

    IMPORTANT:
        No sorting.
        No reranking.
        No score manipulation.

    R3 has already produced the ranked candidate pool.
    R4 only performs the funnel slice.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    return tuple(candidates[:k])


# ============================================================================
# Engine
# ============================================================================


class AdaptiveKEngine:
    """
    QA-R4 funnel engine.

    The engine intentionally has no uncertainty / entropy / margin logic.

    Those decisions have already been made upstream by QA-S4 through
    semantic_k_hint.
    """

    SCHEMA_VERSION = "qa-r4.v1"
    TASK = "qa_r4_adaptive_k"

    def __init__(self, config: Optional[AdaptiveKConfig] = None) -> None:
        self.config = config or AdaptiveKConfig()

    def select(
        self,
        query_id: str,
        candidates: Sequence[Any],
        semantic_k_hint: Any,
    ) -> AdaptiveKResult:
        """
        Funnel one R3 candidate pool using S4 semantic_k_hint.
        """
        if not query_id:
            raise ValueError("query_id must not be empty")

        if candidates is None:
            raise ValueError(
                f"candidates must not be None for query_id={query_id!r}"
            )

        if not isinstance(candidates, Sequence):
            raise TypeError(
                f"candidates must be a sequence for query_id={query_id!r}; "
                f"got {type(candidates).__name__}"
            )

        requested_k = self._parse_requested_k(semantic_k_hint)

        effective_k, was_clamped = _resolve_semantic_k(
            requested_k,
            self.config,
        )

        available = len(candidates)

        selected = select_candidates(
            candidates=candidates,
            k=effective_k,
        )

        if available == 0:
            return AdaptiveKResult(
                schema_version=self.SCHEMA_VERSION,
                task=self.TASK,
                query_id=str(query_id),
                k_requested=requested_k,
                k_effective=0,
                k_available=0,
                selected_candidates=tuple(),
                status="ok_with_warning",
                error=None,
                fallback_reason="empty_candidate_pool",
            )

        if was_clamped:
            return AdaptiveKResult(
                schema_version=self.SCHEMA_VERSION,
                task=self.TASK,
                query_id=str(query_id),
                k_requested=requested_k,
                k_effective=len(selected),
                k_available=available,
                selected_candidates=selected,
                status="ok_with_warning",
                error=None,
                fallback_reason="clamped_invalid_semantic_k",
            )

        if available < effective_k:
            return AdaptiveKResult(
                schema_version=self.SCHEMA_VERSION,
                task=self.TASK,
                query_id=str(query_id),
                k_requested=requested_k,
                k_effective=len(selected),
                k_available=available,
                selected_candidates=selected,
                status="ok_with_warning",
                error=None,
                fallback_reason="candidate_pool_below_requested_k",
            )

        return AdaptiveKResult(
            schema_version=self.SCHEMA_VERSION,
            task=self.TASK,
            query_id=str(query_id),
            k_requested=requested_k,
            k_effective=len(selected),
            k_available=available,
            selected_candidates=selected,
            status="ok",
            error=None,
            fallback_reason=None,
        )

    @staticmethod
    def _parse_requested_k(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("semantic_k_hint must be an integer, not bool")

        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"semantic_k_hint must be an integer, got {value!r}"
            ) from exc

    def select_from_query_plan(
        self,
        query_plan: Mapping[str, Any],
        candidates: Sequence[Any],
    ) -> AdaptiveKResult:
        """
        Select candidates directly from a frozen S4 QueryPlanQA-like mapping.

        Accepted query ID fields:
            - id
            - query_id

        Required:
            - semantic_k_hint

        Existing S4 fields such as:
            - uncertainty
            - preferred_modalities

        are intentionally ignored by R4.
        """
        query_id = query_plan.get("query_id")

        if query_id is None:
            query_id = query_plan.get("id")

        if query_id is None:
            raise KeyError(
                "S4 query plan must contain either 'id' or 'query_id'"
            )

        if "semantic_k_hint" not in query_plan:
            raise KeyError(
                f"S4 query {query_id!r} is missing 'semantic_k_hint'"
            )

        return self.select(
            query_id=str(query_id),
            candidates=candidates,
            semantic_k_hint=query_plan["semantic_k_hint"],
        )


# ============================================================================
# JSONL streaming
# ============================================================================


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Stream JSONL records one line at a time.

    The complete file is never loaded into RAM.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_number}; "
                    f"got {type(record).__name__}"
                )

            yield record


def _get_query_id(record: Mapping[str, Any], source_name: str) -> str:
    """
    Extract query ID from a JSON record.

    R4 accepts both:
        id
        query_id

    R3 is expected to use query_id.
    """
    query_id = record.get("query_id")

    if query_id is None:
        query_id = record.get("id")

    if query_id is None:
        raise KeyError(
            f"{source_name} record is missing 'id'/'query_id'"
        )

    return str(query_id)


def _get_r3_candidates(record: Mapping[str, Any]) -> Sequence[Any]:
    """
    Extract R3 candidate list.
    """
    if "candidates" not in record:
        query_id = record.get("query_id", record.get("id", "<unknown>"))

        raise KeyError(
            f"R3 query {query_id!r} is missing 'candidates'"
        )

    candidates = record["candidates"]

    if candidates is None:
        query_id = record.get("query_id", record.get("id", "<unknown>"))

        raise ValueError(
            f"R3 query {query_id!r} has candidates=None"
        )

    if not isinstance(candidates, Sequence):
        query_id = record.get("query_id", record.get("id", "<unknown>"))

        raise TypeError(
            f"R3 query {query_id!r} candidates must be a sequence; "
            f"got {type(candidates).__name__}"
        )

    return candidates


def run_jsonl_funnel(
    s4_path: Path,
    r3_path: Path,
    output_path: Path,
    config: Optional[AdaptiveKConfig] = None,
) -> Iterator[AdaptiveKResult]:
    """
    Stream S4 + R3 through QA-R4.

    Memory guarantee
    ----------------
    This function does NOT:
        - load the entire S4 file
        - load the entire R3 file
        - build a query_id -> candidate map
        - accumulate results in a list
        - duplicate selected candidate payloads in an internal list

    Instead:

        read S4 record
        read R3 record
        validate query_id alignment
        select candidates
        write output
        yield result
        repeat

    Therefore memory usage does not grow with the total number of queries.

    The caller receives one AdaptiveKResult at a time.

    Input synchronization
    ---------------------
    S4 and R3 are consumed in lockstep.

    We require:
        1. same query_id at every position
        2. same number of records

    Any mismatch raises InputDesyncError.

    Important:
        Because this function is a generator, the body is not executed until
        the caller starts iterating over it.
    """
    s4_path = Path(s4_path)
    r3_path = Path(r3_path)
    output_path = Path(output_path)

    engine = AdaptiveKEngine(config=config)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    s4_iter = iter_jsonl(s4_path)
    r3_iter = iter_jsonl(r3_path)

    with output_path.open("w", encoding="utf-8") as output_handle:
        record_index = 0

        while True:
            # --------------------------------------------------------------
            # Read one S4 record.
            # --------------------------------------------------------------
            try:
                s4_record = next(s4_iter)
                s4_has_record = True
            except StopIteration:
                s4_has_record = False

            # --------------------------------------------------------------
            # Read one R3 record.
            # --------------------------------------------------------------
            try:
                r3_record = next(r3_iter)
                r3_has_record = True
            except StopIteration:
                r3_has_record = False

            # --------------------------------------------------------------
            # Both streams ended -> successful completion.
            # --------------------------------------------------------------
            if not s4_has_record and not r3_has_record:
                break

            record_index += 1

            # --------------------------------------------------------------
            # S4 ended first.
            # --------------------------------------------------------------
            if not s4_has_record:
                r3_query_id = _get_query_id(
                    r3_record,
                    source_name="R3",
                )

                raise InputDesyncError(
                    "S4/R3 input desynchronization: "
                    f"S4 ended before R3 at record index {record_index}; "
                    f"R3 still has query_id={r3_query_id!r}"
                )

            # --------------------------------------------------------------
            # R3 ended first.
            # --------------------------------------------------------------
            if not r3_has_record:
                s4_query_id = _get_query_id(
                    s4_record,
                    source_name="S4",
                )

                raise InputDesyncError(
                    "S4/R3 input desynchronization: "
                    f"R3 ended before S4 at record index {record_index}; "
                    f"S4 still has query_id={s4_query_id!r}"
                )

            # --------------------------------------------------------------
            # Extract and compare IDs.
            # --------------------------------------------------------------
            s4_query_id = _get_query_id(
                s4_record,
                source_name="S4",
            )

            r3_query_id = _get_query_id(
                r3_record,
                source_name="R3",
            )

            if s4_query_id != r3_query_id:
                raise InputDesyncError(
                    "S4/R3 query order mismatch at "
                    f"record index {record_index}: "
                    f"S4 query_id={s4_query_id!r}, "
                    f"R3 query_id={r3_query_id!r}"
                )

            # --------------------------------------------------------------
            # S4 must contain semantic_k_hint.
            # --------------------------------------------------------------
            if "semantic_k_hint" not in s4_record:
                raise KeyError(
                    f"S4 query {s4_query_id!r} is missing "
                    "'semantic_k_hint'"
                )

            # --------------------------------------------------------------
            # R3 must contain candidates.
            # --------------------------------------------------------------
            candidates = _get_r3_candidates(r3_record)

            # --------------------------------------------------------------
            # Run R4 funnel.
            # --------------------------------------------------------------
            result = engine.select(
                query_id=s4_query_id,
                candidates=candidates,
                semantic_k_hint=s4_record["semantic_k_hint"],
            )

            # --------------------------------------------------------------
            # Normalize and serialize immediately.
            #
            # The full selected_candidates payload exists only for the
            # current query/result. It is not accumulated internally.
            # --------------------------------------------------------------
            serializable_result = _normalize_for_json(result)

            output_handle.write(
                json.dumps(
                    serializable_result,
                    ensure_ascii=False,
                    cls=R4JSONEncoder,
                    allow_nan=False,
                )
                + "\n"
            )

            # Make sure the record is physically handed off to the OS/file
            # buffer before yielding the current result.
            output_handle.flush()

            # --------------------------------------------------------------
            # STREAMING:
            #
            # Do NOT:
            #
            #     results.append(result)
            #
            # That would retain all selected_candidates from all previous
            # queries and defeat the memory guarantee.
            #
            # yield returns the current result to the caller immediately.
            # --------------------------------------------------------------
            yield result


# ============================================================================
# JSON serialization
# ============================================================================


def _normalize_for_json(value: Any) -> Any:
    """
    Recursively convert common scientific Python objects into JSON-safe data.

    Supported:
        - None
        - str
        - int
        - float
        - bool
        - dataclass
        - Mapping
        - list / tuple / set / frozenset
        - NumPy-like objects exposing tolist()
        - NumPy / Torch scalar-like objects exposing item()
        - Torch tensors exposing detach().cpu().tolist()
        - generic objects exposing __dict__

    Non-finite floats are rejected.
    """

    # ------------------------------------------------------------------
    # Native JSON primitives.
    # ------------------------------------------------------------------
    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"Cannot serialize non-finite float: {value!r}"
            )

        return value

    # ------------------------------------------------------------------
    # Dataclass.
    # ------------------------------------------------------------------
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize_for_json(
                getattr(value, field.name)
            )
            for field in fields(value)
        }

    # ------------------------------------------------------------------
    # Mapping.
    # ------------------------------------------------------------------
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_for_json(item)
            for key, item in value.items()
        }

    # ------------------------------------------------------------------
    # Sequences / sets.
    # ------------------------------------------------------------------
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _normalize_for_json(item)
            for item in value
        ]

    # ------------------------------------------------------------------
    # Torch tensor.
    #
    # We intentionally use duck typing so that adaptive_k.py does not
    # require torch as a dependency.
    # ------------------------------------------------------------------
    detach = getattr(value, "detach", None)

    if callable(detach):
        try:
            detached = detach()

            cpu = getattr(detached, "cpu", None)

            if callable(cpu):
                detached = cpu()

            tolist = getattr(detached, "tolist", None)

            if callable(tolist):
                return _normalize_for_json(tolist())

        except Exception:
            # Fall through to other serialization strategies.
            pass

    # ------------------------------------------------------------------
    # NumPy-like ndarray / scalar.
    # ------------------------------------------------------------------
    tolist = getattr(value, "tolist", None)

    if callable(tolist):
        try:
            return _normalize_for_json(tolist())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # NumPy scalar / other scalar-like objects.
    # ------------------------------------------------------------------
    item = getattr(value, "item", None)

    if callable(item):
        try:
            return _normalize_for_json(item())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Generic object with __dict__.
    # ------------------------------------------------------------------
    obj_dict = getattr(value, "__dict__", None)

    if isinstance(obj_dict, dict):
        return {
            str(key): _normalize_for_json(item)
            for key, item in obj_dict.items()
        }

    # ------------------------------------------------------------------
    # Final fallback.
    # ------------------------------------------------------------------
    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


class R4JSONEncoder(json.JSONEncoder):
    """
    JSON encoder for QA-R4 scientific / dataclass objects.

    _normalize_for_json() does the recursive heavy lifting; this encoder
    provides an additional safety net for objects passed directly to
    json.dumps().
    """

    def default(self, obj: Any) -> Any:
        try:
            return _normalize_for_json(obj)
        except (TypeError, ValueError):
            return super().default(obj)


# ============================================================================
# Lightweight self-checks
# ============================================================================


def _self_check() -> None:
    """
    Lightweight internal contract checks.

    These checks intentionally avoid depending on NumPy or Torch.
    """

    # ------------------------------------------------------------------
    # Configuration.
    # ------------------------------------------------------------------
    config = AdaptiveKConfig()

    assert config.allowed_k == (100, 300, 500)

    # ------------------------------------------------------------------
    # K=100.
    # ------------------------------------------------------------------
    engine = AdaptiveKEngine(config)

    candidates = [
        {"rank": i, "score": 1.0 / (i + 1)}
        for i in range(500)
    ]

    result_100 = engine.select(
        query_id="q100",
        candidates=candidates,
        semantic_k_hint=100,
    )

    assert result_100.k_requested == 100
    assert result_100.k_effective == 100
    assert result_100.k_available == 500
    assert len(result_100.selected_candidates) == 100
    assert result_100.status == "ok"

    # ------------------------------------------------------------------
    # K=300.
    # ------------------------------------------------------------------
    result_300 = engine.select(
        query_id="q300",
        candidates=candidates,
        semantic_k_hint=300,
    )

    assert result_300.k_requested == 300
    assert result_300.k_effective == 300
    assert len(result_300.selected_candidates) == 300

    # ------------------------------------------------------------------
    # K=500.
    # ------------------------------------------------------------------
    result_500 = engine.select(
        query_id="q500",
        candidates=candidates,
        semantic_k_hint=500,
    )

    assert result_500.k_requested == 500
    assert result_500.k_effective == 500
    assert len(result_500.selected_candidates) == 500

    # ------------------------------------------------------------------
    # R4 must preserve R3 ordering exactly.
    # ------------------------------------------------------------------
    assert (
        result_300.selected_candidates
        == tuple(candidates[:300])
    )

    # ------------------------------------------------------------------
    # R4 must not depend on uncertainty/modalities.
    # ------------------------------------------------------------------
    plan_a = {
        "id": "same",
        "semantic_k_hint": 100,
        "uncertainty": 0.01,
        "preferred_modalities": ["ocr"],
    }

    plan_b = {
        "id": "same",
        "semantic_k_hint": 100,
        "uncertainty": 0.99,
        "preferred_modalities": ["visual", "asr"],
    }

    a = engine.select_from_query_plan(
        plan_a,
        candidates,
    )

    b = engine.select_from_query_plan(
        plan_b,
        candidates,
    )

    assert a.selected_candidates == b.selected_candidates
    assert a.k_effective == b.k_effective

    # ------------------------------------------------------------------
    # Candidate pool below requested K.
    # ------------------------------------------------------------------
    short_pool = candidates[:20]

    short_result = engine.select(
        query_id="short",
        candidates=short_pool,
        semantic_k_hint=100,
    )

    assert short_result.k_requested == 100
    assert short_result.k_effective == 20
    assert short_result.k_available == 20
    assert short_result.status == "ok_with_warning"
    assert (
        short_result.fallback_reason
        == "candidate_pool_below_requested_k"
    )

    # ------------------------------------------------------------------
    # Empty candidate pool.
    # ------------------------------------------------------------------
    empty_result = engine.select(
        query_id="empty",
        candidates=[],
        semantic_k_hint=300,
    )

    assert empty_result.k_requested == 300
    assert empty_result.k_effective == 0
    assert empty_result.k_available == 0
    assert empty_result.status == "ok_with_warning"
    assert (
        empty_result.fallback_reason
        == "empty_candidate_pool"
    )

    # ------------------------------------------------------------------
    # Strict invalid K.
    # ------------------------------------------------------------------
    try:
        engine.select(
            query_id="invalid",
            candidates=candidates,
            semantic_k_hint=250,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "strict=True must reject unsupported semantic_k_hint"
        )

    # ------------------------------------------------------------------
    # Non-strict clamp must be observable.
    # ------------------------------------------------------------------
    non_strict_engine = AdaptiveKEngine(
        AdaptiveKConfig(
            allowed_k=(100, 300, 500),
            strict=False,
        )
    )

    clamped = non_strict_engine.select(
        query_id="clamped",
        candidates=candidates,
        semantic_k_hint=250,
    )

    assert clamped.k_requested == 250
    assert clamped.k_effective == 300
    assert clamped.status == "ok_with_warning"
    assert (
        clamped.fallback_reason
        == "clamped_invalid_semantic_k"
    )

    # ------------------------------------------------------------------
    # S4 id/query_id compatibility.
    # ------------------------------------------------------------------
    by_id = engine.select_from_query_plan(
        {
            "id": "q-id",
            "semantic_k_hint": 100,
        },
        candidates,
    )

    by_query_id = engine.select_from_query_plan(
        {
            "query_id": "q-query-id",
            "semantic_k_hint": 100,
        },
        candidates,
    )

    assert by_id.query_id == "q-id"
    assert by_query_id.query_id == "q-query-id"

    # ------------------------------------------------------------------
    # Missing semantic_k_hint.
    # ------------------------------------------------------------------
    try:
        engine.select_from_query_plan(
            {
                "id": "missing-k",
            },
            candidates,
        )
    except KeyError:
        pass
    else:
        raise AssertionError(
            "Missing semantic_k_hint must raise KeyError"
        )

    # ------------------------------------------------------------------
    # Fake NumPy-like scalar.
    # ------------------------------------------------------------------
    class FakeScalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    fake_result = {
        "score": FakeScalar(0.5),
        "nested": [
            FakeScalar(0.25),
            {"x": FakeScalar(1.0)},
        ],
    }

    normalized = _normalize_for_json(fake_result)

    assert normalized == {
        "score": 0.5,
        "nested": [
            0.25,
            {"x": 1.0},
        ],
    }

    # ------------------------------------------------------------------
    # Dataclass serialization.
    # ------------------------------------------------------------------
    serialized = _normalize_for_json(result_100)

    assert serialized["query_id"] == "q100"
    assert serialized["k_requested"] == 100
    assert serialized["k_effective"] == 100
    assert len(serialized["selected_candidates"]) == 100

    # ------------------------------------------------------------------
    # JSON dumps must succeed.
    # ------------------------------------------------------------------
    payload = json.dumps(
        result_100,
        cls=R4JSONEncoder,
        ensure_ascii=False,
        allow_nan=False,
    )

    assert '"query_id": "q100"' in payload

    # ------------------------------------------------------------------
    # No uncertainty / margin logic exists in result.
    # ------------------------------------------------------------------
    assert not hasattr(result_100, "uncertainty")
    assert not hasattr(result_100, "normalized_margin")


if __name__ == "__main__":
    _self_check()
    print("QA-R4 adaptive_k self-check: PASS")