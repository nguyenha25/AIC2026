"""
QA-R4 — Adaptive K / Semantic Funnel
====================================

Contract
--------
QA-R4 receives exactly two JSONL streams:

    S4:
        query_id / id
        semantic_k_hint

    R3:
        query_id
        candidates

R4 converts the retrieval budget into a small Reader/VLM budget and performs:

    reader_k = reader_k_by_semantic_k[semantic_k_hint]
    candidates[:reader_k]

It does NOT:

    - compute uncertainty
    - compute entropy
    - compute normalized margin
    - rerank candidates
    - deduplicate candidates
    - alter R3 ordering
    - load the entire R3 stream into RAM

The semantic_k_hint is a retrieval-budget signal. It is never sent directly to
the Reader/VLM as its frame count. The default Reader policy is:

    100 -> 4 frames
    300 -> 8 frames
    500 -> 12 frames

Default allowed K:

    (100, 300, 500)

JSONL processing is pairwise/streaming and output is written atomically.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# ============================================================================
# Exceptions
# ============================================================================


class InputDesyncError(RuntimeError):
    """Raised when S4 and R3 JSONL streams are not synchronized."""


# ============================================================================
# Result
# ============================================================================


@dataclass(frozen=True)
class AdaptiveKResult:
    """
    Result of one QA-R4 selection.

    Notes
    -----
    k_requested:
        Original semantic_k_hint requested by S4.

    reader_k_requested:
        Reader/VLM frame budget resolved from semantic_k_hint.

    k_effective:
        Actual number of candidates returned.

    k_available:
        Number of candidates supplied by R3.

    selected_candidates:
        Exact prefix of R3 candidates. No reranking/deduplication.
    """

    query_id: str
    k_requested: int
    reader_k_requested: int
    k_effective: int
    k_available: int
    selected_candidates: Tuple[Dict[str, Any], ...]
    status: str
    fallback_reason: Optional[str] = None


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class AdaptiveKConfig:
    """
    QA-R4 configuration.

    allowed_k:
        Supported semantic K values.

    strict:
        If True, semantic_k_hint must belong to allowed_k.
        If False, invalid values are clamped to the nearest allowed K.

    reader_k_by_semantic_k:
        Explicit mapping from retrieval budget to Reader/VLM frame budget.
    """

    allowed_k: Tuple[int, ...] = (100, 300, 500)
    reader_k_by_semantic_k: Tuple[Tuple[int, int], ...] = (
        (100, 4),
        (300, 8),
        (500, 12),
    )
    strict: bool = True

    def __post_init__(self) -> None:
        normalized = self._normalize_allowed_k(self.allowed_k)
        object.__setattr__(self, "allowed_k", normalized)

        reader_policy = self._normalize_reader_policy(
            self.reader_k_by_semantic_k
        )
        if tuple(key for key, _ in reader_policy) != normalized:
            raise ValueError(
                "reader_k_by_semantic_k keys must exactly match allowed_k"
            )
        object.__setattr__(
            self,
            "reader_k_by_semantic_k",
            reader_policy,
        )

        if not isinstance(self.strict, bool):
            raise ValueError("strict must be a boolean")

    @staticmethod
    def _normalize_allowed_k(
        values: Iterable[int],
    ) -> Tuple[int, ...]:
        if values is None:
            raise ValueError("allowed_k must not be None")

        try:
            values = tuple(values)
        except TypeError as exc:
            raise ValueError(
                "allowed_k must be an iterable of positive integers"
            ) from exc

        if not values:
            raise ValueError("allowed_k must not be empty")

        normalized: List[int] = []

        for value in values:
            if isinstance(value, bool):
                raise ValueError(
                    "allowed_k must contain positive integers; bool is invalid"
                )

            try:
                integer_value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid allowed_k value: {value!r}"
                ) from exc

            # Do not silently turn 100.5 into 100.
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(
                    f"allowed_k values must be integers: {value!r}"
                )

            if integer_value <= 0:
                raise ValueError(
                    f"allowed_k values must be positive: {value!r}"
                )

            normalized.append(integer_value)

        return tuple(sorted(set(normalized)))

    @staticmethod
    def _normalize_reader_policy(
        values: Iterable[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], ...]:
        try:
            pairs = tuple(values)
        except TypeError as exc:
            raise ValueError(
                "reader_k_by_semantic_k must be an iterable of pairs"
            ) from exc

        if not pairs:
            raise ValueError("reader_k_by_semantic_k must not be empty")

        normalized: Dict[int, int] = {}
        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(
                    "reader_k_by_semantic_k must contain (semantic_k, reader_k) pairs"
                )
            semantic_k, reader_k = pair
            if isinstance(semantic_k, bool) or isinstance(reader_k, bool):
                raise ValueError("reader K policy values must be integers")
            try:
                semantic_k = int(semantic_k)
                reader_k = int(reader_k)
            except (TypeError, ValueError) as exc:
                raise ValueError("reader K policy values must be integers") from exc
            if semantic_k <= 0 or reader_k <= 0:
                raise ValueError("reader K policy values must be positive")
            if semantic_k in normalized:
                raise ValueError(
                    f"duplicate semantic K in reader policy: {semantic_k}"
                )
            normalized[semantic_k] = reader_k

        return tuple(sorted(normalized.items()))


# ============================================================================
# JSON normalization
# ============================================================================


def _normalize_for_json(value: Any) -> Any:
    """
    Convert common Python / NumPy / Torch-like values into JSON-safe values.

    Supported:
        - dataclasses
        - None
        - bool
        - int
        - float
        - str
        - Mapping
        - tuple
        - list
        - torch-like tensors
        - NumPy-like arrays/scalars

    Non-finite floats are rejected because JSON output must use
    allow_nan=False semantics.
    """

    # ------------------------------------------------------------------
    # Dataclass
    # ------------------------------------------------------------------
    #
    # AdaptiveKResult is a dataclass. Convert it recursively to a dict
    # before applying the remaining JSON normalization rules.
    #
    if is_dataclass(value):
        return _normalize_for_json(
            asdict(value)
        )

    # ------------------------------------------------------------------
    # Native JSON-safe scalar types
    # ------------------------------------------------------------------

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite float cannot be serialized: {value!r}"
            )

        return value

    if isinstance(value, str):
        return value

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    if isinstance(value, Mapping):
        return {
            str(key): _normalize_for_json(item)
            for key, item in value.items()
        }

    # ------------------------------------------------------------------
    # Tuple / list
    # ------------------------------------------------------------------

    if isinstance(value, tuple):
        return [
            _normalize_for_json(item)
            for item in value
        ]

    if isinstance(value, list):
        return [
            _normalize_for_json(item)
            for item in value
        ]

    # ------------------------------------------------------------------
    # Torch-like tensor
    # ------------------------------------------------------------------

    detach = getattr(value, "detach", None)

    if callable(detach):
        detached = detach()

        cpu = getattr(detached, "cpu", None)

        if callable(cpu):
            detached = cpu()

        tolist = getattr(detached, "tolist", None)

        if callable(tolist):
            return _normalize_for_json(
                tolist()
            )

    # ------------------------------------------------------------------
    # NumPy-like array
    # ------------------------------------------------------------------

    tolist = getattr(value, "tolist", None)

    if callable(tolist):
        return _normalize_for_json(
            tolist()
        )

    # ------------------------------------------------------------------
    # NumPy scalar-like
    # ------------------------------------------------------------------

    item = getattr(value, "item", None)

    if callable(item):
        return _normalize_for_json(
            item()
        )

    # ------------------------------------------------------------------
    # Last resort
    # ------------------------------------------------------------------

    try:
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"object of type {type(value).__name__!r} "
            "is not JSON serializable"
        ) from exc

    return value


class R4JSONEncoder(json.JSONEncoder):
    """JSON encoder used by QA-R4."""

    def default(self, obj: Any) -> Any:
        return _normalize_for_json(obj)


# ============================================================================
# Validation helpers
# ============================================================================


_REQUIRED_CANDIDATE_FIELDS = (
    "video_id",
    "frame_id",
    "n",
    "score",
)


def _normalize_positive_k(value: Any) -> int:
    """
    Normalize semantic_k_hint.

    Accepted:
        100
        "100"

    Rejected:
        True
        False
        0
        negative
        non-integer numeric values
        arbitrary strings
    """

    if isinstance(value, bool):
        raise ValueError(
            "semantic_k_hint must be a positive integer; bool is invalid"
        )

    if isinstance(value, int):
        k = value

    elif isinstance(value, str):
        text = value.strip()

        if not text:
            raise ValueError(
                "semantic_k_hint must not be empty"
            )

        try:
            k = int(text)
        except ValueError as exc:
            raise ValueError(
                f"invalid semantic_k_hint: {value!r}"
            ) from exc

    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "semantic_k_hint must be finite"
            )

        if not value.is_integer():
            raise ValueError(
                "semantic_k_hint must be an integer"
            )

        k = int(value)

    else:
        try:
            k = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid semantic_k_hint: {value!r}"
            ) from exc

    if k <= 0:
        raise ValueError(
            "semantic_k_hint must be positive"
        )

    return k


def validate_candidate_schema(
    candidates: Sequence[Mapping[str, Any]],
    query_id: Optional[str] = None,
) -> None:
    """
    Validate every R3 candidate.

    Required fields:

        video_id
        frame_id
        n
        score

    Validation happens BEFORE slicing so that malformed candidates
    cannot silently hide behind the requested K prefix.
    """

    if candidates is None:
        raise ValueError(
            f"R3 query {query_id!r} candidates must not be None"
        )

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(
                f"R3 query {query_id!r} candidate[{index}] "
                "must be a mapping"
            )

        missing = [
            field
            for field in _REQUIRED_CANDIDATE_FIELDS
            if field not in candidate
        ]

        if missing:
            raise ValueError(
                f"R3 query {query_id!r} candidate[{index}] "
                f"is missing required fields: {missing}"
            )


def _extract_query_id(
    record: Mapping[str, Any],
) -> str:
    """
    Extract query ID.

    R3:
        query_id

    S4:
        query_id preferred, otherwise id
    """

    if "query_id" in record:
        query_id = record["query_id"]

    elif "id" in record:
        query_id = record["id"]

    else:
        raise KeyError(
            "record is missing query_id/id"
        )

    if query_id is None:
        raise KeyError(
            "record query_id/id must not be None"
        )

    query_id = str(query_id)

    if not query_id:
        raise KeyError(
            "record query_id/id must not be empty"
        )

    return query_id


# ============================================================================
# Engine
# ============================================================================


class AdaptiveKEngine:
    """
    QA-R4 semantic candidate funnel.

    The engine performs only:

        1. validate semantic_k_hint
        2. validate all candidate payloads
        3. resolve allowed semantic K
        4. map semantic K to Reader-K
        5. take candidates[:Reader-K]

    It never reranks or modifies candidate order.
    """

    def __init__(
        self,
        config: Optional[AdaptiveKConfig] = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else AdaptiveKConfig()
        )

    # ------------------------------------------------------------------
    # K resolution
    # ------------------------------------------------------------------

    def _resolve_k(
        self,
        semantic_k_hint: Any,
    ) -> Tuple[
        int,
        int,
        str,
        Optional[str],
    ]:
        """
        Return:

            k_requested
            k_effective_request
            status
            fallback_reason
        """

        k_requested = _normalize_positive_k(
            semantic_k_hint
        )

        allowed = self.config.allowed_k

        if k_requested in allowed:
            return (
                k_requested,
                k_requested,
                "ok",
                None,
            )

        if self.config.strict:
            raise ValueError(
                "semantic_k_hint "
                f"{k_requested} is not allowed; "
                f"allowed_k={allowed}"
            )

        # Clamp to nearest allowed K.
        #
        # For an exact midpoint, prefer the larger K.
        #
        # Example:
        #   allowed=(100,300,500)
        #   200 -> 300
        #
        k_clamped = min(
            allowed,
            key=lambda candidate_k: (
                abs(candidate_k - k_requested),
                -candidate_k,
            ),
        )

        return (
            k_requested,
            k_clamped,
            "ok_with_warning",
            "clamped_invalid_semantic_k",
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _reader_k_for(self, semantic_k: int) -> int:
        policy = dict(self.config.reader_k_by_semantic_k)
        try:
            return policy[semantic_k]
        except KeyError as exc:  # Protected by config validation.
            raise ValueError(
                f"no Reader-K policy for semantic_k_hint={semantic_k}"
            ) from exc

    def select(
        self,
        query_id: str,
        candidates: Sequence[Mapping[str, Any]],
        semantic_k_hint: Any,
    ) -> AdaptiveKResult:
        """
        Select an exact R3 prefix using the Reader-K policy.

        No reranking.
        No deduplication.
        No uncertainty.
        No mutation of candidate payloads.
        """

        if query_id is None:
            raise ValueError(
                "query_id must not be None"
            )

        query_id = str(query_id)

        if not query_id:
            raise ValueError(
                "query_id must not be empty"
            )

        if candidates is None:
            raise ValueError(
                f"R3 query {query_id!r} candidates must not be None"
            )

        # Materialize only the current query's candidate list.
        #
        # The JSONL integration itself remains streaming at query level.
        candidate_list = list(candidates)

        # Validate the COMPLETE candidate pool before slicing.
        validate_candidate_schema(
            candidate_list,
            query_id=query_id,
        )

        (
            k_requested,
            k_resolved,
            status,
            fallback_reason,
        ) = self._resolve_k(
            semantic_k_hint
        )

        reader_k_requested = self._reader_k_for(
            k_resolved
        )

        k_available = len(candidate_list)

        # --------------------------------------------------------------
        # Empty candidate pool
        # --------------------------------------------------------------

        if k_available == 0:
            return AdaptiveKResult(
                query_id=query_id,
                k_requested=k_requested,
                reader_k_requested=reader_k_requested,
                k_effective=0,
                k_available=0,
                selected_candidates=(),
                status="ok_with_warning",
                fallback_reason="empty_candidate_pool",
            )

        # --------------------------------------------------------------
        # Candidate pool smaller than requested K
        # --------------------------------------------------------------

        if k_available < reader_k_requested:
            selected = tuple(
                candidate_list[:k_available]
            )

            return AdaptiveKResult(
                query_id=query_id,
                k_requested=k_requested,
                reader_k_requested=reader_k_requested,
                k_effective=k_available,
                k_available=k_available,
                selected_candidates=selected,
                status="ok_with_warning",
                fallback_reason="candidate_pool_below_requested_k",
            )

        # --------------------------------------------------------------
        # Normal selection
        # --------------------------------------------------------------

        selected = tuple(
            candidate_list[:reader_k_requested]
        )

        return AdaptiveKResult(
            query_id=query_id,
            k_requested=k_requested,
            reader_k_requested=reader_k_requested,
            k_effective=reader_k_requested,
            k_available=k_available,
            selected_candidates=selected,
            status=status,
            fallback_reason=fallback_reason,
        )

    # ------------------------------------------------------------------
    # S4 compatibility
    # ------------------------------------------------------------------

    def select_from_query_plan(
        self,
        query_plan: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
    ) -> AdaptiveKResult:
        """
        Apply QA-R4 using an S4 QueryPlan-like mapping.

        Accepted query identifier fields:

            query_id
            id

        query_id has precedence when both exist.

        Required:

            semantic_k_hint

        Other S4 fields such as:

            uncertainty
            preferred_modalities
            normalized_margin

        are intentionally ignored.
        """

        if query_plan is None:
            raise ValueError(
                "query_plan must not be None"
            )

        if not isinstance(query_plan, Mapping):
            raise ValueError(
                "query_plan must be a mapping"
            )

        # query_id takes precedence over id.
        if "query_id" in query_plan:
            query_id = query_plan["query_id"]

        elif "id" in query_plan:
            query_id = query_plan["id"]

        else:
            raise KeyError(
                "query_plan is missing query_id/id"
            )

        if "semantic_k_hint" not in query_plan:
            raise KeyError(
                "query_plan is missing semantic_k_hint"
            )

        return self.select(
            query_id=str(query_id),
            candidates=candidates,
            semantic_k_hint=query_plan[
                "semantic_k_hint"
            ],
        )


# ============================================================================
# JSONL helpers
# ============================================================================


def _iter_jsonl(
    path: Path,
) -> Iterator[Dict[str, Any]]:
    """Stream JSON objects from a JSONL file."""

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path} "
                    f"at line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record in {path} "
                    f"at line {line_number} must be an object"
                )

            yield record


def _result_to_record(
    result: AdaptiveKResult,
) -> Dict[str, Any]:
    """Convert result dataclass to JSON-ready mapping."""

    return _normalize_for_json(
        asdict(result)
    )


def _atomic_jsonl_writer(
    output_path: Path,
) -> Tuple[Path, Any]:
    """
    Create a temporary output file in the same directory.

    The caller must close the file and replace the temporary path with
    the final path only after successful processing.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )

    handle = os.fdopen(
        fd,
        "w",
        encoding="utf-8",
        newline="\n",
    )

    return Path(tmp_name), handle


# ============================================================================
# Two-input streaming funnel
# ============================================================================


def run_jsonl_funnel(
    s4_path: Path,
    r3_path: Path,
    output_path: Path,
    engine: Optional[AdaptiveKEngine] = None,
) -> Iterator[AdaptiveKResult]:
    """
    Stream S4 and R3 JSONL pairwise.

    Synchronization contract:

        S4[q_i] <-> R3[q_i]

    The function is a generator.

    Output is written to a temporary file and atomically replaced only
    after the complete stream succeeds.

    Therefore, if processing fails midway, an existing output file is
    left untouched.
    """

    if engine is None:
        engine = AdaptiveKEngine()

    s4_path = Path(s4_path)
    r3_path = Path(r3_path)
    output_path = Path(output_path)

    def generator() -> Iterator[AdaptiveKResult]:
        tmp_path: Optional[Path] = None
        output_handle = None
        committed = False

        try:
            tmp_path, output_handle = _atomic_jsonl_writer(
                output_path
            )

            s4_iter = _iter_jsonl(s4_path)
            r3_iter = _iter_jsonl(r3_path)

            while True:
                # ------------------------------------------------------
                # Read one S4 record
                # ------------------------------------------------------

                try:
                    s4_record = next(s4_iter)
                    s4_done = False

                except StopIteration:
                    s4_done = True
                    s4_record = None

                # ------------------------------------------------------
                # Read one R3 record
                # ------------------------------------------------------

                try:
                    r3_record = next(r3_iter)
                    r3_done = False

                except StopIteration:
                    r3_done = True
                    r3_record = None

                # ------------------------------------------------------
                # Both streams ended simultaneously.
                # ------------------------------------------------------

                if s4_done and r3_done:
                    break

                # ------------------------------------------------------
                # One stream ended before the other.
                # ------------------------------------------------------

                if s4_done != r3_done:
                    raise InputDesyncError(
                        "S4 and R3 streams have different lengths"
                    )

                assert s4_record is not None
                assert r3_record is not None

                # ------------------------------------------------------
                # Extract and compare query IDs.
                # ------------------------------------------------------

                s4_query_id = _extract_query_id(
                    s4_record
                )

                r3_query_id = _extract_query_id(
                    r3_record
                )

                if s4_query_id != r3_query_id:
                    raise InputDesyncError(
                        "S4/R3 query order mismatch: "
                        f"S4={s4_query_id!r}, "
                        f"R3={r3_query_id!r}"
                    )

                # ------------------------------------------------------
                # Validate required fields.
                # ------------------------------------------------------

                if "semantic_k_hint" not in s4_record:
                    raise KeyError(
                        f"S4 query {s4_query_id!r} "
                        "is missing semantic_k_hint"
                    )

                if "candidates" not in r3_record:
                    raise KeyError(
                        f"R3 query {r3_query_id!r} "
                        "is missing candidates"
                    )

                candidates = r3_record["candidates"]

                if candidates is None:
                    raise ValueError(
                        f"R3 query {r3_query_id!r} "
                        "candidates must not be None"
                    )

                # ------------------------------------------------------
                # Run QA-R4 selection.
                # ------------------------------------------------------

                result = engine.select(
                    query_id=s4_query_id,
                    candidates=candidates,
                    semantic_k_hint=s4_record[
                        "semantic_k_hint"
                    ],
                )

                # ------------------------------------------------------
                # Serialize result.
                # ------------------------------------------------------

                record = _result_to_record(
                    result
                )

                output_handle.write(
                    json.dumps(
                        record,
                        cls=R4JSONEncoder,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

                # Flush each yielded record so streaming behavior is
                # observable while the final output remains protected
                # by the temporary file.
                output_handle.flush()

                yield result

            # ----------------------------------------------------------
            # Complete stream succeeded.
            #
            # Only now commit the temporary file.
            # ----------------------------------------------------------

            output_handle.flush()

            os.fsync(
                output_handle.fileno()
            )

            output_handle.close()
            output_handle = None

            assert tmp_path is not None

            os.replace(
                tmp_path,
                output_path,
            )

            committed = True

        finally:
            # ----------------------------------------------------------
            # Close temporary output if still open.
            # ----------------------------------------------------------

            if output_handle is not None:
                try:
                    output_handle.close()
                except Exception:
                    pass

            # ----------------------------------------------------------
            # If processing failed before commit, remove only the temp
            # file. Never touch an existing final output.
            # ----------------------------------------------------------

            if (
                tmp_path is not None
                and not committed
                and tmp_path.exists()
            ):
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    return generator()


# ============================================================================
# Module self-check
# ============================================================================


def _self_check() -> None:
    """
    Lightweight executable self-check.

    Run:

        python scripts/adaptive_k.py
    """

    config = AdaptiveKConfig()

    assert config.allowed_k == (
        100,
        300,
        500,
    )

    assert config.strict is True
    assert config.reader_k_by_semantic_k == (
        (100, 4),
        (300, 8),
        (500, 12),
    )

    engine = AdaptiveKEngine(config)

    candidates = [
        {
            "rank": i,
            "video_id": f"video_{i}",
            "frame_id": f"frame_{i}",
            "n": i,
            "score": 1.0 / (i + 1),
        }
        for i in range(500)
    ]

    # --------------------------------------------------------------
    # Normal selection
    # --------------------------------------------------------------

    result = engine.select(
        query_id="self-check",
        candidates=candidates,
        semantic_k_hint=100,
    )

    assert result.k_requested == 100
    assert result.reader_k_requested == 4
    assert result.k_effective == 4
    assert result.k_available == 500

    assert len(
        result.selected_candidates
    ) == 4

    assert (
        result.selected_candidates
        == tuple(candidates[:4])
    )

    # --------------------------------------------------------------
    # QueryPlan compatibility
    # --------------------------------------------------------------

    query_plan_result = engine.select_from_query_plan(
        {
            "id": "self-check-query-plan",
            "semantic_k_hint": 300,
            "uncertainty": 0.99,
        },
        candidates,
    )

    assert (
        query_plan_result.query_id
        == "self-check-query-plan"
    )

    assert (
        query_plan_result.k_effective
        == 8
    )

    # --------------------------------------------------------------
    # Non-strict clamping
    # --------------------------------------------------------------

    non_strict = AdaptiveKEngine(
        AdaptiveKConfig(
            allowed_k=(100, 300, 500),
            strict=False,
        )
    )

    clamped = non_strict.select(
        query_id="self-check-clamp",
        candidates=candidates,
        semantic_k_hint=250,
    )

    assert clamped.k_requested == 250
    assert clamped.reader_k_requested == 8
    assert clamped.k_effective == 8

    assert (
        clamped.status
        == "ok_with_warning"
    )

    assert (
        clamped.fallback_reason
        == "clamped_invalid_semantic_k"
    )

    # --------------------------------------------------------------
    # Dataclass serialization
    # --------------------------------------------------------------

    normalized = _normalize_for_json(
        result
    )

    assert isinstance(
        normalized,
        dict,
    )

    assert normalized["query_id"] == "self-check"

    encoded = json.dumps(
        result,
        cls=R4JSONEncoder,
        ensure_ascii=False,
        allow_nan=False,
    )

    decoded = json.loads(encoded)

    assert decoded["query_id"] == "self-check"
    assert decoded["k_requested"] == 100
    assert decoded["reader_k_requested"] == 4
    assert decoded["k_effective"] == 4

    print(
        "QA-R4 adaptive_k self-check: PASS"
    )


if __name__ == "__main__":
    _self_check()
