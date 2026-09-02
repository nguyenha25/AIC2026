"""
QA-R4 — Adaptive K Benchmark
============================

Benchmark thin funnel:

    QA-S4 semantic parsing
            +
    QA-R3 fused candidates
            |
            v
          QA-R4
            |
            v
    adaptive candidate output

QA-R4 benchmark scope:
    - task == "qa" only
    - S4 may contain both QA and TRAKE
    - TRAKE records are excluded because they do not have semantic_k_hint

IMPORTANT
---------
QA-S4 and QA-R3 may contain the same query IDs in different orders,
AND R3 may additionally contain query IDs (e.g. TRAKE) that do not
exist in the requested S4 task at all.

`run_jsonl_funnel()` (adaptive_k.py) is a STRICT POSITIONAL streaming
funnel: it reads S4 and R3 line-by-line in lockstep and expects
line i of S4 and line i of R3 to refer to the SAME query_id. It does
NOT build an in-memory query_id -> record map.

Therefore this benchmark must align BOTH sides before handing them to
the funnel:

    1. reads R3 query order (as-is, may contain QA + TRAKE)
    2. extracts QA records from S4
    3. intersects R3 order with the S4 QA query IDs, preserving R3
       order -> r3_order_task
    4. reorders S4 to exactly match r3_order_task
    5. reorders/filters R3 to exactly match r3_order_task
    6. feeds the two aligned temporary files into the strict R4 funnel

This does NOT:
    - modify original S4
    - modify original R3
    - modify adaptive_k.py
    - rerank R3 candidates
    - recompute uncertainty
    - recompute entropy
    - evaluate VLM answer quality

R4 behavior:
    candidates[:semantic_k_hint]

Benchmark timing contract:
    The benchmark timer starts immediately before R4 funnel execution
    and ends only after the funnel has completely finished producing
    and committing the output.

Therefore measured runtime includes:
    - QA-R4 adaptive K selection
    - candidate/result serialization
    - JSONL output writing
    - atomic output commit

Alignment is excluded because it is benchmark setup rather than
QA-R4 execution.

`validate_result()` is benchmark verification/bookkeeping and is not
used to define a separate "selection runtime".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.adaptive_k import (
    AdaptiveKResult,
    InputDesyncError,
    R4JSONEncoder,
    run_jsonl_funnel,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_S4 = Path(
    r"D:\aic-data\runs\n01_semantic_parsing.jsonl"
)

DEFAULT_R3_CANDIDATE_FILES = (
    # Current / preferred QA-R3 export.
    Path(r"D:\aic-data\runs\qa_r3_candidates.json"),

    # Existing JSONL variants.
    Path(r"D:\aic-data\runs\r3_fused_candidates.jsonl"),
    Path(r"D:\aic-data\runs\r3_candidates.jsonl"),
    Path(r"D:\aic-data\runs\qa_r3_fused_candidates.jsonl"),
)

DEFAULT_OUTPUT = Path(
    r"D:\aic-data\runs\qa_r4_adaptive_candidates.jsonl"
)

DEFAULT_REPORT = Path(
    r"D:\aic-data\runs\qa_r4_benchmark.json"
)

DEFAULT_TASK = "qa"

# This benchmark reports atomic_output=True only because the R4 funnel
# contract requires atomic output commit. The underlying implementation
# must use an atomic temporary-file -> os.replace() workflow.
ATOMIC_OUTPUT_CONTRACT = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def finite_number(value: Any) -> bool:
    """Return True only for finite int/float values."""
    if isinstance(value, bool):
        return False

    if isinstance(value, (int, float)):
        return math.isfinite(float(value))

    return False


def percentile(
    values: List[float],
    p: float,
) -> Optional[float]:
    """
    Simple linear-interpolated percentile.

    Returns None for an empty list.
    """
    if not values:
        return None

    if len(values) == 1:
        return float(values[0])

    values = sorted(values)

    rank = (len(values) - 1) * p
    low = int(math.floor(rank))
    high = int(math.ceil(rank))

    if low == high:
        return float(values[low])

    fraction = rank - low

    return (
        float(values[low])
        + fraction * (
            float(values[high])
            - float(values[low])
        )
    )


def resolve_r3_input(
    explicit_path: Optional[Path],
) -> Path:
    """
    Resolve R3 candidate file.

    If --r3 is supplied, use it directly.

    Otherwise search known default locations and require exactly one
    existing candidate.
    """
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(
                f"R3 input does not exist: {explicit_path}"
            )

        return explicit_path

    existing = [
        path
        for path in DEFAULT_R3_CANDIDATE_FILES
        if path.exists()
    ]

    if not existing:
        candidates = "\n".join(
            f"  - {path}"
            for path in DEFAULT_R3_CANDIDATE_FILES
        )

        raise FileNotFoundError(
            "Could not find an R3 candidate file automatically.\n"
            "Use --r3 explicitly.\n"
            f"Checked:\n{candidates}"
        )

    if len(existing) > 1:
        candidates = "\n".join(
            f"  - {path}"
            for path in existing
        )

        raise RuntimeError(
            "Multiple possible R3 candidate files were found.\n"
            "Use --r3 explicitly to avoid benchmarking "
            "the wrong file.\n"
            f"Found:\n{candidates}"
        )

    return existing[0]


def validate_input_paths(
    s4_path: Path,
    r3_path: Path,
    output_path: Path,
    report_path: Path,
) -> None:
    """Validate benchmark paths before execution."""
    if not s4_path.exists():
        raise FileNotFoundError(
            f"S4 input does not exist: {s4_path}"
        )

    if not r3_path.exists():
        raise FileNotFoundError(
            f"R3 input does not exist: {r3_path}"
        )

    s4_resolved = s4_path.resolve()
    r3_resolved = r3_path.resolve()
    output_resolved = output_path.resolve()
    report_resolved = report_path.resolve()

    if output_resolved in {
        s4_resolved,
        r3_resolved,
    }:
        raise ValueError(
            "Output path must not overwrite an input file."
        )

    if report_resolved in {
        s4_resolved,
        r3_resolved,
    }:
        raise ValueError(
            "Report path must not overwrite an input file."
        )

    if output_resolved == report_resolved:
        raise ValueError(
            "Output JSONL and report JSON must use different paths."
        )


def ensure_parent(path: Path) -> None:
    """Create parent directory if necessary."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def _make_temp_jsonl(prefix: str) -> Path:
    """
    Create an empty temp file and return its Path.

    The OS-level descriptor returned by mkstemp() is explicitly closed
    before the path is reopened using normal Python file I/O.
    """
    fd, temp_name = tempfile.mkstemp(
        prefix=prefix,
        suffix=".jsonl",
    )

    os.close(fd)

    return Path(temp_name)


# ---------------------------------------------------------------------------
# R3 input loading
# ---------------------------------------------------------------------------

def _read_r3_records(
    r3_path: Path,
) -> List[Dict[str, Any]]:
    """
    Read R3 input.

    Supported formats:

    1. JSON ARRAY

           [
             {"query_id": "04", "candidates": [...]},
             {"query_id": "05", "candidates": [...]}
           ]

    2. SINGLE JSON OBJECT

           {"query_id": "04", "candidates": [...]}

    3. JSONL

           {"query_id": "04", "candidates": [...]}
           {"query_id": "05", "candidates": [...]}

    Format is detected from content rather than filename extension.
    """
    text = r3_path.read_text(
        encoding="utf-8-sig"
    )

    stripped = text.lstrip()

    if not stripped:
        raise ValueError(
            f"R3 input trống: {r3_path}"
        )

    if stripped[0] in "[{":
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            document = None

        if isinstance(document, list):
            records: List[Dict[str, Any]] = []

            for index, record in enumerate(
                document
            ):
                if not isinstance(record, dict):
                    raise ValueError(
                        f"R3 array item [{index}] phải là "
                        "JSON object; "
                        f"got {type(record).__name__}."
                    )

                records.append(record)

            return records

        if isinstance(document, dict):
            return [document]

        # If the complete document cannot be parsed, fall back
        # to JSONL parsing.
        records = []

        for line_number, raw_line in enumerate(
            text.splitlines(),
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in R3 at line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"R3 line {line_number} must be a JSON "
                    "object; "
                    f"got {type(record).__name__}"
                )

            records.append(record)

        if not records:
            raise ValueError(
                f"Không đọc được record nào từ R3 input: "
                f"{r3_path}."
            )

        return records

    raise ValueError(
        f"Không nhận diện được format R3 input: {r3_path}"
    )


# ---------------------------------------------------------------------------
# R3 query-order loading
# ---------------------------------------------------------------------------

def load_r3_query_order(
    r3_path: Path,
) -> tuple[List[str], Counter]:
    """
    Read R3 query IDs in their actual record order.

    R3 is the authoritative order for the lockstep R4 funnel.

    Returns:
        query_order
        candidate_count_by_query
    """
    records = _read_r3_records(r3_path)

    query_order: List[str] = []
    candidate_count_by_query: Counter = Counter()

    for index, record in enumerate(
        records,
        start=1,
    ):
        if "query_id" not in record:
            raise KeyError(
                f"R3 record #{index} is missing 'query_id'."
            )

        query_id = str(
            record["query_id"]
        )

        if query_id in candidate_count_by_query:
            raise ValueError(
                f"Duplicate R3 query_id={query_id!r} "
                f"at record #{index}."
            )

        candidates = record.get("candidates")

        if not isinstance(candidates, list):
            raise ValueError(
                f"R3 query {query_id!r} has invalid "
                "'candidates'; expected list."
            )

        query_order.append(query_id)
        candidate_count_by_query[query_id] = len(
            candidates
        )

    if not query_order:
        raise ValueError(
            f"R3 input contains no query records: "
            f"{r3_path}"
        )

    return (
        query_order,
        candidate_count_by_query,
    )


def _load_r3_raw_lines_by_query_id(
    r3_path: Path,
) -> Dict[str, str]:
    """
    Return query_id -> compact JSON text for each R3 record.

    This works for JSON arrays, single JSON objects, and JSONL.
    """
    records = _read_r3_records(r3_path)

    result: Dict[str, str] = {}

    for index, record in enumerate(
        records,
        start=1,
    ):
        if "query_id" not in record:
            raise KeyError(
                f"R3 record #{index} is missing 'query_id'."
            )

        query_id = str(
            record["query_id"]
        )

        if query_id in result:
            raise ValueError(
                f"Duplicate R3 query_id={query_id!r} "
                f"at record #{index}."
            )

        result[query_id] = json.dumps(
            record,
            ensure_ascii=False,
        )

    return result


# ---------------------------------------------------------------------------
# S4/R3 alignment
# ---------------------------------------------------------------------------

def build_aligned_inputs(
    s4_path: Path,
    r3_path: Path,
    task: str,
) -> tuple[Path, Path, Dict[str, Any]]:
    """
    Build temporary S4 and R3 JSONL files that are both restricted
    and reordered to the exact same query_id sequence.

    R3 order is authoritative.
    """
    (
        r3_order_all,
        r3_candidate_counts_all,
    ) = load_r3_query_order(
        r3_path
    )

    # --------------------------------------------------------------
    # Read S4 and keep only the requested task.
    # --------------------------------------------------------------
    s4_records: Dict[
        str,
        Dict[str, Any],
    ] = {}

    task_counts: Counter = Counter()

    with s4_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in S4 at line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"S4 line {line_number} must be a JSON "
                    "object; "
                    f"got {type(record).__name__}"
                )

            record_task = str(
                record.get("task", "")
            ).lower()

            task_counts[record_task] += 1

            if record_task != task.lower():
                continue

            if (
                "query_id" not in record
                and "id" not in record
            ):
                raise KeyError(
                    f"S4 line {line_number} has neither "
                    "'query_id' nor 'id'."
                )

            query_id = str(
                record.get(
                    "query_id",
                    record.get("id"),
                )
            )

            if query_id in s4_records:
                raise ValueError(
                    f"Duplicate S4 {task} "
                    f"query_id={query_id!r} "
                    f"at line {line_number}."
                )

            if "semantic_k_hint" not in record:
                raise KeyError(
                    f"S4 {task!r} query {query_id!r} "
                    "is missing 'semantic_k_hint'."
                )

            s4_records[query_id] = record

    if not s4_records:
        raise ValueError(
            f"No S4 records found for task={task!r}."
        )

    # --------------------------------------------------------------
    # Intersect R3 order with S4 task IDs.
    # --------------------------------------------------------------
    s4_task_ids = set(
        s4_records
    )

    r3_order_task = [
        query_id
        for query_id in r3_order_all
        if query_id in s4_task_ids
    ]

    r3_candidate_counts = {
        query_id: r3_candidate_counts_all[query_id]
        for query_id in r3_order_task
    }

    # --------------------------------------------------------------
    # Every requested-task S4 query must exist in R3.
    # --------------------------------------------------------------
    r3_task_id_set = set(
        r3_order_task
    )

    missing_in_r3 = [
        query_id
        for query_id in s4_records
        if query_id not in r3_task_id_set
    ]

    if missing_in_r3:
        raise ValueError(
            f"S4 {task} contains query IDs missing "
            f"from R3: "
            f"{sorted(missing_in_r3, key=str)}"
        )

    if len(s4_records) != len(r3_order_task):
        raise ValueError(
            "S4/R3 query count mismatch after task "
            "filtering: "
            f"S4 {task}={len(s4_records)}, "
            f"R3 matching {task}={len(r3_order_task)}"
        )

    excluded_r3_ids = [
        query_id
        for query_id in r3_order_all
        if query_id not in s4_task_ids
    ]

    # --------------------------------------------------------------
    # Write aligned S4.
    # --------------------------------------------------------------
    temp_s4_path = _make_temp_jsonl(
        prefix="qa_r4_s4_aligned_"
    )

    try:
        with temp_s4_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output:
            for query_id in r3_order_task:
                record = s4_records[
                    query_id
                ]

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        cls=R4JSONEncoder,
                        allow_nan=False,
                    )
                    + "\n"
                )

    except Exception:
        try:
            temp_s4_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        raise

    # --------------------------------------------------------------
    # Write aligned R3.
    # --------------------------------------------------------------
    r3_raw_lines = (
        _load_r3_raw_lines_by_query_id(
            r3_path
        )
    )

    temp_r3_path = _make_temp_jsonl(
        prefix="qa_r4_r3_aligned_"
    )

    try:
        with temp_r3_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output:
            for query_id in r3_order_task:
                output.write(
                    r3_raw_lines[
                        query_id
                    ]
                    + "\n"
                )

    except Exception:
        try:
            temp_r3_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        try:
            temp_s4_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        raise

    alignment_report = {
        "task": task,

        "s4_total_records_by_task": dict(
            sorted(
                task_counts.items()
            )
        ),

        "s4_task_query_count": len(
            s4_records
        ),

        "r3_total_query_count": len(
            r3_order_all
        ),

        "r3_matching_task_query_count": len(
            r3_order_task
        ),

        "r3_excluded_non_task_query_count": len(
            excluded_r3_ids
        ),

        "r3_excluded_non_task_query_ids": (
            excluded_r3_ids
        ),

        "query_order_source": "R3",

        "query_order_aligned": True,

        "r3_restricted_to_task": True,

        "query_order": r3_order_task,

        "candidate_count_by_query": {
            query_id: r3_candidate_counts[
                query_id
            ]
            for query_id in r3_order_task
        },
    }

    return (
        temp_s4_path,
        temp_r3_path,
        alignment_report,
    )


# ---------------------------------------------------------------------------
# Benchmark statistics
# ---------------------------------------------------------------------------

class R4BenchmarkStats:
    """Streaming benchmark accumulator."""

    def __init__(self) -> None:
        self.query_count = 0

        self.requested_k = Counter()
        self.effective_k = Counter()

        self.status = Counter()
        self.fallback_reason = Counter()

        self.candidate_pool_below_requested = 0
        self.empty_candidate_pool = 0

        self.selected_candidate_total = 0

        self.available_candidates: List[int] = []
        self.selection_ratios: List[float] = []

        # Per-query timing is intentionally not reported.
        #
        # The authoritative benchmark timing is the complete funnel
        # runtime measured around the streaming generator:
        #
        # selection + serialization + output write + atomic commit.
        self.elapsed_seconds: List[float] = []

        self.first_query_id: Optional[str] = None
        self.last_query_id: Optional[str] = None

    def update(
        self,
        result: AdaptiveKResult,
    ) -> None:
        """Update aggregate statistics with one R4 result."""
        self.query_count += 1

        query_id = str(
            result.query_id
        )

        if self.first_query_id is None:
            self.first_query_id = query_id

        self.last_query_id = query_id

        self.requested_k[
            result.k_requested
        ] += 1

        self.effective_k[
            result.k_effective
        ] += 1

        self.status[
            result.status
        ] += 1

        if result.fallback_reason:
            self.fallback_reason[
                result.fallback_reason
            ] += 1

        if (
            result.fallback_reason
            == "candidate_pool_below_requested_k"
        ):
            self.candidate_pool_below_requested += 1

        if (
            result.fallback_reason
            == "empty_candidate_pool"
        ):
            self.empty_candidate_pool += 1

        selected_count = len(
            result.selected_candidates
        )

        self.selected_candidate_total += (
            selected_count
        )

        self.available_candidates.append(
            result.k_available
        )

        if result.k_requested > 0:
            self.selection_ratios.append(
                selected_count
                / result.k_requested
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize benchmark statistics."""
        total_selected = (
            self.selected_candidate_total
        )

        return {
            "query_count": self.query_count,

            "first_query_id": self.first_query_id,

            "last_query_id": self.last_query_id,

            "requested_k_distribution": {
                str(k): count
                for k, count in sorted(
                    self.requested_k.items()
                )
            },

            "effective_k_distribution": {
                str(k): count
                for k, count in sorted(
                    self.effective_k.items()
                )
            },

            "status_distribution": dict(
                sorted(
                    self.status.items()
                )
            ),

            "fallback_reason_distribution": dict(
                sorted(
                    self.fallback_reason.items()
                )
            ),

            "candidate_pool_below_requested_k": (
                self.candidate_pool_below_requested
            ),

            "empty_candidate_pool": (
                self.empty_candidate_pool
            ),

            "selected_candidate_total": (
                total_selected
            ),

            "mean_selected_candidates": (
                total_selected / self.query_count
                if self.query_count
                else 0.0
            ),

            "candidate_availability": {
                "min": (
                    min(
                        self.available_candidates
                    )
                    if self.available_candidates
                    else None
                ),

                "p50": percentile(
                    [
                        float(x)
                        for x in self.available_candidates
                    ],
                    0.50,
                ),

                "p95": percentile(
                    [
                        float(x)
                        for x in self.available_candidates
                    ],
                    0.95,
                ),

                "max": (
                    max(
                        self.available_candidates
                    )
                    if self.available_candidates
                    else None
                ),
            },

            "selection_ratio": {
                "mean": (
                    statistics.fmean(
                        self.selection_ratios
                    )
                    if self.selection_ratios
                    else None
                ),

                "min": (
                    min(
                        self.selection_ratios
                    )
                    if self.selection_ratios
                    else None
                ),

                "p50": percentile(
                    self.selection_ratios,
                    0.50,
                ),

                "p95": percentile(
                    self.selection_ratios,
                    0.95,
                ),
            },

            "runtime_seconds": {
                "total": None,
                "mean_per_query": None,
                "p50_per_query": None,
                "p95_per_query": None,
                "max_per_query": None,
                "scope": (
                    "full_r4_funnel: "
                    "adaptive K selection + serialization "
                    "+ JSONL write + atomic commit"
                ),
            },
        }


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_result(
    result: AdaptiveKResult,
) -> None:
    """
    Validate one R4 result.

    Frozen contract:
        semantic_k_hint ∈ {100, 300, 500}
    """
    if not result.query_id:
        raise AssertionError(
            "R4 result has empty query_id."
        )

    if result.k_requested not in {
        100,
        300,
        500,
    }:
        raise AssertionError(
            f"Unexpected requested K: "
            f"{result.k_requested}"
        )

    if result.k_effective < 0:
        raise AssertionError(
            f"Negative effective K: "
            f"{result.k_effective}"
        )

    if result.k_effective > result.k_requested:
        raise AssertionError(
            "Effective K exceeds requested K: "
            f"{result.k_effective} > "
            f"{result.k_requested}"
        )

    if len(result.selected_candidates) != (
        result.k_effective
    ):
        raise AssertionError(
            "selected_candidates length does not "
            "match k_effective for query "
            f"{result.query_id}: "
            f"{len(result.selected_candidates)} != "
            f"{result.k_effective}"
        )

    if result.k_available < 0:
        raise AssertionError(
            f"Negative k_available: "
            f"{result.k_available}"
        )

    if result.k_effective > result.k_available:
        raise AssertionError(
            f"k_effective exceeds k_available for "
            f"{result.query_id}: "
            f"{result.k_effective} > "
            f"{result.k_available}"
        )

    if result.status not in {
        "ok",
        "ok_with_warning",
        "fallback",
        "error",
    }:
        raise AssertionError(
            f"Unexpected R4 status: "
            f"{result.status}"
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    s4_path: Path,
    r3_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    task: str = DEFAULT_TASK,
) -> Dict[str, Any]:
    """
    Execute one complete streaming R4 benchmark.

    Timing contract
    ---------------
    The benchmark timer starts immediately before R4 funnel
    execution and ends after the funnel has completely exhausted
    its generator.

    Therefore the measured runtime includes:

        adaptive K selection
        candidate/result serialization
        JSONL output writing
        atomic output commit

    Alignment is intentionally outside the R4 runtime.

    Result validation and benchmark statistics are performed as
    bookkeeping around the yielded results and do not define a
    separate selection timer.
    """
    ensure_parent(output_path)
    ensure_parent(report_path)

    filtered_s4_path: Optional[Path] = None
    filtered_r3_path: Optional[Path] = None

    print("=" * 72)
    print("QA-R4 Adaptive K Benchmark")
    print("=" * 72)

    print(
        f"S4 input : {s4_path}"
    )

    print(
        f"R3 input : {r3_path}"
    )

    print(
        f"Task     : {task}"
    )

    print(
        f"Output   : {output_path}"
    )

    print(
        f"Report   : {report_path}"
    )

    print()

    stats = R4BenchmarkStats()

    alignment_report: Dict[str, Any] = {}

    # --------------------------------------------------------------
    # Align inputs BEFORE starting the R4 selection timer.
    #
    # This keeps benchmark runtime focused on QA-R4 itself rather
    # than benchmark-only preprocessing.
    # --------------------------------------------------------------
    (
        filtered_s4_path,
        filtered_r3_path,
        alignment_report,
    ) = build_aligned_inputs(
        s4_path=s4_path,
        r3_path=r3_path,
        task=task,
    )

    print(
        "S4 task distribution : "
        f"{alignment_report['s4_total_records_by_task']}"
    )

    print(
        f"S4 selected ({task})  : "
        f"{alignment_report['s4_task_query_count']}"
    )

    print(
        f"R3 total query count  : "
        f"{alignment_report['r3_total_query_count']}"
    )

    print(
        f"R3 matching QA count  : "
        f"{alignment_report['r3_matching_task_query_count']}"
    )

    print(
        f"R3 excluded non-QA    : "
        f"{alignment_report['r3_excluded_non_task_query_count']}"
    )

    print(
        "Query order alignment : PASS "
        "(both S4 and R3 restricted+reordered)"
    )

    print()

    # --------------------------------------------------------------
    # R4 benchmark timer
    #
    # IMPORTANT:
    # Start BEFORE R4 funnel execution.
    #
    # The timer covers the complete streaming funnel execution,
    # including selection, serialization, output write, and atomic
    # output commit performed by run_jsonl_funnel().
    # --------------------------------------------------------------
    r4_started = time.perf_counter()

    try:
        iterator = run_jsonl_funnel(
            s4_path=filtered_s4_path,
            r3_path=filtered_r3_path,
            output_path=output_path,
        )

        for result in iterator:
            # Validation is bookkeeping/verification.
            #
            # No separate validate_result() timing is recorded.
            validate_result(result)

            stats.update(
                result=result,
            )

        r4_elapsed = (
            time.perf_counter()
            - r4_started
        )

    except InputDesyncError:
        print()
        print(
            "ERROR: S4/R3 input desynchronization "
            "detected."
        )
        print(
            "The alignment step was unable to produce "
            "matching query order."
        )
        raise

    finally:
        for temp_path in (
            filtered_s4_path,
            filtered_r3_path,
        ):
            if temp_path is not None:
                try:
                    temp_path.unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass

    # --------------------------------------------------------------
    # The generator has been fully exhausted here.
    #
    # Therefore the R4 funnel has completed its:
    #
    #   selection
    #   serialization
    #   output write
    #   atomic commit
    #
    # before the timer is stopped.
    # --------------------------------------------------------------
    output_committed = output_path.exists()

    if not output_committed:
        raise RuntimeError(
            "R4 funnel completed without producing "
            f"the expected output: {output_path}"
        )

    if not ATOMIC_OUTPUT_CONTRACT:
        raise RuntimeError(
            "R4 benchmark cannot report atomic_output=true "
            "because the atomic output contract is disabled."
        )

    # --------------------------------------------------------------
    # Runtime statistics
    #
    # There is intentionally one authoritative full-run measurement.
    # Per-query p50/p95 are not fabricated because the streaming
    # generator performs write/serialization across generator
    # advancement boundaries.
    # --------------------------------------------------------------
    result_stats = stats.to_dict()

    result_stats["runtime_seconds"] = {
        "total": r4_elapsed,

        "mean_per_query": (
            r4_elapsed / stats.query_count
            if stats.query_count
            else None
        ),

        "p50_per_query": None,

        "p95_per_query": None,

        "max_per_query": None,

        "scope": (
            "full_r4_funnel: "
            "adaptive K selection + serialization "
            "+ JSONL write + atomic commit"
        ),
    }

    if stats.query_count == 0:
        print(
            "WARNING: benchmark processed 0 queries."
        )

    report: Dict[str, Any] = {
        "schema_version": (
            "qa_r4_benchmark.v1"
        ),

        "task": "QA-R4",

        "benchmark": {
            "name": "adaptive_k_funnel",

            "target_task": task,

            "started_at": utc_now_iso(),

            "elapsed_seconds": r4_elapsed,

            "timer_scope": (
                "R4 adaptive K selection "
                "+ JSON serialization "
                "+ JSONL write "
                "+ atomic output commit"
            ),

            "timer_excludes_alignment": True,

            "output_committed": (
                output_committed
            ),
        },

        "contract": {
            "s4_field": "semantic_k_hint",

            "s4_task_filter": task,

            "s4_alignment": (
                "reordered_to_r3_query_order"
            ),

            "r3_alignment": (
                "restricted_and_reordered_to_task"
            ),

            "r3_field": "candidates",

            "selection": (
                "candidates[:semantic_k_hint]"
            ),

            "allowed_k": [
                100,
                300,
                500,
            ],

            "reranking": False,

            "uncertainty_recomputed": False,

            "entropy_recomputed": False,

            "streaming": True,

            "atomic_output": True,
        },

        "inputs": {
            "s4": str(s4_path),
            "r3": str(r3_path),
        },

        "outputs": {
            "candidate_jsonl": str(
                output_path
            ),

            "report_json": str(
                report_path
            ),
        },

        "alignment": alignment_report,

        "statistics": result_stats,
    }

    with report_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
            cls=R4JSONEncoder,
            allow_nan=False,
        )

        handle.write("\n")

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    print(
        f"Queries processed : "
        f"{stats.query_count:,}"
    )

    print(
        "Requested K       : "
        f"{dict(sorted(stats.requested_k.items()))}"
    )

    print(
        "Effective K       : "
        f"{dict(sorted(stats.effective_k.items()))}"
    )

    print(
        "Status             : "
        f"{dict(sorted(stats.status.items()))}"
    )

    if stats.fallback_reason:
        print(
            "Fallback reasons   : "
            f"{dict(sorted(stats.fallback_reason.items()))}"
        )

    print(
        "Selected candidates: "
        f"{stats.selected_candidate_total:,}"
    )

    print(
        "R4 full runtime    : "
        f"{r4_elapsed:.6f}s"
    )

    if stats.query_count:
        print(
            "Mean/query          : "
            f"{r4_elapsed / stats.query_count:.6f}s"
        )

    print(
        "Atomic output       : "
        f"{output_committed and ATOMIC_OUTPUT_CONTRACT}"
    )

    print()
    print(
        f"Output : {output_path}"
    )

    print(
        f"Report : {report_path}"
    )

    print("=" * 72)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark QA-R4 Adaptive K using frozen "
            "QA-S4 and QA-R3 outputs."
        )
    )

    parser.add_argument(
        "--s4",
        type=Path,
        default=DEFAULT_S4,
        help=(
            "QA-S4 semantic parsing JSONL "
            f"(default: {DEFAULT_S4})"
        ),
    )

    parser.add_argument(
        "--r3",
        type=Path,
        default=None,
        help=(
            "QA-R3 candidate file. "
            "If omitted, known default paths are searched."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "R4 candidate output JSONL "
            f"(default: {DEFAULT_OUTPUT})"
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=(
            "Benchmark report JSON "
            f"(default: {DEFAULT_REPORT})"
        ),
    )

    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        choices=("qa",),
        help=(
            "Task to benchmark. "
            "QA-R4 currently supports QA only "
            "(default: qa)."
        ),
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        r3_path = resolve_r3_input(
            args.r3
        )

        validate_input_paths(
            s4_path=args.s4,
            r3_path=r3_path,
            output_path=args.output,
            report_path=args.report,
        )

        run_benchmark(
            s4_path=args.s4,
            r3_path=r3_path,
            output_path=args.output,
            report_path=args.report,
            task=args.task,
        )

        return 0

    except Exception as exc:
        print()
        print("=" * 72)
        print("BENCHMARK FAILED")
        print("=" * 72)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("=" * 72)

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )