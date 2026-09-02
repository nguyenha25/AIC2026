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
NOT build an in-memory query_id -> record map (that's the whole point
of "streaming": bounded memory even for very large runs).

Therefore this benchmark must align BOTH sides before handing them to
the funnel:

    1. reads R3 query order (as-is, may contain QA + TRAKE)
    2. extracts QA records from S4
    3. intersects R3 order with the S4 QA query IDs, preserving R3
       order -> r3_order_task
    4. reorders S4 to exactly match r3_order_task
    5. reorders/filters R3 to exactly match r3_order_task too
       (this is the piece that was previously MISSING and caused
       InputDesyncError: R3 was passed to run_jsonl_funnel unfiltered,
       still containing TRAKE records interleaved with QA, so line i
       of the aligned S4 (QA-only) no longer matched line i of R3
       (QA+TRAKE mixed) as soon as a TRAKE record appeared before the
       i-th QA record.)
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


def percentile(values: List[float], p: float) -> Optional[float]:
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
        + fraction * (float(values[high]) - float(values[low]))
    )


def resolve_r3_input(explicit_path: Optional[Path]) -> Path:
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
            f"  - {path}" for path in DEFAULT_R3_CANDIDATE_FILES
        )

        raise FileNotFoundError(
            "Could not find an R3 candidate file automatically.\n"
            "Use --r3 explicitly.\n"
            f"Checked:\n{candidates}"
        )

    if len(existing) > 1:
        candidates = "\n".join(
            f"  - {path}" for path in existing
        )

        raise RuntimeError(
            "Multiple possible R3 candidate files were found.\n"
            "Use --r3 explicitly to avoid benchmarking the wrong file.\n"
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

    if output_resolved in {s4_resolved, r3_resolved}:
        raise ValueError(
            "Output path must not overwrite an input file."
        )

    if report_resolved in {s4_resolved, r3_resolved}:
        raise ValueError(
            "Report path must not overwrite an input file."
        )

    if output_resolved == report_resolved:
        raise ValueError(
            "Output JSONL and report JSON must use different paths."
        )


def ensure_parent(path: Path) -> None:
    """Create parent directory if necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _make_temp_jsonl(prefix: str) -> Path:
    """
    Create an empty temp file and return its Path with the OS-level
    file descriptor already closed.

    NOTE (bugfix):
        `tempfile.mkstemp()` returns an OS-level fd that must be
        closed explicitly if the caller is going to reopen the path
        with a normal `open()` call (as this module does, to get
        text-mode + newline control). The previous version never
        closed this fd, leaking one file descriptor per benchmark run
        until process exit. Harmless for a handful of files, but easy
        to fix and worth fixing.
    """
    fd, temp_name = tempfile.mkstemp(
        prefix=prefix,
        suffix=".jsonl",
    )

    os.close(fd)

    return Path(temp_name)


def _read_r3_records(r3_path: Path) -> List[Dict[str, Any]]:
    """
    Đọc R3 input, tự động nhận diện 2 format có thể gặp trong thực tế:

    1. JSON ARRAY — format THẬT của export_qa_r3.py (script sản xuất
       ra qa_r3_candidates.json) khi chạy không kèm --query-id:

           [
             {"query_id": "04", "candidates": [...]},
             {"query_id": "05", "candidates": [...]},
             ...
           ]

       Toàn bộ file là MỘT JSON document duy nhất (list ở top-level).

    2. JSON OBJECT ĐƠN — khi export_qa_r3.py chạy với --query-id,
       output là đúng MỘT object (không phải list):

           {"query_id": "04", "candidates": [...]}

    3. JSONL — một JSON object mỗi dòng (định dạng cũ mà bản trước
       của benchmark_qa_r4.py giả định LÀ DUY NHẤT — giả định này sai
       với script export thật, gây ra lỗi
       "Expecting value: line 1 column 2" khi trỏ vào file .json thật
       vì dòng đầu tiên của một JSON array pretty-print chỉ có ký tự
       '[').

    Không đoán theo phần đuôi file (.json/.jsonl) — nhiều dữ liệu thực
    tế đặt sai đuôi. Thay vào đó xét nội dung: thử parse cả file như
    một JSON document trước; nếu là list -> format (1); nếu là dict
    -> format (2); nếu parse cả file thất bại -> fallback đọc JSONL
    theo từng dòng (format 3), giữ nguyên báo lỗi rõ theo số dòng như
    trước.
    """

    text = r3_path.read_text(encoding="utf-8-sig")

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

            for index, record in enumerate(document):
                if not isinstance(record, dict):
                    raise ValueError(
                        f"R3 array item [{index}] phải là JSON "
                        f"object; got {type(record).__name__}."
                    )
                records.append(record)

            return records

        if isinstance(document, dict):
            # Output của export_qa_r3.py khi chạy --query-id: đúng
            # một object bao trọn file.
            return [document]

        # document là None (parse cả file thất bại) HOẶC không phải
        # list/dict -> coi là JSONL, đọc từng dòng như cũ.
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
                    f"object; got {type(record).__name__}"
                )

            records.append(record)

        if not records:
            raise ValueError(
                f"Không đọc được record nào từ R3 input: {r3_path} "
                "(không phải JSON array/object hợp lệ, cũng không "
                "phải JSONL hợp lệ)."
            )

        return records

    raise ValueError(
        f"Không nhận diện được format R3 input: {r3_path} "
        "(nội dung không bắt đầu bằng '[' hoặc '{')."
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

    Hỗ trợ cả JSON array (format thật của export_qa_r3.py) lẫn JSONL
    thông qua `_read_r3_records()`.

    Returns:
        query_order
        candidate_count_by_query
    """
    records = _read_r3_records(r3_path)

    query_order: List[str] = []
    candidate_count_by_query: Counter = Counter()

    for index, record in enumerate(records, start=1):

        if "query_id" not in record:
            raise KeyError(
                f"R3 record #{index} is missing 'query_id'."
            )

        query_id = str(record["query_id"])

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
        candidate_count_by_query[query_id] = len(candidates)

    if not query_order:
        raise ValueError(
            f"R3 input contains no query records: {r3_path}"
        )

    return query_order, candidate_count_by_query


def _load_r3_raw_lines_by_query_id(
    r3_path: Path,
) -> Dict[str, str]:
    """
    Trả về query_id -> JSON text một dòng (compact, re-serialized) cho
    từng R3 record, bất kể R3 input gốc là JSON array hay JSONL.

    Lưu ý:
        Với JSONL, bản trước tái sử dụng nguyên văn dòng gốc để tránh
        rủi ro đổi định dạng. Với JSON array (format thật hiện tại)
        thì không có "dòng gốc" riêng cho từng record — chúng nằm
        chung trong một document lớn — nên bắt buộc phải
        re-serialize bằng `json.dumps()`. Áp dụng thống nhất cho cả
        hai trường hợp: nội dung record không đổi, chỉ khác khoảng
        trắng/định dạng hiển thị.
    """

    records = _read_r3_records(r3_path)

    result: Dict[str, str] = {}

    for index, record in enumerate(records, start=1):

        if "query_id" not in record:
            raise KeyError(
                f"R3 record #{index} is missing 'query_id'."
            )

        query_id = str(record["query_id"])

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
    Build temporary S4 AND R3 JSONL files that are BOTH restricted and
    reordered to the exact same query_id sequence, so that
    `run_jsonl_funnel()` (a strict positional streaming zip) sees line
    i of S4 and line i of R3 referring to the same query.

    IMPORTANT:
        R3 may contain multiple task types (QA + TRAKE), while this
        benchmark is QA-R4 only. If only S4 is filtered/reordered and
        R3 is passed through as-is, R3's TRAKE records interleaved
        among the QA records will desync the two streams as soon as a
        TRAKE record appears before the corresponding QA position —
        this was the previous bug (InputDesyncError).

    Therefore:
        1. Read all S4 records for the requested task.
        2. Read all R3 query IDs (task-agnostic).
        3. Keep only R3 query IDs that belong to the requested S4
           task, preserving R3's original relative order
           -> r3_order_task.
        4. Require every requested-task S4 query to exist in R3.
        5. Write a temp S4 JSONL reordered to r3_order_task.
        6. Write a temp R3 JSONL RESTRICTED AND reordered to
           r3_order_task (this is the piece that was missing before).
        7. Return both temp paths for run_jsonl_funnel().

    Original S4/R3 files are never modified.
    """

    # --------------------------------------------------------------
    # Read R3 in its original order.
    #
    # R3 may contain both QA and TRAKE candidates.
    # --------------------------------------------------------------
    r3_order_all, r3_candidate_counts_all = load_r3_query_order(
        r3_path
    )

    # --------------------------------------------------------------
    # Read S4 records for the requested task only.
    # --------------------------------------------------------------
    s4_records: Dict[str, Dict[str, Any]] = {}
    task_counts: Counter = Counter()

    with s4_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
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
                    f"S4 line {line_number} must be a JSON object; "
                    f"got {type(record).__name__}"
                )

            record_task = str(
                record.get("task", "")
            ).lower()

            task_counts[record_task] += 1

            if record_task != task.lower():
                continue

            if "query_id" not in record and "id" not in record:
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
                    f"Duplicate S4 {task} query_id={query_id!r} "
                    f"at line {line_number}."
                )

            if "semantic_k_hint" not in record:
                raise KeyError(
                    f"S4 {task!r} query {query_id!r} is missing "
                    "'semantic_k_hint'."
                )

            s4_records[query_id] = record

    if not s4_records:
        raise ValueError(
            f"No S4 records found for task={task!r}."
        )

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # R3 contains QA + TRAKE.
    # QA-R4 only consumes QA.
    #
    # Therefore intersect R3 with the requested S4 task IDs.
    # Preserve R3 ordering.
    # --------------------------------------------------------------
    s4_task_ids = set(s4_records)

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
    # Every requested-task S4 query must have an R3 candidate record.
    # --------------------------------------------------------------
    missing_in_r3 = [
        query_id
        for query_id in s4_records
        if query_id not in set(r3_order_task)
    ]

    if missing_in_r3:
        raise ValueError(
            f"S4 {task} contains query IDs missing from R3: "
            f"{sorted(missing_in_r3, key=str)}"
        )

    # --------------------------------------------------------------
    # Sanity check after task intersection.
    # --------------------------------------------------------------
    if len(s4_records) != len(r3_order_task):
        raise ValueError(
            "S4/R3 query count mismatch after task filtering: "
            f"S4 {task}={len(s4_records)}, "
            f"R3 matching {task}={len(r3_order_task)}"
        )

    # --------------------------------------------------------------
    # IDs excluded from R3 because they are not part of this task.
    # These are expected for mixed QA/TRAKE R3 output.
    # --------------------------------------------------------------
    excluded_r3_ids = [
        query_id
        for query_id in r3_order_all
        if query_id not in s4_task_ids
    ]

    # --------------------------------------------------------------
    # Write temporary S4 in EXACT filtered-R3 order.
    # --------------------------------------------------------------
    temp_s4_path = _make_temp_jsonl(
        prefix="qa_r4_s4_aligned_"
    )

    try:
        with open(
            temp_s4_path,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output:
            for query_id in r3_order_task:
                record = s4_records[query_id]

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
            temp_s4_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    # --------------------------------------------------------------
    # Write temporary R3, RESTRICTED to r3_order_task and in EXACTLY
    # that order.
    #
    # This is the fix: previously the ORIGINAL (unfiltered) r3_path
    # was handed to run_jsonl_funnel, which still contained TRAKE
    # records interleaved with QA records. Since run_jsonl_funnel does
    # a strict line-by-line positional zip against the aligned
    # QA-only S4, any TRAKE record appearing before the corresponding
    # QA position caused an immediate InputDesyncError.
    #
    # We reuse the raw line text (no re-serialization) so we do not
    # risk altering field order/formatting of records that must stay
    # untouched.
    # --------------------------------------------------------------
    r3_raw_lines = _load_r3_raw_lines_by_query_id(
        r3_path
    )

    temp_r3_path = _make_temp_jsonl(
        prefix="qa_r4_r3_aligned_"
    )

    try:
        with open(
            temp_r3_path,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output:
            for query_id in r3_order_task:
                output.write(
                    r3_raw_lines[query_id] + "\n"
                )

    except Exception:
        try:
            temp_r3_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            temp_s4_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    alignment_report = {
        "task": task,

        "s4_total_records_by_task": dict(
            sorted(task_counts.items())
        ),

        "s4_task_query_count": len(s4_records),

        "r3_total_query_count": len(r3_order_all),

        "r3_matching_task_query_count": len(r3_order_task),

        "r3_excluded_non_task_query_count": len(
            excluded_r3_ids
        ),

        "r3_excluded_non_task_query_ids": excluded_r3_ids,

        "query_order_source": "R3",

        "query_order_aligned": True,

        "r3_restricted_to_task": True,

        "query_order": r3_order_task,

        "candidate_count_by_query": {
            query_id: r3_candidate_counts[query_id]
            for query_id in r3_order_task
        },
    }

    return temp_s4_path, temp_r3_path, alignment_report


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

        self.elapsed_seconds: List[float] = []

        self.first_query_id: Optional[str] = None
        self.last_query_id: Optional[str] = None

    def update(
        self,
        result: AdaptiveKResult,
        elapsed_seconds: float,
    ) -> None:
        """Update aggregate statistics with one R4 result."""
        self.query_count += 1

        query_id = str(result.query_id)

        if self.first_query_id is None:
            self.first_query_id = query_id

        self.last_query_id = query_id

        self.requested_k[result.k_requested] += 1
        self.effective_k[result.k_effective] += 1

        self.status[result.status] += 1

        if result.fallback_reason:
            self.fallback_reason[result.fallback_reason] += 1

        if result.fallback_reason == "candidate_pool_below_requested_k":
            self.candidate_pool_below_requested += 1

        if result.fallback_reason == "empty_candidate_pool":
            self.empty_candidate_pool += 1

        selected_count = len(result.selected_candidates)

        self.selected_candidate_total += selected_count

        self.available_candidates.append(result.k_available)

        if result.k_requested > 0:
            self.selection_ratios.append(
                selected_count / result.k_requested
            )

        self.elapsed_seconds.append(elapsed_seconds)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize benchmark statistics."""
        elapsed = self.elapsed_seconds
        total_selected = self.selected_candidate_total

        return {
            "query_count": self.query_count,
            "first_query_id": self.first_query_id,
            "last_query_id": self.last_query_id,

            "requested_k_distribution": {
                str(k): count
                for k, count in sorted(self.requested_k.items())
            },

            "effective_k_distribution": {
                str(k): count
                for k, count in sorted(self.effective_k.items())
            },

            "status_distribution": dict(
                sorted(self.status.items())
            ),

            "fallback_reason_distribution": dict(
                sorted(self.fallback_reason.items())
            ),

            "candidate_pool_below_requested_k": (
                self.candidate_pool_below_requested
            ),

            "empty_candidate_pool": self.empty_candidate_pool,

            "selected_candidate_total": total_selected,

            "mean_selected_candidates": (
                total_selected / self.query_count
                if self.query_count
                else 0.0
            ),

            "candidate_availability": {
                "min": (
                    min(self.available_candidates)
                    if self.available_candidates
                    else None
                ),
                "p50": percentile(
                    [float(x) for x in self.available_candidates],
                    0.50,
                ),
                "p95": percentile(
                    [float(x) for x in self.available_candidates],
                    0.95,
                ),
                "max": (
                    max(self.available_candidates)
                    if self.available_candidates
                    else None
                ),
            },

            "selection_ratio": {
                "mean": (
                    statistics.fmean(self.selection_ratios)
                    if self.selection_ratios
                    else None
                ),
                "min": (
                    min(self.selection_ratios)
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
                "total": (
                    sum(elapsed)
                    if elapsed
                    else 0.0
                ),
                "mean_per_query": (
                    statistics.fmean(elapsed)
                    if elapsed
                    else None
                ),
                "p50_per_query": percentile(
                    elapsed,
                    0.50,
                ),
                "p95_per_query": percentile(
                    elapsed,
                    0.95,
                ),
                "max_per_query": (
                    max(elapsed)
                    if elapsed
                    else None
                ),
            },
        }


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_result(result: AdaptiveKResult) -> None:
    """
    Validate one R4 result.

    Frozen contract:
        semantic_k_hint ∈ {100, 300, 500}
    """
    if not result.query_id:
        raise AssertionError(
            "R4 result has empty query_id."
        )

    if result.k_requested not in {100, 300, 500}:
        raise AssertionError(
            f"Unexpected requested K: {result.k_requested}"
        )

    if result.k_effective < 0:
        raise AssertionError(
            f"Negative effective K: {result.k_effective}"
        )

    if result.k_effective > result.k_requested:
        raise AssertionError(
            "Effective K exceeds requested K: "
            f"{result.k_effective} > {result.k_requested}"
        )

    if len(result.selected_candidates) != result.k_effective:
        raise AssertionError(
            "selected_candidates length does not match "
            f"k_effective for query {result.query_id}: "
            f"{len(result.selected_candidates)} != "
            f"{result.k_effective}"
        )

    if result.k_available < 0:
        raise AssertionError(
            f"Negative k_available: {result.k_available}"
        )

    if result.k_effective > result.k_available:
        raise AssertionError(
            f"k_effective exceeds k_available for "
            f"{result.query_id}: "
            f"{result.k_effective} > {result.k_available}"
        )

    if result.status not in {
        "ok",
        "ok_with_warning",
        "fallback",
        "error",
    }:
        raise AssertionError(
            f"Unexpected R4 status: {result.status}"
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

    Both S4 and R3 are first aligned (filtered + reordered) to the
    exact same query sequence before being handed to the strict
    positional streaming funnel.
    """
    ensure_parent(output_path)
    ensure_parent(report_path)

    benchmark_started = time.perf_counter()
    first_result_time: Optional[float] = None

    filtered_s4_path: Optional[Path] = None
    filtered_r3_path: Optional[Path] = None

    print("=" * 72)
    print("QA-R4 Adaptive K Benchmark")
    print("=" * 72)
    print(f"S4 input : {s4_path}")
    print(f"R3 input : {r3_path}")
    print(f"Task     : {task}")
    print(f"Output   : {output_path}")
    print(f"Report   : {report_path}")
    print()

    try:
        # --------------------------------------------------------------
        # Align S4 AND R3 to the exact same restricted query order.
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

        stats = R4BenchmarkStats()

        iterator = run_jsonl_funnel(
            s4_path=filtered_s4_path,
            r3_path=filtered_r3_path,
            output_path=output_path,
        )

        for result in iterator:
            if first_result_time is None:
                first_result_time = time.perf_counter()

            query_started = time.perf_counter()

            validate_result(result)

            query_elapsed = (
                time.perf_counter() - query_started
            )

            stats.update(
                result=result,
                elapsed_seconds=query_elapsed,
            )

            if stats.query_count % 1000 == 0:
                print(
                    f"  processed {stats.query_count:,} queries..."
                )

    except InputDesyncError:
        print()
        print("ERROR: S4/R3 input desynchronization detected.")
        print(
            "The alignment step was unable to produce matching "
            "query order."
        )
        raise

    finally:
        for temp_path in (filtered_s4_path, filtered_r3_path):
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    total_elapsed = (
        time.perf_counter() - benchmark_started
    )

    result_stats = stats.to_dict()

    report: Dict[str, Any] = {
        "schema_version": "qa_r4_benchmark.v1",

        "task": "QA-R4",

        "benchmark": {
            "name": "adaptive_k_funnel",
            "target_task": task,
            "started_at": utc_now_iso(),
            "elapsed_seconds": total_elapsed,
            "first_result_latency_seconds": (
                (
                    first_result_time - benchmark_started
                )
                if first_result_time is not None
                else None
            ),
        },

        "contract": {
            "s4_field": "semantic_k_hint",
            "s4_task_filter": task,
            "s4_alignment": "reordered_to_r3_query_order",
            "r3_alignment": "restricted_and_reordered_to_task",
            "r3_field": "candidates",
            "selection": "candidates[:semantic_k_hint]",
            "allowed_k": [100, 300, 500],
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
            "candidate_jsonl": str(output_path),
            "report_json": str(report_path),
        },

        "alignment": alignment_report,

        "statistics": result_stats,
    }

    if stats.query_count == 0:
        print(
            "WARNING: benchmark processed 0 queries."
        )

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
        )
        handle.write("\n")

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    print(
        f"Queries processed : {stats.query_count:,}"
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
        f"Total runtime      : {total_elapsed:.4f}s"
    )

    if stats.query_count:
        print(
            "Mean/query          : "
            f"{total_elapsed / stats.query_count:.6f}s"
        )

    print()
    print(f"Output : {output_path}")
    print(f"Report : {report_path}")
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
            "QA-R3 fused candidates JSONL. "
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
        r3_path = resolve_r3_input(args.r3)

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
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 72)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())