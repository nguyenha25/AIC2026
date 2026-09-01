"""
QA-R2 — Multi-Source Score Normalization Benchmark
===================================================

Mục đích:
---------
Đánh giá hiệu năng và độ ổn định số học của các phương pháp
score normalization:

    - Min-Max
    - Robust
    - Rank

trên pipeline đa nguồn:

    CLIP-B / OCR / ASR
        -> normalize từng source
        -> weighted fusion

Benchmark gồm:

1. Candidate Pool Scaling
   - Small  : 100 candidates / query
   - Medium : 1,000 candidates / query
   - Large  : 10,000 candidates / query

2. Performance
   - Single-query latency
   - P95 latency
   - Batch throughput (queries/sec)

3. Numerical Stability
   - Dense floating-point scores (CLIP)
   - Heavy ties (OCR / keyword-like scores)
   - Extreme outliers
   - Negative log-likelihood scale (ASR)
   - Constant-score edge case
   - NaN / Inf contamination

Acceptance Criteria:
--------------------
Tại candidate pool N = 1,000:

    - Average latency <= 2.0 ms / query
    - Throughput >= 500 queries / sec

Numerical stability:
--------------------
Normalization output phải:

    - hữu hạn (finite)
    - nằm trong [0, 1]
    - không sinh NaN / Inf
    - xử lý được constant-score input

Lưu ý:
------
Benchmark này đánh giá implementation hiện tại của
scripts.score_normalization.

Chạy từ root repository:

    python -u -m scripts.benchmark_qa_r2

Report:

    artifacts/qa_r2/benchmark_qa_r2.json
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from scripts.score_normalization import (
    MultiSourceScoreNormalizer,
    NormMethod,
    normalize_scores,
)


# ============================================================
# CONFIG — QA-R2 CONTRACT
# ============================================================

CANDIDATE_SCALES = [100, 1_000, 10_000]

NUM_QUERIES = 100

PRIMITIVE_CANDIDATES = 1_000
PRIMITIVE_ITERATIONS = 500

WARMUP_QUERIES = 5

ACCEPTANCE_CANDIDATES = 1_000

LATENCY_TARGET_MS = 2.0
THROUGHPUT_TARGET_QPS = 500.0

REPORT_PATH = Path("artifacts/qa_r2/benchmark_qa_r2.json")

WEIGHTS = {
    "clip_b": 1.0,
    "ocr": 0.5,
    "asr": 0.3,
}


# ============================================================
# RESULT CONTRACTS
# ============================================================

@dataclass(slots=True, frozen=True)
class BenchmarkResult:
    method: str
    num_candidates: int
    num_queries: int
    total_time_sec: float
    avg_latency_ms: float
    p95_latency_ms: float
    throughput_qps: float


@dataclass(slots=True, frozen=True)
class StabilityResult:
    method: str
    scenario: str
    passed: bool
    finite: bool
    in_unit_range: bool
    constant_input_safe: bool
    details: str


@dataclass(slots=True, frozen=True)
class AcceptanceResult:
    candidate_pool: int
    latency_target_ms: float
    latency_actual_ms: float
    latency_pass: bool
    throughput_target_qps: float
    throughput_actual_qps: float
    throughput_pass: bool
    numerical_stability_pass: bool
    overall_pass: bool


@dataclass(slots=True)
class BenchmarkReport:
    contract: Dict[str, float | int | List[int]]
    performance: List[BenchmarkResult] = field(default_factory=list)
    stability: List[StabilityResult] = field(default_factory=list)
    acceptance: AcceptanceResult | None = None


# ============================================================
# MOCK DATA GENERATOR
# ============================================================

def generate_mock_query_data(
    num_queries: int,
    num_candidates: int,
    seed: int = 42,
) -> List[Dict[str, Dict[str, float]]]:
    """
    Sinh dữ liệu mô phỏng score từ nhiều nguồn.

    CLIP:
        Dense floating-point cosine-like scores.

    OCR:
        Discrete / tied scores + extreme outliers.

    ASR:
        Negative log-likelihood-like scale.
    """

    rng = np.random.default_rng(seed)

    dataset: List[Dict[str, Dict[str, float]]] = []

    items = [f"item_{i}" for i in range(num_candidates)]

    for _ in range(num_queries):
        # ----------------------------------------------------
        # 1. CLIP — dense floating-point distribution
        # ----------------------------------------------------
        clip_scores = rng.uniform(
            0.20,
            0.35,
            size=num_candidates,
        )

        # ----------------------------------------------------
        # 2. OCR — heavy ties / discrete scores
        # ----------------------------------------------------
        ocr_scores = rng.choice(
            [
                0.0,
                0.1,
                0.2,
                0.5,
                1.0,
                2.0,
                5.0,
            ],
            size=num_candidates,
            p=[
                0.30,
                0.20,
                0.15,
                0.15,
                0.10,
                0.07,
                0.03,
            ],
        ).astype(np.float64)

        # Extreme outliers.
        if num_candidates >= 10:
            outlier_indices = rng.choice(
                num_candidates,
                size=min(3, num_candidates),
                replace=False,
            )
            ocr_scores[outlier_indices] *= 10.0

        # ----------------------------------------------------
        # 3. ASR — negative log-likelihood-like distribution
        # ----------------------------------------------------
        asr_scores = rng.uniform(
            -50.0,
            -5.0,
            size=num_candidates,
        )

        query_data = {
            "clip_b": dict(
                zip(items, clip_scores.tolist())
            ),
            "ocr": dict(
                zip(items, ocr_scores.tolist())
            ),
            "asr": dict(
                zip(items, asr_scores.tolist())
            ),
        }

        dataset.append(query_data)

    return dataset


# ============================================================
# NUMERICAL STABILITY TEST DATA
# ============================================================

def generate_stability_cases(
    num_candidates: int = 1_000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Tạo các input đặc biệt để kiểm tra numerical stability.
    """

    rng = np.random.default_rng(seed)

    return {
        "dense_float": rng.uniform(
            0.20,
            0.35,
            size=num_candidates,
        ),

        "heavy_ties": rng.choice(
            [0.0, 0.1, 0.2, 0.5, 1.0],
            size=num_candidates,
        ).astype(np.float64),

        "extreme_outlier": np.concatenate(
            [
                rng.uniform(
                    -10.0,
                    10.0,
                    size=num_candidates - 1,
                ),
                np.array([1e12], dtype=np.float64),
            ]
        ),

        "negative_scores": rng.uniform(
            -50.0,
            -5.0,
            size=num_candidates,
        ),

        "constant": np.full(
            num_candidates,
            7.0,
            dtype=np.float64,
        ),

        "tiny_range": (
            1.0
            + rng.uniform(
                -1e-12,
                1e-12,
                size=num_candidates,
            )
        ),

        "large_magnitude": rng.uniform(
            -1e15,
            1e15,
            size=num_candidates,
        ),
    }


# ============================================================
# HELPER — VALIDATE NORMALIZATION OUTPUT
# ============================================================

def validate_normalized_output(
    output: np.ndarray,
) -> tuple[bool, bool]:
    """
    Kiểm:

    1. finite:
       Không có NaN / +Inf / -Inf.

    2. in_unit_range:
       Tất cả output nằm trong [0, 1],
       cho phép sai số floating-point rất nhỏ.
    """

    values = np.asarray(output, dtype=np.float64)

    finite = bool(np.all(np.isfinite(values)))

    if not finite:
        return False, False

    eps = 1e-12

    in_unit_range = bool(
        np.all(values >= -eps)
        and np.all(values <= 1.0 + eps)
    )

    return finite, in_unit_range


# ============================================================
# NUMERICAL STABILITY BENCHMARK
# ============================================================

def benchmark_numerical_stability(
    num_candidates: int = 1_000,
) -> List[StabilityResult]:
    """
    Kiểm tra numerical stability cho từng normalization method.
    """

    print("\n" + "=" * 76)
    print("NUMERICAL STABILITY TEST")
    print("=" * 76)

    cases = generate_stability_cases(
        num_candidates=num_candidates,
    )

    results: List[StabilityResult] = []

    for method in NormMethod:
        print(f"\n[{method.value.upper()}]")

        for scenario, raw_scores in cases.items():

            # ------------------------------------------------
            # Normal input
            # ------------------------------------------------
            try:
                output = normalize_scores(
                    raw_scores,
                    method,
                )

                finite, in_unit_range = validate_normalized_output(
                    output
                )

            except Exception as exc:
                results.append(
                    StabilityResult(
                        method=method.value,
                        scenario=scenario,
                        passed=False,
                        finite=False,
                        in_unit_range=False,
                        constant_input_safe=False,
                        details=f"exception: {type(exc).__name__}: {exc}",
                    )
                )

                print(
                    f"  {scenario:18s} -> FAIL "
                    f"(exception: {type(exc).__name__})"
                )

                continue

            # ------------------------------------------------
            # Constant input safety
            # ------------------------------------------------
            constant_input_safe = True

            if scenario == "constant":
                constant_input_safe = finite and in_unit_range

            passed = (
                finite
                and in_unit_range
                and constant_input_safe
            )

            results.append(
                StabilityResult(
                    method=method.value,
                    scenario=scenario,
                    passed=passed,
                    finite=finite,
                    in_unit_range=in_unit_range,
                    constant_input_safe=constant_input_safe,
                    details="ok" if passed else "invalid normalized output",
                )
            )

            print(
                f"  {scenario:18s} -> "
                f"[{'PASS' if passed else 'FAIL'}] "
                f"finite={finite} "
                f"range01={in_unit_range}"
            )

    return results


# ============================================================
# PRIMITIVE PERFORMANCE BENCHMARK
# ============================================================

def benchmark_single_primitive(
    num_candidates: int = PRIMITIVE_CANDIDATES,
    iterations: int = PRIMITIVE_ITERATIONS,
) -> Dict[str, Dict[str, float]]:
    """
    Micro-benchmark normalize_scores() trên một mảng 1D.
    """

    print("\n" + "=" * 76)
    print("PRIMITIVE NORMALIZATION BENCHMARK")
    print("=" * 76)

    print(
        f"N = {num_candidates:,} candidates | "
        f"Iterations = {iterations:,}"
    )

    rng = np.random.default_rng(42)

    raw_scores = rng.uniform(
        -10.0,
        100.0,
        size=num_candidates,
    )

    results: Dict[str, Dict[str, float]] = {}

    for method in NormMethod:

        # ----------------------------------------------------
        # Warmup
        # ----------------------------------------------------
        for _ in range(10):
            normalize_scores(
                raw_scores,
                method,
            )

        latencies_ms: List[float] = []

        # ----------------------------------------------------
        # Timed iterations
        # ----------------------------------------------------
        for _ in range(iterations):
            t0 = time.perf_counter()

            normalize_scores(
                raw_scores,
                method,
            )

            t1 = time.perf_counter()

            latencies_ms.append(
                (t1 - t0) * 1000.0
            )

        avg_lat = float(
            np.mean(latencies_ms)
        )

        p95_lat = float(
            np.percentile(
                latencies_ms,
                95,
            )
        )

        qps = (
            1000.0 / avg_lat
            if avg_lat > 0.0
            else float("inf")
        )

        results[method.value] = {
            "avg_latency_ms": avg_lat,
            "p95_latency_ms": p95_lat,
            "throughput_ops": qps,
        }

        print(
            f"  {method.value:10s} | "
            f"Avg: {avg_lat:7.3f} ms | "
            f"P95: {p95_lat:7.3f} ms | "
            f"Throughput: {qps:9.1f} ops/sec"
        )

    return results


# ============================================================
# MULTI-SOURCE PIPELINE BENCHMARK
# ============================================================

def benchmark_multi_source_pipeline(
    candidate_scales: List[int] | None = None,
    num_queries: int = NUM_QUERIES,
) -> List[BenchmarkResult]:
    """
    Benchmark:

        MultiSourceScoreNormalizer
            -> normalize_source_dict()

    trên nhiều candidate-pool sizes.
    """

    if candidate_scales is None:
        candidate_scales = list(CANDIDATE_SCALES)

    print("\n" + "=" * 76)
    print("MULTI-SOURCE PIPELINE BENCHMARK")
    print("=" * 76)

    print(
        f"Queries = {num_queries:,} | "
        f"Candidate scales = {candidate_scales}"
    )

    benchmark_records: List[BenchmarkResult] = []

    for n_cand in candidate_scales:

        print(
            f"\n--- Candidate Pool: "
            f"{n_cand:,} items/query ---"
        )

        dataset = generate_mock_query_data(
            num_queries=num_queries,
            num_candidates=n_cand,
            seed=42,
        )

        for method in NormMethod:

            normalizer = MultiSourceScoreNormalizer(
                method=method,
                weights=WEIGHTS,
            )

            # ------------------------------------------------
            # Warmup
            # ------------------------------------------------
            warmup_count = min(
                WARMUP_QUERIES,
                len(dataset),
            )

            for query_raw in dataset[:warmup_count]:
                normalizer.normalize_source_dict(
                    query_raw
                )

            # ------------------------------------------------
            # Timed queries
            # ------------------------------------------------
            latencies_ms: List[float] = []

            t0_total = time.perf_counter()

            for query_raw in dataset:

                t0_q = time.perf_counter()

                normalizer.normalize_source_dict(
                    query_raw
                )

                t1_q = time.perf_counter()

                latencies_ms.append(
                    (t1_q - t0_q) * 1000.0
                )

            t1_total = time.perf_counter()

            total_time = t1_total - t0_total

            avg_lat = float(
                np.mean(latencies_ms)
            )

            p95_lat = float(
                np.percentile(
                    latencies_ms,
                    95,
                )
            )

            qps = (
                num_queries / total_time
                if total_time > 0.0
                else float("inf")
            )

            rec = BenchmarkResult(
                method=method.value,
                num_candidates=n_cand,
                num_queries=num_queries,
                total_time_sec=total_time,
                avg_latency_ms=avg_lat,
                p95_latency_ms=p95_lat,
                throughput_qps=qps,
            )

            benchmark_records.append(rec)

            print(
                f"  [{method.value.upper():8s}] "
                f"Avg: {avg_lat:7.3f} ms/q | "
                f"P95: {p95_lat:7.3f} ms/q | "
                f"Throughput: {qps:8.1f} QPS"
            )

    return benchmark_records


# ============================================================
# ACCEPTANCE
# ============================================================

def verify_acceptance(
    performance_results: List[BenchmarkResult],
    stability_results: List[StabilityResult],
) -> AcceptanceResult:
    """
    QA-R2 acceptance tại N=1,000.

    Contract:

        Average latency <= 2.0 ms
        Throughput >= 500 QPS

    Numerical stability:
        Tất cả stability cases của tất cả methods PASS.
    """

    candidates = [
        r
        for r in performance_results
        if r.num_candidates == ACCEPTANCE_CANDIDATES
    ]

    if not candidates:
        raise RuntimeError(
            "Missing performance benchmark result "
            f"for N={ACCEPTANCE_CANDIDATES}"
        )

    # --------------------------------------------------------
    # QA-R2 performance acceptance
    #
    # Contract applies to the normalization implementation
    # as a whole, therefore use the worst method at N=1000.
    # --------------------------------------------------------

    worst_latency = max(
        r.avg_latency_ms
        for r in candidates
    )

    worst_throughput = min(
        r.throughput_qps
        for r in candidates
    )

    latency_pass = (
        worst_latency <= LATENCY_TARGET_MS
    )

    throughput_pass = (
        worst_throughput >= THROUGHPUT_TARGET_QPS
    )

    # --------------------------------------------------------
    # Numerical stability
    # --------------------------------------------------------

    numerical_stability_pass = all(
        r.passed
        for r in stability_results
    )

    overall_pass = (
        latency_pass
        and throughput_pass
        and numerical_stability_pass
    )

    acceptance = AcceptanceResult(
        candidate_pool=ACCEPTANCE_CANDIDATES,
        latency_target_ms=LATENCY_TARGET_MS,
        latency_actual_ms=worst_latency,
        latency_pass=latency_pass,
        throughput_target_qps=THROUGHPUT_TARGET_QPS,
        throughput_actual_qps=worst_throughput,
        throughput_pass=throughput_pass,
        numerical_stability_pass=numerical_stability_pass,
        overall_pass=overall_pass,
    )

    return acceptance


# ============================================================
# REPORT
# ============================================================

def save_report(
    primitive_results: Dict[str, Dict[str, float]],
    pipeline_results: List[BenchmarkResult],
    stability_results: List[StabilityResult],
    acceptance: AcceptanceResult,
) -> None:
    """
    Lưu report JSON cho QA artifact.
    """

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "qa": "QA-R2",
        "benchmark": "Multi-Source Score Normalization",
        "contract": {
            "candidate_scales": CANDIDATE_SCALES,
            "num_queries": NUM_QUERIES,
            "acceptance_candidates": ACCEPTANCE_CANDIDATES,
            "latency_target_ms": LATENCY_TARGET_MS,
            "throughput_target_qps": THROUGHPUT_TARGET_QPS,
            "weights": WEIGHTS,
        },
        "primitive": primitive_results,
        "performance": [
            asdict(r)
            for r in pipeline_results
        ],
        "stability": [
            asdict(r)
            for r in stability_results
        ],
        "acceptance": asdict(acceptance),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[REPORT] Saved to: {REPORT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 76)
    print("QA-R2 — SCORE NORMALIZATION PERFORMANCE BENCHMARK")
    print("=" * 76)

    print("\n[CONTRACT]")
    print(
        f"  Candidate scales       : "
        f"{CANDIDATE_SCALES}"
    )
    print(
        f"  Queries per scale      : "
        f"{NUM_QUERIES}"
    )
    print(
        f"  Acceptance pool        : "
        f"N={ACCEPTANCE_CANDIDATES:,}"
    )
    print(
        f"  Average latency target : "
        f"<= {LATENCY_TARGET_MS:.3f} ms"
    )
    print(
        f"  Throughput target      : "
        f">= {THROUGHPUT_TARGET_QPS:.1f} QPS"
    )

    # ========================================================
    # 1. Primitive benchmark
    # ========================================================

    primitive_results = benchmark_single_primitive(
        num_candidates=PRIMITIVE_CANDIDATES,
        iterations=PRIMITIVE_ITERATIONS,
    )

    # ========================================================
    # 2. Multi-source pipeline benchmark
    # ========================================================

    pipeline_results = benchmark_multi_source_pipeline(
        candidate_scales=CANDIDATE_SCALES,
        num_queries=NUM_QUERIES,
    )

    # ========================================================
    # 3. Numerical stability
    # ========================================================

    stability_results = benchmark_numerical_stability(
        num_candidates=ACCEPTANCE_CANDIDATES,
    )

    # ========================================================
    # 4. Acceptance
    # ========================================================

    acceptance = verify_acceptance(
        performance_results=pipeline_results,
        stability_results=stability_results,
    )

    print("\n" + "=" * 76)
    print("QA-R2 ACCEPTANCE VERIFICATION")
    print("=" * 76)

    print(
        f"\nAcceptance pool: "
        f"N={acceptance.candidate_pool:,}"
    )

    print(
        f"  Average latency : "
        f"<= {acceptance.latency_target_ms:.3f} ms | "
        f"Actual worst-method: "
        f"{acceptance.latency_actual_ms:.3f} ms | "
        f"[{'PASS' if acceptance.latency_pass else 'FAIL'}]"
    )

    print(
        f"  Throughput      : "
        f">= {acceptance.throughput_target_qps:.1f} QPS | "
        f"Actual worst-method: "
        f"{acceptance.throughput_actual_qps:.1f} QPS | "
        f"[{'PASS' if acceptance.throughput_pass else 'FAIL'}]"
    )

    print(
        f"  Numerical       : "
        f"[{'PASS' if acceptance.numerical_stability_pass else 'FAIL'}]"
    )

    print("\n" + "-" * 76)

    if acceptance.overall_pass:
        print("QA-R2 RESULT: PASS")
    else:
        print("QA-R2 RESULT: FAIL")

    print("-" * 76)

    # ========================================================
    # 5. Save artifact
    # ========================================================

    save_report(
        primitive_results=primitive_results,
        pipeline_results=pipeline_results,
        stability_results=stability_results,
        acceptance=acceptance,
    )

    # ========================================================
    # 6. Return non-zero exit code on failure
    # ========================================================

    if not acceptance.overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()