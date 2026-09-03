"""
QA-R3 — Fusion Engine Benchmark Suite
======================================

Benchmark hiệu năng độc lập cho QA-R3 MultiModalFusionEngine.

Đo:
    1. Latency: Min / Mean / P50 / P95 / P99 / Max
    2. Scalability theo Candidate Pool N
    3. Scalability theo Target Top-K
    4. Greedy vs MMR
    5. Throughput (QPS)
    6. Correctness contract
    7. Boundary acceptance
    8. JSON report
    9. CI-friendly exit status

Modes:
    Smoke:
        - Chạy nhanh để kiểm tra benchmark và regression.
        - N=[100, 500, 1000]
        - K=[10, 50]
        - 30 timed iterations.

    Full:
        - Đo đầy đủ để lấy số liệu báo cáo QA-R3.
        - N=[100, 300, 500, 1000]
        - K=[10, 20, 50]
        - 50 timed iterations.

Chạy từ root repository:

    python -u -m scripts.benchmark_qa_r3_fusion_engine --smoke

    python -u -m scripts.benchmark_qa_r3_fusion_engine --full

Nếu không truyền mode:
    mặc định chạy smoke.

Design constraints:
    - Không thay đổi scripts.fusion_engine.
    - Không thay đổi selection logic.
    - Benchmark chỉ đo thời gian select_top_k().
    - Không hard-code performance guarantee vào engine.
    - Deterministic mock data.
    - Warmup được tách khỏi timed iterations.
    - Correctness được kiểm tra ngoài timed section.
    - Garbage collection được disable ONLY trong timed section.
    - Candidate pools được generate trước benchmark.
    - Case execution order deterministic nhưng được shuffle.
    - Acceptance chỉ xét boundary workload:
          N = MAX_CANDIDATES
          K = MAX_TOP_K
          GREEDY + MMR
    - Các workload khác chỉ phục vụ scalability/regression.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from scripts.fusion_engine import (
    CandidateMetadata,
    MultiModalFusionEngine,
    SelectionMethod,
    SemanticQuery,
)


# ============================================================================
# CONFIG
# ============================================================================

REPORT_DIR = Path("artifacts/qa_r3")

SMOKE_REPORT_PATH = (
    REPORT_DIR / "benchmark_qa_r3_fusion_engine_smoke.json"
)

FULL_REPORT_PATH = (
    REPORT_DIR / "benchmark_qa_r3_fusion_engine.json"
)


# ----------------------------------------------------------------------------
# Smoke benchmark
# ----------------------------------------------------------------------------

SMOKE_POOL_SIZES = [
    100,
    500,
    1_000,
]

SMOKE_TOP_K_VALUES = [
    10,
    50,
]

SMOKE_WARMUP = 5
SMOKE_ITERATIONS = 30


# ----------------------------------------------------------------------------
# Full benchmark
# ----------------------------------------------------------------------------

FULL_POOL_SIZES = [
    100,
    300,
    500,
    1_000,
]

FULL_TOP_K_VALUES = [
    10,
    20,
    50,
]

FULL_WARMUP = 5
FULL_ITERATIONS = 50


# ----------------------------------------------------------------------------
# QA-R3 boundary contract
# ----------------------------------------------------------------------------

MAX_CANDIDATES = 1_000
MAX_TOP_K = 50


# ----------------------------------------------------------------------------
# Performance targets
#
# Đây là benchmark acceptance target.
# KHÔNG phải performance guarantee của engine.
# ----------------------------------------------------------------------------

P95_TARGET_MS = 10.0
QPS_TARGET = 100.0


# ----------------------------------------------------------------------------
# Engine configuration
# ----------------------------------------------------------------------------

SAGE_LAMBDA = 0.3
TIME_THRESHOLD_SEC = 3.0
LAMBDA_MMR = 0.7
REDUNDANCY_PENALTY_WEIGHT = 0.10


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------

QUERY_SEED = 2026
ORDER_SEED = 20260902


# ============================================================================
# MOCK VOCABULARY
# ============================================================================

VOCAB_ENTITIES = [
    f"entity_{i}"
    for i in range(50)
]

VOCAB_ACTIONS = [
    f"action_{i}"
    for i in range(50)
]

VOCAB_ATTRIBUTES = [
    f"attribute_{i}"
    for i in range(30)
]

VOCAB_RELATIONS = [
    f"relation_{i}"
    for i in range(30)
]


# ============================================================================
# RESULT CONTRACTS
# ============================================================================


@dataclass(slots=True, frozen=True)
class BenchmarkResult:
    n_candidates: int
    top_k: int
    method: str
    iterations: int

    min_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    qps: float

    output_size_ok: bool
    unique_ids_ok: bool
    finite_scores_ok: bool
    deterministic_ok: bool

    correctness_pass: bool
    latency_pass: bool
    throughput_pass: bool

    passed: bool


@dataclass(slots=True, frozen=True)
class AcceptanceResult:
    candidate_pool: int
    max_top_k: int

    methods: List[str]

    p95_target_ms: float
    worst_p95_ms: float
    worst_p95_method: str
    worst_p95_n: int
    worst_p95_k: int
    latency_pass: bool

    qps_target: float
    worst_qps: float
    worst_qps_method: str
    worst_qps_n: int
    worst_qps_k: int
    throughput_pass: bool

    correctness_pass: bool

    overall_pass: bool


@dataclass(slots=True)
class BenchmarkReport:
    mode: str
    config: Dict[str, object]

    performance: List[BenchmarkResult] = field(
        default_factory=list
    )

    acceptance: AcceptanceResult | None = None


# ============================================================================
# MOCK QUERY
# ============================================================================


def generate_mock_query(
    seed: int = QUERY_SEED,
) -> SemanticQuery:
    """
    Tạo SemanticQuery deterministic.

    Tổng constraints:

        3 entities
        5 actions
        2 attributes
        2 relations

        = 12 constraints
    """

    rng = np.random.default_rng(seed)

    return SemanticQuery(
        entities=frozenset(
            rng.choice(
                VOCAB_ENTITIES,
                size=3,
                replace=False,
            )
        ),
        actions=frozenset(
            rng.choice(
                VOCAB_ACTIONS,
                size=5,
                replace=False,
            )
        ),
        attributes=frozenset(
            rng.choice(
                VOCAB_ATTRIBUTES,
                size=2,
                replace=False,
            )
        ),
        relations=frozenset(
            rng.choice(
                VOCAB_RELATIONS,
                size=2,
                replace=False,
            )
        ),
    )


# ============================================================================
# MOCK CANDIDATE POOL
# ============================================================================


def generate_mock_candidate_pool(
    n_candidates: int,
    feature_dim: int = 512,
    seed: int = 42,
) -> Tuple[
    Dict[str, float],
    Dict[str, CandidateMetadata],
]:
    """
    Tạo deterministic candidate pool.

    Mỗi candidate gồm:
        - fused relevance score
        - semantic labels
        - timestamp
        - normalized feature vector

    Candidate generation nằm ngoài timed benchmark.
    """

    if n_candidates <= 0:
        raise ValueError(
            "n_candidates must be > 0"
        )

    rng = np.random.default_rng(seed)

    fused_scores: Dict[str, float] = {}
    metadata_map: Dict[str, CandidateMetadata] = {}

    for i in range(n_candidates):

        candidate_id = f"cand_{i:05d}"

        # Mỗi video có 20 frames.
        video_id = f"vid_{i // 20:04d}"

        timestamp = float(
            (i % 20) * 0.5
        )

        # --------------------------------------------------------
        # Semantic labels
        # --------------------------------------------------------

        n_entities = int(
            rng.integers(0, 3)
        )

        n_actions = int(
            rng.integers(0, 4)
        )

        n_attributes = int(
            rng.integers(0, 2)
        )

        n_relations = int(
            rng.integers(0, 2)
        )

        entities = (
            frozenset(
                rng.choice(
                    VOCAB_ENTITIES,
                    size=n_entities,
                    replace=False,
                )
            )
            if n_entities > 0
            else frozenset()
        )

        actions = (
            frozenset(
                rng.choice(
                    VOCAB_ACTIONS,
                    size=n_actions,
                    replace=False,
                )
            )
            if n_actions > 0
            else frozenset()
        )

        attributes = (
            frozenset(
                rng.choice(
                    VOCAB_ATTRIBUTES,
                    size=n_attributes,
                    replace=False,
                )
            )
            if n_attributes > 0
            else frozenset()
        )

        relations = (
            frozenset(
                rng.choice(
                    VOCAB_RELATIONS,
                    size=n_relations,
                    replace=False,
                )
            )
            if n_relations > 0
            else frozenset()
        )

        # --------------------------------------------------------
        # Feature vector
        # --------------------------------------------------------

        vector = rng.standard_normal(
            feature_dim
        ).astype(np.float32)

        norm = float(
            np.linalg.norm(vector)
        )

        if norm > 1e-8:
            vector /= norm

        metadata_map[candidate_id] = CandidateMetadata(
            candidate_id=candidate_id,
            video_id=video_id,
            frame_index=i,
            timestamp_sec=timestamp,
            entities=entities,
            actions=actions,
            attributes=attributes,
            relations=relations,
            feature_vector=vector,
        )

        # Upstream fused score đã được normalize.
        fused_scores[candidate_id] = float(
            rng.uniform(
                0.30,
                0.99,
            )
        )

    return (
        fused_scores,
        metadata_map,
    )


# ============================================================================
# SELECTION HELPER
# ============================================================================


def run_selection(
    *,
    engine: MultiModalFusionEngine,
    fused_scores: Dict[str, float],
    metadata_map: Dict[str, CandidateMetadata],
    query: SemanticQuery,
    top_k: int,
    selection_method: SelectionMethod,
):
    """
    Thin wrapper để benchmark call site luôn giống nhau.

    Wrapper này nằm ngoài logic engine.
    """

    return engine.select_top_k(
        fused_scores=fused_scores,
        metadata_map=metadata_map,
        query=query,
        top_k=top_k,
        selection_method=selection_method,
    )


# ============================================================================
# CORRECTNESS
# ============================================================================


def validate_selection_output(
    result: List,
    expected_top_k: int,
    fused_scores: Dict[str, float],
) -> Tuple[
    bool,
    bool,
    bool,
]:
    """
    Validate output contract.

    Returns:
        output_size_ok
        unique_ids_ok
        finite_scores_ok
    """

    expected_size = min(
        expected_top_k,
        len(fused_scores),
    )

    output_size_ok = (
        len(result) == expected_size
    )

    ids = [
        item.candidate_id
        for item in result
    ]

    unique_ids_ok = (
        len(ids) == len(set(ids))
    )

    finite_scores_ok = all(
        np.isfinite(
            item.fused_score
        )
        and np.isfinite(
            item.coverage_score
        )
        and np.isfinite(
            item.sage_score
        )
        for item in result
    )

    return (
        output_size_ok,
        unique_ids_ok,
        finite_scores_ok,
    )


def check_determinism(
    engine: MultiModalFusionEngine,
    fused_scores: Dict[str, float],
    metadata_map: Dict[str, CandidateMetadata],
    query: SemanticQuery,
    top_k: int,
    selection_method: SelectionMethod,
) -> bool:
    """
    Kiểm tra deterministic selection.

    Chạy hoàn toàn ngoài timed section.
    """

    result_a = run_selection(
        engine=engine,
        fused_scores=fused_scores,
        metadata_map=metadata_map,
        query=query,
        top_k=top_k,
        selection_method=selection_method,
    )

    result_b = run_selection(
        engine=engine,
        fused_scores=fused_scores,
        metadata_map=metadata_map,
        query=query,
        top_k=top_k,
        selection_method=selection_method,
    )

    ids_a = [
        item.candidate_id
        for item in result_a
    ]

    ids_b = [
        item.candidate_id
        for item in result_b
    ]

    return ids_a == ids_b


# ============================================================================
# SINGLE BENCHMARK
# ============================================================================


def run_single_benchmark(
    *,
    engine: MultiModalFusionEngine,
    fused_scores: Dict[str, float],
    metadata_map: Dict[str, CandidateMetadata],
    query: SemanticQuery,
    top_k: int,
    selection_method: SelectionMethod,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    """
    Benchmark một combination:

        N × K × selection_method

    Correctness:
        - chạy trước timed section

    Timing:
        - chỉ đo select_top_k()
        - GC disabled trong timed section
        - candidate generation không được tính
    """

    if iterations <= 0:
        raise ValueError(
            "iterations must be > 0"
        )

    if warmup < 0:
        raise ValueError(
            "warmup must be >= 0"
        )

    # ========================================================================
    # CORRECTNESS
    # ========================================================================

    reference_result = run_selection(
        engine=engine,
        fused_scores=fused_scores,
        metadata_map=metadata_map,
        query=query,
        top_k=top_k,
        selection_method=selection_method,
    )

    (
        output_size_ok,
        unique_ids_ok,
        finite_scores_ok,
    ) = validate_selection_output(
        result=reference_result,
        expected_top_k=top_k,
        fused_scores=fused_scores,
    )

    deterministic_ok = check_determinism(
        engine=engine,
        fused_scores=fused_scores,
        metadata_map=metadata_map,
        query=query,
        top_k=top_k,
        selection_method=selection_method,
    )

    correctness_pass = (
        output_size_ok
        and unique_ids_ok
        and finite_scores_ok
        and deterministic_ok
    )

    # ========================================================================
    # WARMUP
    # ========================================================================

    for _ in range(warmup):

        run_selection(
            engine=engine,
            fused_scores=fused_scores,
            metadata_map=metadata_map,
            query=query,
            top_k=top_k,
            selection_method=selection_method,
        )

    # ========================================================================
    # TIMED SECTION
    # ========================================================================

    latencies_sec: List[float] = []

    gc_was_enabled = gc.isenabled()

    if gc_was_enabled:
        gc.disable()

    try:

        for _ in range(iterations):

            t0 = time.perf_counter()

            run_selection(
                engine=engine,
                fused_scores=fused_scores,
                metadata_map=metadata_map,
                query=query,
                top_k=top_k,
                selection_method=selection_method,
            )

            t1 = time.perf_counter()

            latencies_sec.append(
                t1 - t0
            )

    finally:

        if gc_was_enabled:
            gc.enable()

    # ========================================================================
    # STATISTICS
    # ========================================================================

    latencies_ms = [
        value * 1000.0
        for value in latencies_sec
    ]

    latencies_ms.sort()

    min_ms = float(
        min(latencies_ms)
    )

    mean_ms = float(
        statistics.mean(
            latencies_ms
        )
    )

    p50_ms = float(
        np.percentile(
            latencies_ms,
            50,
        )
    )

    p95_ms = float(
        np.percentile(
            latencies_ms,
            95,
        )
    )

    p99_ms = float(
        np.percentile(
            latencies_ms,
            99,
        )
    )

    max_ms = float(
        max(latencies_ms)
    )

    qps = (
        float(
            1000.0 / mean_ms
        )
        if mean_ms > 0.0
        else 0.0
    )

    # ========================================================================
    # PER-CASE PERFORMANCE
    #
    # Chỉ phản ánh case hiện tại.
    # Không phải acceptance scope.
    # ========================================================================

    latency_pass = (
        p95_ms <= P95_TARGET_MS
    )

    throughput_pass = (
        qps >= QPS_TARGET
    )

    passed = (
        correctness_pass
        and latency_pass
        and throughput_pass
    )

    return BenchmarkResult(
        n_candidates=len(fused_scores),
        top_k=top_k,
        method=selection_method.value,
        iterations=iterations,

        min_ms=min_ms,
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        max_ms=max_ms,

        qps=qps,

        output_size_ok=output_size_ok,
        unique_ids_ok=unique_ids_ok,
        finite_scores_ok=finite_scores_ok,
        deterministic_ok=deterministic_ok,

        correctness_pass=correctness_pass,
        latency_pass=latency_pass,
        throughput_pass=throughput_pass,

        passed=passed,
    )


# ============================================================================
# ACCEPTANCE
# ============================================================================


def build_acceptance_result(
    results: List[BenchmarkResult],
) -> AcceptanceResult:
    """
    Aggregate QA-R3 acceptance.

    Acceptance scope:

        N = MAX_CANDIDATES
        K = MAX_TOP_K

    Cả GREEDY và MMR đều phải đạt:

        P95 <= 10 ms
        QPS >= 100
        Correctness PASS

    Các workload khác KHÔNG tham gia acceptance.
    """

    boundary_results = [
        result
        for result in results
        if (
            result.n_candidates == MAX_CANDIDATES
            and result.top_k == MAX_TOP_K
            and result.method in {
                SelectionMethod.GREEDY.value,
                SelectionMethod.MMR.value,
            }
        )
    ]

    expected_methods = {
        SelectionMethod.GREEDY.value,
        SelectionMethod.MMR.value,
    }

    actual_methods = {
        result.method
        for result in boundary_results
    }

    boundary_complete = (
        actual_methods == expected_methods
    )

    if not boundary_results or not boundary_complete:

        return AcceptanceResult(
            candidate_pool=MAX_CANDIDATES,
            max_top_k=MAX_TOP_K,

            methods=sorted(
                expected_methods
            ),

            p95_target_ms=P95_TARGET_MS,
            worst_p95_ms=float("inf"),
            worst_p95_method="N/A",
            worst_p95_n=MAX_CANDIDATES,
            worst_p95_k=MAX_TOP_K,
            latency_pass=False,

            qps_target=QPS_TARGET,
            worst_qps=0.0,
            worst_qps_method="N/A",
            worst_qps_n=MAX_CANDIDATES,
            worst_qps_k=MAX_TOP_K,
            throughput_pass=False,

            correctness_pass=False,

            overall_pass=False,
        )

    # ========================================================================
    # Worst P95
    # ========================================================================

    worst_p95_result = max(
        boundary_results,
        key=lambda result: result.p95_ms,
    )

    worst_p95_ms = (
        worst_p95_result.p95_ms
    )

    latency_pass = (
        worst_p95_ms <= P95_TARGET_MS
    )

    # ========================================================================
    # Worst QPS
    # ========================================================================

    worst_qps_result = min(
        boundary_results,
        key=lambda result: result.qps,
    )

    worst_qps = (
        worst_qps_result.qps
    )

    throughput_pass = (
        worst_qps >= QPS_TARGET
    )

    # ========================================================================
    # Correctness
    # ========================================================================

    correctness_pass = all(
        result.correctness_pass
        for result in boundary_results
    )

    overall_pass = (
        latency_pass
        and throughput_pass
        and correctness_pass
    )

    return AcceptanceResult(
        candidate_pool=MAX_CANDIDATES,
        max_top_k=MAX_TOP_K,

        methods=sorted(
            expected_methods
        ),

        p95_target_ms=P95_TARGET_MS,
        worst_p95_ms=worst_p95_ms,
        worst_p95_method=(
            worst_p95_result.method
        ),
        worst_p95_n=(
            worst_p95_result.n_candidates
        ),
        worst_p95_k=(
            worst_p95_result.top_k
        ),
        latency_pass=latency_pass,

        qps_target=QPS_TARGET,
        worst_qps=worst_qps,
        worst_qps_method=(
            worst_qps_result.method
        ),
        worst_qps_n=(
            worst_qps_result.n_candidates
        ),
        worst_qps_k=(
            worst_qps_result.top_k
        ),
        throughput_pass=throughput_pass,

        correctness_pass=correctness_pass,

        overall_pass=overall_pass,
    )


# ============================================================================
# BENCHMARK ORDER
# ============================================================================


def build_case_order(
    pool_sizes: List[int],
    top_k_values: List[int],
) -> List[
    Tuple[
        int,
        int,
        SelectionMethod,
    ]
]:
    """
    Tạo execution order deterministic.

    Toàn bộ cases được shuffle bằng seed cố định.

    Lưu ý:
        Đây là randomized case order để giảm systematic
        order bias; không phải statistical cross-run balancing.
    """

    cases: List[
        Tuple[
            int,
            int,
            SelectionMethod,
        ]
    ] = []

    for n_candidates in pool_sizes:

        for top_k in top_k_values:

            if top_k > n_candidates:
                continue

            for method in (
                SelectionMethod.GREEDY,
                SelectionMethod.MMR,
            ):
                cases.append(
                    (
                        n_candidates,
                        top_k,
                        method,
                    )
                )

    rng = random.Random(
        ORDER_SEED
    )

    rng.shuffle(cases)

    return cases


# ============================================================================
# BENCHMARK SUITE
# ============================================================================


def run_benchmark_suite(
    *,
    mode: str,
    pool_sizes: List[int],
    top_k_values: List[int],
    warmup: int,
    iterations: int,
) -> BenchmarkReport:
    """
    Chạy toàn bộ benchmark suite.
    """

    engine = MultiModalFusionEngine(
        sage_lambda=SAGE_LAMBDA,
        time_threshold_sec=TIME_THRESHOLD_SEC,
        lambda_mmr=LAMBDA_MMR,
        redundancy_penalty_weight=(
            REDUNDANCY_PENALTY_WEIGHT
        ),
    )

    query = generate_mock_query(
        seed=QUERY_SEED
    )

    cases = build_case_order(
        pool_sizes=pool_sizes,
        top_k_values=top_k_values,
    )

    total_cases = len(cases)

    results: List[BenchmarkResult] = []

    # ========================================================================
    # HEADER
    # ========================================================================

    print("=" * 130)
    print(
        " QA-R3 FUSION ENGINE BENCHMARK ".center(
            130,
            "=",
        )
    )
    print("=" * 130)

    print(
        f"Mode             : {mode.upper()}"
    )

    print(
        f"Warmup           : {warmup}"
    )

    print(
        f"Iterations       : {iterations}"
    )

    print(
        f"Pool sizes       : {pool_sizes}"
    )

    print(
        f"Top-K values     : {top_k_values}"
    )

    print(
        f"P95 target       : <= {P95_TARGET_MS:.2f} ms"
    )

    print(
        f"QPS target       : >= {QPS_TARGET:.1f}"
    )

    print(
        f"Acceptance case  : "
        f"N={MAX_CANDIDATES}, "
        f"K={MAX_TOP_K}"
    )

    print(
        f"Total cases      : {total_cases}"
    )

    print(
        f"Query constraints: "
        f"{len(query.entities)} entities + "
        f"{len(query.actions)} actions + "
        f"{len(query.attributes)} attributes + "
        f"{len(query.relations)} relations"
    )

    print()

    print(
        f"{'Method':<8} | "
        f"{'N':>5} | "
        f"{'K':>3} | "
        f"{'Min':>8} | "
        f"{'Mean':>8} | "
        f"{'P50':>8} | "
        f"{'P95':>8} | "
        f"{'P99':>8} | "
        f"{'Max':>8} | "
        f"{'QPS':>8} | "
        f"{'Contract':>9} | "
        f"{'Perf':>6}"
    )

    print("-" * 130)

    # ========================================================================
    # VALIDATE CONFIG
    # ========================================================================

    for n_candidates in pool_sizes:

        if n_candidates <= 0:
            raise ValueError(
                f"Candidate pool must be > 0: {n_candidates}"
            )

        if n_candidates > MAX_CANDIDATES:
            raise ValueError(
                f"Candidate pool {n_candidates} exceeds "
                f"QA-R3 maximum {MAX_CANDIDATES}"
            )

    for top_k in top_k_values:

        if top_k <= 0:
            raise ValueError(
                f"Top-K must be > 0: {top_k}"
            )

        if top_k > MAX_TOP_K:
            raise ValueError(
                f"Top-K {top_k} exceeds "
                f"QA-R3 maximum {MAX_TOP_K}"
            )

    # ========================================================================
    # CACHE CANDIDATE POOLS
    #
    # Candidate generation is completely outside timing.
    # ========================================================================

    candidate_pools: Dict[
        int,
        Tuple[
            Dict[str, float],
            Dict[str, CandidateMetadata],
        ],
    ] = {}

    for n_candidates in pool_sizes:

        candidate_pools[n_candidates] = (
            generate_mock_candidate_pool(
                n_candidates=n_candidates,
                seed=n_candidates,
            )
        )

    # ========================================================================
    # EXECUTE CASES
    # ========================================================================

    for case_index, (
        n_candidates,
        top_k,
        method,
    ) in enumerate(
        cases,
        start=1,
    ):

        (
            fused_scores,
            metadata_map,
        ) = candidate_pools[n_candidates]

        print(
            f"\rRunning case "
            f"{case_index}/{total_cases}...",
            end="",
            flush=True,
        )

        result = run_single_benchmark(
            engine=engine,
            fused_scores=fused_scores,
            metadata_map=metadata_map,
            query=query,
            top_k=top_k,
            selection_method=method,
            warmup=warmup,
            iterations=iterations,
        )

        results.append(
            result
        )

        print(
            "\r"
            + " " * 70
            + "\r",
            end="",
        )

        contract_ok = (
            result.output_size_ok
            and result.unique_ids_ok
            and result.finite_scores_ok
            and result.deterministic_ok
        )

        performance_ok = (
            result.latency_pass
            and result.throughput_pass
        )

        print(
            f"{result.method.upper():<8} | "
            f"{result.n_candidates:>5} | "
            f"{result.top_k:>3} | "
            f"{result.min_ms:>8.3f} | "
            f"{result.mean_ms:>8.3f} | "
            f"{result.p50_ms:>8.3f} | "
            f"{result.p95_ms:>8.3f} | "
            f"{result.p99_ms:>8.3f} | "
            f"{result.max_ms:>8.3f} | "
            f"{result.qps:>8.1f} | "
            f"{'PASS' if contract_ok else 'FAIL':>9} | "
            f"{'PASS' if performance_ok else 'FAIL':>6}"
        )

    # ========================================================================
    # ACCEPTANCE
    # ========================================================================

    acceptance = build_acceptance_result(
        results
    )

    # ========================================================================
    # CONFIG
    # ========================================================================

    config = {
        "max_candidates": MAX_CANDIDATES,
        "max_top_k": MAX_TOP_K,

        "p95_target_ms": P95_TARGET_MS,
        "qps_target": QPS_TARGET,

        "pool_sizes": pool_sizes,
        "top_k_values": top_k_values,

        "warmup": warmup,
        "iterations": iterations,

        "query_seed": QUERY_SEED,
        "order_seed": ORDER_SEED,

        "acceptance_scope": {
            "candidate_pool": MAX_CANDIDATES,
            "top_k": MAX_TOP_K,
            "methods": [
                SelectionMethod.GREEDY.value,
                SelectionMethod.MMR.value,
            ],
        },

        "engine": {
            "sage_lambda": SAGE_LAMBDA,
            "time_threshold_sec": TIME_THRESHOLD_SEC,
            "lambda_mmr": LAMBDA_MMR,
            "redundancy_penalty_weight": (
                REDUNDANCY_PENALTY_WEIGHT
            ),
        },
    }

    report = BenchmarkReport(
        mode=mode,
        config=config,
        performance=results,
        acceptance=acceptance,
    )

    # ========================================================================
    # ACCEPTANCE OUTPUT
    # ========================================================================

    print()

    print("=" * 110)
    print(
        " ACCEPTANCE ".center(
            110,
            "=",
        )
    )
    print("=" * 110)

    print(
        f"Boundary      : "
        f"N={acceptance.candidate_pool}, "
        f"K={acceptance.max_top_k}"
    )

    print(
        f"Methods       : "
        f"{', '.join(acceptance.methods)}"
    )

    print(
        f"P95 latency   : "
        f"{acceptance.worst_p95_ms:.3f} ms "
        f"(target <= "
        f"{acceptance.p95_target_ms:.3f}) "
        f"-> "
        f"{'PASS' if acceptance.latency_pass else 'FAIL'} "
        f"[{acceptance.worst_p95_method}]"
    )

    print(
        f"Worst QPS     : "
        f"{acceptance.worst_qps:.1f} "
        f"(target >= "
        f"{acceptance.qps_target:.1f}) "
        f"-> "
        f"{'PASS' if acceptance.throughput_pass else 'FAIL'} "
        f"[{acceptance.worst_qps_method}]"
    )

    print(
        f"Correctness   : "
        f"{'PASS' if acceptance.correctness_pass else 'FAIL'}"
    )

    print(
        f"OVERALL       : "
        f"{'PASS' if acceptance.overall_pass else 'FAIL'}"
    )

    print("=" * 110)

    return report


# ============================================================================
# REPORT
# ============================================================================


def save_report(
    report: BenchmarkReport,
    path: Path,
) -> None:
    """
    Lưu benchmark report thành JSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = asdict(
        report
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[REPORT] {path}"
    )


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "QA-R3 Fusion Engine benchmark"
        )
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Quick benchmark: "
            "N=[100,500,1000], "
            "K=[10,50], "
            "30 iterations."
        ),
    )

    group.add_argument(
        "--full",
        action="store_true",
        help=(
            "Full benchmark: "
            "N=[100,300,500,1000], "
            "K=[10,20,50], "
            "50 iterations."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    args = parse_args()

    if args.full:

        mode = "full"

        pool_sizes = FULL_POOL_SIZES
        top_k_values = FULL_TOP_K_VALUES

        warmup = FULL_WARMUP
        iterations = FULL_ITERATIONS

        report_path = FULL_REPORT_PATH

    else:

        mode = "smoke"

        pool_sizes = SMOKE_POOL_SIZES
        top_k_values = SMOKE_TOP_K_VALUES

        warmup = SMOKE_WARMUP
        iterations = SMOKE_ITERATIONS

        report_path = SMOKE_REPORT_PATH

    report = run_benchmark_suite(
        mode=mode,
        pool_sizes=pool_sizes,
        top_k_values=top_k_values,
        warmup=warmup,
        iterations=iterations,
    )

    save_report(
        report,
        report_path,
    )

    # ========================================================================
    # CI / REGRESSION EXIT STATUS
    # ========================================================================

    raise SystemExit(
        0
        if (
            report.acceptance is not None
            and report.acceptance.overall_pass
        )
        else 1
    )


if __name__ == "__main__":
    main()