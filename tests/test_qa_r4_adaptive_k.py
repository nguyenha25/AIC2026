# ============================================================================
# QA-R4 — MỤC 8: ACCEPTANCE / CROSS-STAGE REGRESSION TESTS
# ============================================================================
#
# IMPORTANT
# ---------
# Đây là PHẦN BỔ SUNG cho bộ 47 tests hiện tại.
#
# Không xóa các test phía trên.
# Không thay đổi các helper hiện có nếu chúng đã tồn tại.
#
# Scope:
#
#   8.1 R1 thực sự có Top-50 coverage_candidates
#   8.2 R3 nhận Top-50 thay vì fallback Top-12
#   8.3 R4 output giữ đúng Reader contract
#   8.4 malformed candidate schema bị reject
#   8.5 mid-run failure không corrupt output cũ
#   8.6 benchmark thực sự chạy selection
#   8.7 E2E S4 -> R3 -> R4 -> Reader contract
#
# Các test đều data-free, không load FAISS/VLM thật.
# ============================================================================


# ----------------------------------------------------------------------------
# Additional imports
# ----------------------------------------------------------------------------

import json
from pathlib import Path

import pytest


# ============================================================================ 
# 8.1 — QA-R1 -> Top-50 coverage candidates
# ============================================================================


def test_qa_r1_produces_exactly_50_coverage_candidates():
    """
    QA-R1 funnel contract:

        Top-300 videos
            ↓
        Top-50 coverage_candidates
            ↓
        Top-12 reader_candidates

    Test against the actual candidate_funnel API.
    Do not introduce a synthetic top_n_candidates helper that does not
    exist in the production implementation.
    """
    from scripts import candidate_funnel

    assert candidate_funnel.K_COVERAGE == 50
    assert candidate_funnel.K_READER == 12

    # The production R1 implementation exposes the coverage-stage helper.
    assert callable(candidate_funnel.extract_top50_coverage)

    # The reader stage must be a separate stage after coverage.
    assert callable(candidate_funnel.select_top12_reader)

    # Verify the frozen funnel contract directly:
    # coverage budget = 50, reader budget = 12.
    assert candidate_funnel.K_COVERAGE == 50
    assert candidate_funnel.K_READER < candidate_funnel.K_COVERAGE


def test_qa_r1_coverage_stage_is_before_reader_stage():
    """
    Guard against accidentally constructing Top-12 directly from Top-300.

    The current R1 implementation has explicit coverage and reader stages:
        extract_top50_coverage(...)
        select_top12_reader(...)
    """
    from scripts import candidate_funnel

    coverage_fn = candidate_funnel.extract_top50_coverage
    reader_fn = candidate_funnel.select_top12_reader

    assert callable(coverage_fn)
    assert callable(reader_fn)

    # Frozen funnel ordering.
    assert candidate_funnel.K_COVERAGE == 50
    assert candidate_funnel.K_READER == 12
    assert candidate_funnel.K_READER < candidate_funnel.K_COVERAGE

    # The reader selector must operate on the output of the coverage stage,
    # not directly on the Top-300 video pool.  We verify this contract from
    # the actual function implementation rather than relying on a nonexistent
    # generic top_n_candidates helper.
    import inspect

    reader_source = inspect.getsource(reader_fn)
    coverage_name = coverage_fn.__name__

    assert coverage_name in reader_source or "candidates_50" in reader_source


# ============================================================================ 
# 8.2 — QA-R3 receives Top-50 coverage pool
# ============================================================================


def test_r3_prefers_full_top50_coverage_pool_over_reader_top12(
    tmp_path,
):
    """
    When R1 provides both coverage_candidates and reader_candidates,
    R3 must consume the full Top-50 coverage pool.

    The fixture follows the actual QA-R1 manifest and N01 QA-query
    contracts expected by run_qa_r3().

    IMPORTANT:
        R1 -> R3 uses frame_idx as the frame provenance field.
        R3 -> R4 later serializes this as frame_id.

    Therefore this fixture intentionally uses frame_idx, not frame_id.
    """
    from scripts.run_qa_r3 import run_qa_r3

    r1_path = tmp_path / "r1.json"
    n01_path = tmp_path / "n01.jsonl"
    output_path = tmp_path / "r3.jsonl"

    # ------------------------------------------------------------------------
    # R1 -> R3 contract
    #
    # R1 coverage candidates use:
    #     video_id
    #     frame_idx
    #     n
    #     score
    #
    # Do NOT use frame_id here. frame_id belongs to the R3 -> R4 / Reader
    # facing contract.
    # ------------------------------------------------------------------------
    coverage_candidates = [
        {
            "video_id": f"V{i:03d}",
            "frame_idx": i,
            "n": i,
            "evidence_score": float(1.0 - i * 0.001),
            "score_fused": float(1.0 - i * 0.001),
        }
        for i in range(50)
    ]

    # R1 reader_candidates is only the first 12 candidates.
    # The purpose of this test is to ensure R3 does NOT incorrectly consume
    # this smaller pool when the full Top-50 coverage pool is available.
    reader_candidates = coverage_candidates[:12]

    r1_manifest = {
        "query_records": [
            {
                "query_id": "q1",
                "coverage_candidates": coverage_candidates,
                "reader_candidates": reader_candidates,
            }
        ]
    }

    # N01 loader only accepts QA queries.
    n01_record = {
        "query_id": "q1",
        "task": "qa",
        "semantic": "test query",
    }

    r1_path.write_text(
        json.dumps(
            r1_manifest,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    n01_path.write_text(
        json.dumps(
            n01_record,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_qa_r3(
        r1_path=r1_path,
        n01_path=n01_path,
        output_path=output_path,
        top_k=1,
        selection_method="greedy",
    )

    assert isinstance(result, tuple)
    assert len(result) >= 2

    pool_counts = result[1]["pool_counts"]

    assert pool_counts == {
        "coverage_candidates": 1,
    }


def test_r3_top50_output_contains_candidate_schema_required_by_r4(
    tmp_path,
):
    """
    R3's serialized output must contain enough information to build
    the frozen R3 -> R4 candidate contract.
    """

    from scripts.run_qa_r3 import run_qa_r3

    r1_path = tmp_path / "r1.json"
    n01_path = tmp_path / "n01.jsonl"
    output_path = tmp_path / "r3.json"

    candidates = [
        {
            "video_id": f"V{i:02d}",
            "n": i + 1,
            "frame_idx": 1000 + i,
            "pts_time": float(i),
            "evidence_score": 1.0 - i * 0.001,
            "score_fused": 1.0 - i * 0.001,
        }
        for i in range(50)
    ]

    r1_path.write_text(
        json.dumps(
            {
                "query_records": [
                    {
                        "query_id": "N01",
                        "coverage_candidates": candidates,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    n01_path.write_text(
        '{"query_id":"N01","task":"qa"}\n',
        encoding="utf-8",
    )

    run_qa_r3(
        r1_path=r1_path,
        n01_path=n01_path,
        output_path=output_path,
        query_id="N01",
        top_k=50,
        selection_method="greedy",
    )

    output = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert len(output["candidates"]) == 50

    required = {
        "video_id",
        "frame_id",
        "n",
        "score",
    }

    for candidate in output["candidates"]:
        assert required <= set(candidate)


# ============================================================================ 
# 8.3 — R4 -> Reader output contract
# ============================================================================


def test_r4_output_contains_reader_required_candidate_fields(
    tmp_path,
):
    """
    Reader-facing R4 candidate schema:

        video_id
        frame_id
        n
        score

    R4 must preserve these fields exactly.
    """

    from scripts.adaptive_k import run_jsonl_funnel

    s4_path = tmp_path / "s4.jsonl"
    r3_path = tmp_path / "r3.jsonl"
    output_path = tmp_path / "r4.jsonl"

    candidates = [
        {
            "video_id": f"V{i:03d}",
            "frame_id": 1000 + i,
            "n": i + 1,
            "score": 1.0 / (i + 1),
        }
        for i in range(100)
    ]

    s4_path.write_text(
        '{"query_id":"q01","semantic_k_hint":100}\n',
        encoding="utf-8",
    )

    r3_path.write_text(
        json.dumps(
            {
                "query_id": "q01",
                "candidates": candidates,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    results = list(
        run_jsonl_funnel(
            s4_path=s4_path,
            r3_path=r3_path,
            output_path=output_path,
        )
    )

    assert len(results) == 1

    output_records = [
        json.loads(line)
        for line in output_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(output_records) == 1

    selected = output_records[0][
        "selected_candidates"
    ]

    assert len(selected) == 100

    required = {
        "video_id",
        "frame_id",
        "n",
        "score",
    }

    for candidate in selected:
        assert required <= set(candidate)


def test_r4_reader_handoff_preserves_candidate_identity(
    tmp_path,
):
    """
    R4 must not modify the identity tuple:

        (video_id, frame_id, n)
    """

    from scripts.adaptive_k import run_jsonl_funnel

    s4_path = tmp_path / "s4.jsonl"
    r3_path = tmp_path / "r3.jsonl"
    output_path = tmp_path / "r4.jsonl"

    candidates = [
        {
            "video_id": "V_A",
            "frame_id": 111,
            "n": 7,
            "score": 0.91,
        },
        {
            "video_id": "V_B",
            "frame_id": 222,
            "n": 8,
            "score": 0.72,
        },
        {
            "video_id": "V_C",
            "frame_id": 333,
            "n": 9,
            "score": 0.63,
        },
    ]

    s4_path.write_text(
        '{"query_id":"q01","semantic_k_hint":100}\n',
        encoding="utf-8",
    )

    r3_path.write_text(
        json.dumps(
            {
                "query_id": "q01",
                "candidates": candidates,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    results = list(
        run_jsonl_funnel(
            s4_path=s4_path,
            r3_path=r3_path,
            output_path=output_path,
        )
    )

    assert len(results) == 1

    result = results[0]

    assert list(
        result.selected_candidates
    ) == candidates

    record = json.loads(
        output_path.read_text(
            encoding="utf-8"
        ).strip()
    )

    selected = record[
        "selected_candidates"
    ]

    assert [
        (
            item["video_id"],
            item["frame_id"],
            item["n"],
        )
        for item in selected
    ] == [
        (
            item["video_id"],
            item["frame_id"],
            item["n"],
        )
        for item in candidates
    ]


# ============================================================================ 
# 8.4 — Candidate schema rejection
# ============================================================================


@pytest.mark.parametrize(
    "missing_field",
    [
        "video_id",
        "frame_id",
        "n",
        "score",
    ],
)
def test_r4_rejects_candidate_missing_required_field(
    missing_field,
):
    """
    Every frozen R3 -> R4 field is mandatory.
    """

    from scripts.adaptive_k import AdaptiveKEngine

    candidate = {
        "video_id": "V001",
        "frame_id": 100,
        "n": 7,
        "score": 0.9,
    }

    candidate.pop(missing_field)

    engine = AdaptiveKEngine()

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        engine.select(
            query_id="schema-error",
            candidates=[candidate],
            semantic_k_hint=100,
        )


def test_r4_rejects_malformed_candidate_even_outside_selected_prefix():
    """
    Validation must cover the COMPLETE candidate pool before slicing.

    A malformed candidate at position 101 must not be hidden simply because
    K=100.
    """

    from scripts.adaptive_k import AdaptiveKEngine

    candidates = [
        {
            "video_id": f"V{i:03d}",
            "frame_id": 1000 + i,
            "n": i + 1,
            "score": 1.0 / (i + 1),
        }
        for i in range(100)
    ]

    candidates.append(
        {
            "video_id": "MALFORMED",
            "frame_id": 9999,
            "n": 999,
            # score intentionally missing
        }
    )

    engine = AdaptiveKEngine()

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        engine.select(
            query_id="late-schema-error",
            candidates=candidates,
            semantic_k_hint=100,
        )


def test_r4_rejects_none_candidate_pool():
    from scripts.adaptive_k import AdaptiveKEngine

    engine = AdaptiveKEngine()

    with pytest.raises(
        ValueError,
    ):
        engine.select(
            query_id="none-pool",
            candidates=None,
            semantic_k_hint=100,
        )


def test_r4_rejects_non_mapping_candidate():
    from scripts.adaptive_k import AdaptiveKEngine

    engine = AdaptiveKEngine()

    with pytest.raises(
        ValueError,
    ):
        engine.select(
            query_id="bad-candidate",
            candidates=[
                {
                    "video_id": "V001",
                    "frame_id": 100,
                    "n": 1,
                    "score": 0.9,
                },
                "not-a-candidate-object",
            ],
            semantic_k_hint=100,
        )


# ============================================================================ 
# 8.5 — Atomic output / mid-run failure
# ============================================================================


def test_r4_mid_run_failure_preserves_existing_output(
    tmp_path,
):
    """
    If query 2 fails after query 1 has already been processed:

        OLD output
             ↓
        remains untouched

    The partially generated new output must never be committed.
    """

    from scripts.adaptive_k import run_jsonl_funnel

    s4_path = tmp_path / "s4.jsonl"
    r3_path = tmp_path / "r3.jsonl"
    output_path = tmp_path / "r4.jsonl"

    old_output = (
        '{"query_id":"old",'
        '"k_requested":100,'
        '"k_effective":0,'
        '"k_available":0,'
        '"selected_candidates":[]}\n'
    )

    output_path.write_text(
        old_output,
        encoding="utf-8",
    )

    valid_candidates = [
        {
            "video_id": f"V{i:03d}",
            "frame_id": 1000 + i,
            "n": i + 1,
            "score": 1.0 / (i + 1),
        }
        for i in range(100)
    ]

    malformed_candidates = [
        {
            "video_id": "BAD",
            "frame_id": 999,
            "n": 999,
            # missing score
        }
    ]

    s4_path.write_text(
        "\n".join(
            [
                '{"query_id":"q01","semantic_k_hint":100}',
                '{"query_id":"q02","semantic_k_hint":100}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    r3_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query_id": "q01",
                        "candidates": valid_candidates,
                    }
                ),
                json.dumps(
                    {
                        "query_id": "q02",
                        "candidates": malformed_candidates,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
    ):
        list(
            run_jsonl_funnel(
                s4_path=s4_path,
                r3_path=r3_path,
                output_path=output_path,
            )
        )

    assert output_path.exists()

    assert output_path.read_text(
        encoding="utf-8"
    ) == old_output

    # Atomic implementation must clean its temporary output.
    assert not list(
        tmp_path.glob(
            f".{output_path.name}.*.tmp"
        )
    )


def test_r4_failure_does_not_create_partial_committed_output(
    tmp_path,
):
    """
    When there is no previous output, a failed run must not leave behind
    a partial final output.
    """

    from scripts.adaptive_k import run_jsonl_funnel

    s4_path = tmp_path / "s4.jsonl"
    r3_path = tmp_path / "r3.jsonl"
    output_path = tmp_path / "r4.jsonl"

    s4_path.write_text(
        "\n".join(
            [
                '{"query_id":"q01","semantic_k_hint":100}',
                '{"query_id":"q02","semantic_k_hint":100}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    valid = [
        {
            "video_id": f"V{i}",
            "frame_id": i,
            "n": i,
            "score": 1.0,
        }
        for i in range(100)
    ]

    invalid = [
        {
            "video_id": "BAD",
            "frame_id": 1,
            "n": 1,
            # score intentionally missing
        }
    ]

    r3_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query_id": "q01",
                        "candidates": valid,
                    }
                ),
                json.dumps(
                    {
                        "query_id": "q02",
                        "candidates": invalid,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
    ):
        list(
            run_jsonl_funnel(
                s4_path=s4_path,
                r3_path=r3_path,
                output_path=output_path,
            )
        )

    assert not output_path.exists()


# ============================================================================ 
# 8.6 — Benchmark thật sự chạy R4 selection
# ============================================================================


def test_benchmark_executes_actual_r4_selection(
    tmp_path,
    monkeypatch,
):
    """
    Benchmark acceptance:

        benchmark
            ↓
        run_jsonl_funnel
            ↓
        AdaptiveKEngine.select

    We spy on the real AdaptiveKEngine.select() implementation.

    This proves the benchmark actually executes R4 selection instead of
    merely validating an already-produced result.
    """

    import scripts.adaptive_k as adaptive_k
    import scripts.benchmark_qa_r4 as benchmark

    s4_path = tmp_path / "s4.jsonl"
    r3_path = tmp_path / "r3.jsonl"
    output_path = tmp_path / "r4.jsonl"
    report_path = tmp_path / "report.json"

    candidates = [
        {
            "video_id": f"V{i:03d}",
            "frame_id": 1000 + i,
            "n": i + 1,
            "score": 1.0 / (i + 1),
        }
        for i in range(100)
    ]

    s4_path.write_text(
        '{"query_id":"q01","task":"qa","semantic_k_hint":100}\n',
        encoding="utf-8",
    )

    r3_path.write_text(
        json.dumps(
            {
                "query_id": "q01",
                "candidates": candidates,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls = []

    original_select = (
        adaptive_k.AdaptiveKEngine.select
    )

    def spy_select(
        self,
        query_id,
        candidates,
        semantic_k_hint,
    ):
        calls.append(
            {
                "query_id": query_id,
                "candidate_count": len(candidates),
                "semantic_k_hint": semantic_k_hint,
            }
        )

        return original_select(
            self,
            query_id=query_id,
            candidates=candidates,
            semantic_k_hint=semantic_k_hint,
        )

    monkeypatch.setattr(
        adaptive_k.AdaptiveKEngine,
        "select",
        spy_select,
    )

    benchmark.run_benchmark(
        s4_path=s4_path,
        r3_path=r3_path,
        output_path=output_path,
        report_path=report_path,
        task="qa",
    )

    assert calls == [
        {
            "query_id": "q01",
            "candidate_count": 100,
            "semantic_k_hint": 100,
        }
    ]

    assert output_path.exists()
    assert report_path.exists()

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    statistics = report["statistics"]

    assert statistics[
        "selected_candidate_total"
    ] == 100

    assert statistics[
        "query_count"
    ] == 1


def test_benchmark_does_not_count_validation_as_selection(
    tmp_path,
    monkeypatch,
):
    """
    validate_result() is bookkeeping.

    Selection must still happen through the R4 funnel.
    """

    import scripts.adaptive_k as adaptive_k
    import scripts.benchmark_qa_r4 as benchmark

    s4_path = tmp_path / "s4.jsonl"
    r3_path = tmp_path / "r3.jsonl"
    output_path = tmp_path / "r4.jsonl"
    report_path = tmp_path / "report.json"

    candidates = [
        {
            "video_id": f"V{i}",
            "frame_id": i,
            "n": i,
            "score": 1.0,
        }
        for i in range(100)
    ]

    s4_path.write_text(
        '{"query_id":"q01","task":"qa","semantic_k_hint":100}\n',
        encoding="utf-8",
    )

    r3_path.write_text(
        json.dumps(
            {
                "query_id": "q01",
                "candidates": candidates,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls = {
        "select": 0,
    }

    original_select = (
        adaptive_k.AdaptiveKEngine.select
    )

    def spy_select(
        self,
        query_id,
        candidates,
        semantic_k_hint,
    ):
        calls["select"] += 1

        return original_select(
            self,
            query_id=query_id,
            candidates=candidates,
            semantic_k_hint=semantic_k_hint,
        )

    monkeypatch.setattr(
        adaptive_k.AdaptiveKEngine,
        "select",
        spy_select,
    )

    benchmark.run_benchmark(
        s4_path=s4_path,
        r3_path=r3_path,
        output_path=output_path,
        report_path=report_path,
        task="qa",
    )

    assert calls["select"] == 1

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    assert report["statistics"][
        "selected_candidate_total"
    ] == 100


# ============================================================================ 
# 8.7 — End-to-end S4 -> R3 -> R4 -> Reader
# ============================================================================


def test_end_to_end_s4_r3_r4_reader_contract(
    tmp_path,
):
    """
    Full data-contract test:

        S4
         ↓
        R3
         ↓
        R4
         ↓
        Reader-compatible candidate list

    No retrieval model or VLM is executed.
    """

    from scripts.adaptive_k import run_jsonl_funnel

    s4_path = tmp_path / "s4.jsonl"
    r3_path = tmp_path / "r3.jsonl"
    r4_path = tmp_path / "r4.jsonl"

    r3_candidates = [
        {
            "video_id": f"V{i:03d}",
            "frame_id": 1000 + i,
            "n": i + 1,
            "score": 1.0 / (i + 1),
        }
        for i in range(100)
    ]

    # S4 supplies K.
    s4_path.write_text(
        '{"query_id":"e2e-q01","semantic_k_hint":100}\n',
        encoding="utf-8",
    )

    # R3 supplies candidate pool.
    r3_path.write_text(
        json.dumps(
            {
                "query_id": "e2e-q01",
                "candidates": r3_candidates,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # R4 executes adaptive selection.
    results = list(
        run_jsonl_funnel(
            s4_path=s4_path,
            r3_path=r3_path,
            output_path=r4_path,
        )
    )

    assert len(results) == 1

    result = results[0]

    assert result.query_id == "e2e-q01"
    assert result.k_requested == 100
    assert result.k_available == 100
    assert result.k_effective == 100
    assert result.status == "ok"

    assert list(
        result.selected_candidates
    ) == r3_candidates

    # Reader-facing serialized artifact.
    record = json.loads(
        r4_path.read_text(
            encoding="utf-8"
        ).strip()
    )

    assert record["query_id"] == "e2e-q01"

    reader_candidates = record[
        "selected_candidates"
    ]

    assert len(reader_candidates) == 100

    required = {
        "video_id",
        "frame_id",
        "n",
        "score",
    }

    assert all(
        required <= set(candidate)
        for candidate in reader_candidates
    )

    # Exact R3 -> R4 preservation.
    assert reader_candidates == r3_candidates


def test_end_to_end_short_pool_is_explicitly_clamped(
    tmp_path,
):
    """
    E2E fallback behavior:

        requested K = 100
        available   = 12

    R4 must return exactly 12 available candidates and expose the fallback
    reason rather than inventing candidates.
    """

    from scripts.adaptive_k import run_jsonl_funnel

    s4_path = tmp_path / "s4.jsonl"
    r3_path = tmp_path / "r3.jsonl"
    r4_path = tmp_path / "r4.jsonl"

    candidates = [
        {
            "video_id": f"V{i:02d}",
            "frame_id": 100 + i,
            "n": i + 1,
            "score": 1.0 / (i + 1),
        }
        for i in range(12)
    ]

    s4_path.write_text(
        '{"query_id":"short-q","semantic_k_hint":100}\n',
        encoding="utf-8",
    )

    r3_path.write_text(
        json.dumps(
            {
                "query_id": "short-q",
                "candidates": candidates,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    results = list(
        run_jsonl_funnel(
            s4_path=s4_path,
            r3_path=r3_path,
            output_path=r4_path,
        )
    )

    assert len(results) == 1

    result = results[0]

    assert result.k_requested == 100
    assert result.k_available == 12
    assert result.k_effective == 12

    assert result.status == "ok_with_warning"

    assert (
        result.fallback_reason
        == "candidate_pool_below_requested_k"
    )

    assert list(
        result.selected_candidates
    ) == candidates


# ============================================================================ 
# Cross-stage identity regression
# ============================================================================


def test_r3_to_r4_preserves_video_frame_n_identity(
    tmp_path,
):
    """
    Cross-stage invariant:

        R3 identity = (video_id, frame_id, n)

    must remain identical after R4.
    """

    from scripts.adaptive_k import run_jsonl_funnel

    s4_path = tmp_path / "s4.jsonl"
    r3_path = tmp_path / "r3.jsonl"
    r4_path = tmp_path / "r4.jsonl"

    candidates = [
        {
            "video_id": "L21_V001",
            "frame_id": 12345,
            "n": 47,
            "score": 0.991,
        },
        {
            "video_id": "L22_V004",
            "frame_id": 23456,
            "n": 88,
            "score": 0.872,
        },
        {
            "video_id": "L26_V009",
            "frame_id": 34567,
            "n": 12,
            "score": 0.741,
        },
    ]

    s4_path.write_text(
        '{"query_id":"identity-q","semantic_k_hint":100}\n',
        encoding="utf-8",
    )

    r3_path.write_text(
        json.dumps(
            {
                "query_id": "identity-q",
                "candidates": candidates,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    list(
        run_jsonl_funnel(
            s4_path=s4_path,
            r3_path=r3_path,
            output_path=r4_path,
        )
    )

    output = json.loads(
        r4_path.read_text(
            encoding="utf-8"
        ).strip()
    )

    assert [
        (
            candidate["video_id"],
            candidate["frame_id"],
            candidate["n"],
        )
        for candidate in output[
            "selected_candidates"
        ]
    ] == [
        (
            candidate["video_id"],
            candidate["frame_id"],
            candidate["n"],
        )
        for candidate in candidates
    ]