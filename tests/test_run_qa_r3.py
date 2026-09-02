from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.fusion_engine import SelectionMethod
from scripts.run_qa_r3 import QAInputError, run_qa_r3


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    r1_path = tmp_path / "dev_qa_r1_profile.json"
    n01_path = tmp_path / "n01_semantic_parsing.jsonl"

    candidates = [
        {
            "query_id": "N01",
            "video_id": "video_a",
            "n": 7,
            "frame_idx": 101,
            "pts_time": 4.2,
            "evidence_score": 0.7,
            "score_fused": 0.2,
            "status": "ok",
        },
        {
            "query_id": "N01",
            "video_id": "video_b",
            "n": 8,
            "frame_idx": 202,
            "pts_time": 8.4,
            "evidence_score": 0.9,
            "score_fused": 0.3,
            "status": "ok",
        },
        {
            "query_id": "N01",
            "video_id": "video_c",
            "n": 9,
            "frame_idx": 303,
            "pts_time": 12.6,
            "evidence_score": 0.8,
            "score_fused": 0.1,
            "status": "ok",
        },
    ]

    r1_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "query_records": [
                    {
                        "query_id": "N01",
                        "query_type": "hoi_dap",
                        "reader_candidates": candidates,
                    },
                    {
                        "query_id": "TR01",
                        "query_type": "chuoi_su_kien",
                        "reader_candidates": candidates,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    n01_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": "1.1",
                        "query_id": "N01",
                        "task": "qa",
                        "entities": ["người"],
                        "actions": ["chạy"],
                        "attributes": [],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "schema_version": "1.1",
                        "query_id": "TR01",
                        "task": "trake",
                        "events": [],
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return r1_path, n01_path


def test_single_query_writes_exact_r4_object(tmp_path: Path) -> None:
    r1_path, n01_path = _write_inputs(tmp_path)
    output_path = tmp_path / "qa_r3_candidates.json"

    document, summary = run_qa_r3(
        r1_path=r1_path,
        n01_path=n01_path,
        output_path=output_path,
        query_id="N01",
        top_k=2,
    )

    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8")) == document
    assert set(document) == {"query_id", "candidates"}
    assert document["query_id"] == "N01"
    assert len(document["candidates"]) == 2

    assert all(
        set(candidate) == {"video_id", "frame_id", "n", "score"}
        for candidate in document["candidates"]
    )

    assert document["candidates"][0]["video_id"] == "video_b"
    assert document["candidates"][0]["frame_id"] == 202
    assert document["candidates"][0]["n"] == 8

    assert summary["pool_counts"] == {"reader_candidates": 1}


def test_batch_mode_writes_array_of_r4_objects(tmp_path: Path) -> None:
    r1_path, n01_path = _write_inputs(tmp_path)
    output_path = tmp_path / "qa_r3_candidates.json"

    document, summary = run_qa_r3(
        r1_path=r1_path,
        n01_path=n01_path,
        output_path=output_path,
        top_k=12,
        selection_method=SelectionMethod.MMR,
    )

    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8")) == document
    assert isinstance(document, list)
    assert len(document) == 1

    assert document[0]["query_id"] == "N01"
    assert set(document[0]) == {"query_id", "candidates"}

    assert all(
        set(candidate) == {"video_id", "frame_id", "n", "score"}
        for candidate in document[0]["candidates"]
    )

    assert summary["pool_counts"] == {"reader_candidates": 1}


def test_prefers_real_coverage_pool_when_available(tmp_path: Path) -> None:
    r1_path, n01_path = _write_inputs(tmp_path)

    data = json.loads(r1_path.read_text(encoding="utf-8"))
    data["query_records"][0]["coverage_candidates"] = [
        {
            "query_id": "N01",
            "video_id": "coverage_video",
            "n": 42,
            "frame_idx": 4242,
            "pts_time": 42.0,
            "score_fused": 0.99,
            "status": "ok",
        }
    ]
    r1_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    output_path = tmp_path / "qa_r3_candidates.json"

    document, summary = run_qa_r3(
        r1_path=r1_path,
        n01_path=n01_path,
        output_path=output_path,
        query_id="N01",
        top_k=1,
    )

    assert document["query_id"] == "N01"
    assert len(document["candidates"]) == 1

    candidate = document["candidates"][0]
    assert set(candidate) == {"video_id", "frame_id", "n", "score"}
    assert candidate["video_id"] == "coverage_video"
    assert candidate["frame_id"] == 4242
    assert candidate["n"] == 42

    assert summary["pool_counts"] == {"coverage_candidates": 1}


def test_missing_requested_query_is_rejected(tmp_path: Path) -> None:
    r1_path, n01_path = _write_inputs(tmp_path)
    output_path = tmp_path / "qa_r3_candidates.json"

    with pytest.raises(QAInputError):
        run_qa_r3(
            r1_path=r1_path,
            n01_path=n01_path,
            output_path=output_path,
            query_id="MISSING",
            top_k=2,
        )


def test_output_comes_from_upstream_ids_not_mock_pool(tmp_path: Path) -> None:
    r1_path, n01_path = _write_inputs(tmp_path)
    output_path = tmp_path / "qa_r3_candidates.json"

    document, _ = run_qa_r3(
        r1_path=r1_path,
        n01_path=n01_path,
        output_path=output_path,
        query_id="N01",
        top_k=2,
    )

    upstream_pairs = {
        (candidate["video_id"], candidate["frame_idx"], candidate["n"])
        for candidate in json.loads(r1_path.read_text(encoding="utf-8"))[
            "query_records"
        ][0]["reader_candidates"]
    }

    for candidate in document["candidates"]:
        assert (
            candidate["video_id"],
            candidate["frame_id"],
            candidate["n"],
        ) in upstream_pairs