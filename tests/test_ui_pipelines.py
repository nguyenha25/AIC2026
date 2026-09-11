from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from aic2026.submit import KIS, QA, TRAKE
from aic2026.ui.pipelines import (
    dense_frame_path,
    ensure_repo_root_importable,
    make_trake_inference_row,
    parse_trake_event_lines,
    submission_csv_bytes,
)


def _rows(payload: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(payload.decode("utf-8"))))


def test_parse_trake_event_lines_removes_common_prefixes():
    assert parse_trake_event_lines("(1) đặt nồi\nE2: mở lửa\n- 3. cho dầu") == [
        "đặt nồi",
        "mở lửa",
        "cho dầu",
    ]


def test_parse_trake_event_lines_rejects_single_event():
    with pytest.raises(ValueError, match="ít nhất 2"):
        parse_trake_event_lines("chỉ có một sự kiện")


def test_make_trake_inference_row_contains_no_real_gt():
    row = make_trake_inference_row("32", ["đặt nồi", "mở lửa"])
    assert row["id"] == "32"
    assert [stage["su_kien"] for stage in row["cac_giai_doan"]] == [
        "đặt nồi",
        "mở lửa",
    ]
    assert all(stage["frame_start"] == 0 for stage in row["cac_giai_doan"])


def test_dense_frame_path_uses_six_digit_frame_id():
    assert dense_frame_path(Path("D:/aic-data"), "L26_V091", 4486) == Path(
        "D:/aic-data/derived/frames_dense/L26_V091/004486.jpg"
    )


def test_streamlit_adapter_makes_top_level_scripts_importable():
    repo_root = ensure_repo_root_importable()
    assert (repo_root / "scripts" / "run_trake_e2.py").is_file()
    __import__("scripts")


def test_submission_csv_formats_all_three_tasks_without_header():
    kis = submission_csv_bytes(
        KIS,
        [{"video_id": "L01_V001", "frame_idx": 12}],
    )
    qa = submission_csv_bytes(
        QA,
        [{"video_id": "L01_V001", "frame_ids": [12], "answer": "xanh"}],
    )
    trake = submission_csv_bytes(
        TRAKE,
        [{"video_id": "L01_V001", "frame_idx": [12, 25, 40]}],
    )

    assert _rows(kis) == [["L01_V001", "12"]]
    assert _rows(qa) == [["L01_V001", "12", "xanh"]]
    assert _rows(trake) == [["L01_V001", "12", "25", "40"]]


def test_submission_csv_deduplicates_and_rejects_empty_qa_answer():
    duplicate = {"video_id": "L01_V001", "frame_idx": 12}
    assert _rows(submission_csv_bytes(KIS, [duplicate, duplicate])) == [
        ["L01_V001", "12"]
    ]

    with pytest.raises(ValueError, match="rỗng"):
        submission_csv_bytes(
            QA,
            [{"video_id": "L01_V001", "frame_idx": 12, "answer": ""}],
        )
