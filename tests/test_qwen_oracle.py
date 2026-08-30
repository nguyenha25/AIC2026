from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.qwen_oracle as m


def test_chuan_hoa_exact_lowercase_punctuation_whitespace():
    assert m.chuan_hoa_exact("  Nước Nghệ Tươi, và Bột Canh!  ") == "nước nghệ tươi và bột canh"


def test_exact_match_accepts_case_and_punctuation_only():
    assert m.exact_match("Nước nghệ tươi và bột canh", "nước nghệ tươi, và bột canh!")


def test_exact_match_rejects_different_answer():
    assert not m.exact_match("Nước nghệ tươi và bột canh", "Nước mắm và đường")


def test_tao_prompt_keeps_question_and_short_vietnamese_instruction():
    question = "Có bao nhiêu người?"
    prompt = m.tao_prompt(question)
    assert question in prompt
    assert "Trả lời ngắn gọn bằng tiếng Việt" in prompt


def test_tao_prompt_rejects_unknown_mode():
    with pytest.raises(ValueError):
        m.tao_prompt("Câu hỏi?", "unknown")


def test_doc_qa_filters_only_hoi_dap(tmp_path: Path):
    path = tmp_path / "dev.jsonl"
    rows = [
        {"id": "01", "loai_truy_van": "hoi_dap", "cau_hoi": "Q1"},
        {"id": "02", "loai_truy_van": "kis", "cau_hoi": "Q2"},
        {"id": "03", "loai_truy_van": "hoi_dap", "cau_hoi": "Q3"},
    ]
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
        encoding="utf-8",
    )

    out = m.doc_qa(path)

    assert [x["id"] for x in out] == ["01", "03"]


def test_doc_qa_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        m.doc_qa(tmp_path / "missing.jsonl")


def test_chon_frame_oracle_uses_nearest_when_inside_gt_range():
    row = SimpleNamespace(frame_idx=105, n=7, fps=25.0, pts_time=4.2)

    class FakeFrameMap:
        def nearest_by_time(self, pts):
            return row

    q = {"frame_start": 100, "frame_end": 110, "pts_time": 4.2}

    assert m.chon_frame_oracle(q, FakeFrameMap()) is row


def test_chon_frame_oracle_falls_back_to_middle_of_gt_range():
    outside = SimpleNamespace(frame_idx=90, n=5, fps=25.0, pts_time=3.6)
    inside = SimpleNamespace(frame_idx=105, n=7, fps=25.0, pts_time=4.2)

    class FakeFrameMap:
        def __init__(self):
            self.calls = 0

        def nearest_by_time(self, pts):
            self.calls += 1
            return outside if self.calls == 1 else inside

    q = {"frame_start": 100, "frame_end": 110, "pts_time": 3.6}

    assert m.chon_frame_oracle(q, FakeFrameMap()) is inside


def test_chon_frame_oracle_returns_none_when_no_keyframe_in_range():
    outside1 = SimpleNamespace(frame_idx=90, n=5, fps=25.0, pts_time=3.6)
    outside2 = SimpleNamespace(frame_idx=120, n=8, fps=25.0, pts_time=4.8)

    class FakeFrameMap:
        def __init__(self):
            self.calls = 0

        def nearest_by_time(self, pts):
            self.calls += 1
            return outside1 if self.calls == 1 else outside2

    q = {"frame_start": 100, "frame_end": 110, "pts_time": 3.6}

    assert m.chon_frame_oracle(q, FakeFrameMap()) is None


def test_tim_anh_theo_n_finds_zero_padded_jpg(tmp_path: Path):
    video_dir = tmp_path / "Keyframes_L26_e" / "L26_V495"
    video_dir.mkdir(parents=True)
    image = video_dir / "097.jpg"
    image.write_bytes(b"fake")

    assert m.tim_anh_theo_n("L26_V495", 97, tmp_path) == image


def test_tong_hop_counts_ok_matches_latency_and_errors():
    records = [
        {
            "status": "ok",
            "intent": "dem",
            "exact_match": True,
            "semantic_match": True,
            "latency_ms": 100.0,
        },
        {
            "status": "ok",
            "intent": "khac",
            "exact_match": False,
            "semantic_match": True,
            "latency_ms": 300.0,
        },
        {"status": "missing", "reason": "keyframe_image"},
    ]

    summary = m.tong_hop(records)

    assert summary["total_records"] == 3
    assert summary["ok"] == 2
    assert summary["missing_or_error"] == 1
    assert summary["overall"]["exact_match"] == 1
    assert summary["overall"]["semantic_match"] == 2
    assert summary["overall"]["exact_rate"] == pytest.approx(0.5)
    assert summary["overall"]["semantic_rate"] == pytest.approx(1.0)
    assert summary["overall"]["avg_latency_ms"] == pytest.approx(200.0)
    assert summary["non_ok_reasons"] == {"keyframe_image": 1}
