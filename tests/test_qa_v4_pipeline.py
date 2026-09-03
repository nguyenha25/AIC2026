from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.qa_v4_pipeline as qv4
from scripts.adaptive_k import AdaptiveKResult


def _candidate(
    *,
    video_id: str,
    n: int,
    frame_idx: int,
    pts_time: float | None,
    score: float,
    include_frame_idx: bool = True,
) -> dict:
    candidate = {
        "video_id": video_id,
        "n": n,
        "frame_id": frame_idx,
        "score": score,
    }

    if include_frame_idx:
        candidate["frame_idx"] = frame_idx

    if pts_time is not None:
        candidate["pts_time"] = pts_time

    return candidate


def _result(
    *,
    semantic_k: int = 300,
    reader_k: int = 8,
    candidates: tuple[dict, ...] | None = None,
) -> AdaptiveKResult:
    if candidates is None:
        candidates = tuple(
            _candidate(
                video_id=f"L26_V{i:03d}",
                n=i,
                frame_idx=i * 100,
                pts_time=float(i * 4),
                score=1.0 / i,
            )
            for i in range(1, reader_k + 1)
        )

    return AdaptiveKResult(
        query_id="Q_TEST",
        k_requested=semantic_k,
        reader_k_requested=reader_k,
        k_effective=len(candidates),
        k_available=50,
        selected_candidates=candidates,
        status="ok",
        fallback_reason=None,
    )


def test_r4_to_reader_hits_preserves_order_and_fields():
    candidates = (
        _candidate(
            video_id="L26_V001",
            n=7,
            frame_idx=700,
            pts_time=28.0,
            score=0.9,
        ),
        _candidate(
            video_id="L26_V002",
            n=3,
            frame_idx=320,
            pts_time=12.8,
            score=0.8,
        ),
    )
    result = _result(reader_k=4, candidates=candidates)

    hits = qv4.r4_to_reader_hits(result)

    assert len(hits) == 2

    assert hits[0].video_id == "L26_V001"
    assert hits[0].n == 7
    assert hits[0].frame_idx == 700
    assert hits[0].pts_time == 28.0
    assert hits[0].score == pytest.approx(0.9)
    assert hits[0].source == "qa_r4"

    assert hits[1].video_id == "L26_V002"
    assert hits[1].n == 3
    assert hits[1].frame_idx == 320
    assert hits[1].pts_time == 12.8
    assert hits[1].score == pytest.approx(0.8)



def test_r4_to_reader_hits_accepts_real_r4_contract_without_frame_idx_pts_time():
    candidates = (
        _candidate(
            video_id="L26_V166",
            n=134,
            frame_idx=5210,
            pts_time=None,
            score=1.0,
            include_frame_idx=False,
        ),
        _candidate(
            video_id="L26_V305",
            n=97,
            frame_idx=3720,
            pts_time=None,
            score=0.9,
            include_frame_idx=False,
        ),
    )

    result = _result(
        reader_k=4,
        candidates=candidates,
    )

    hits = qv4.r4_to_reader_hits(result)

    assert [hit.frame_idx for hit in hits] == [5210, 3720]
    assert [hit.pts_time for hit in hits] == [0.0, 0.0]
    assert [hit.n for hit in hits] == [134, 97]

    qv4.validate_r4_reader_contract(result, hits)

def test_validate_contract_accepts_valid_r4_result():
    result = _result(reader_k=8)

    hits = qv4.r4_to_reader_hits(result)

    qv4.validate_r4_reader_contract(result, hits)


def test_validate_contract_rejects_more_hits_than_effective_k():
    result = _result(reader_k=4)
    hits = qv4.r4_to_reader_hits(result)

    hits.append(hits[-1])

    with pytest.raises(ValueError, match="k_effective"):
        qv4.validate_r4_reader_contract(result, hits)


def test_validate_contract_rejects_reader_budget_overflow():
    candidates = tuple(
        _candidate(
            video_id=f"L26_V{i:03d}",
            n=i,
            frame_idx=i * 100,
            pts_time=float(i),
            score=0.5,
        )
        for i in range(1, 6)
    )

    result = AdaptiveKResult(
        query_id="Q_TEST",
        k_requested=100,
        reader_k_requested=4,
        k_effective=5,
        k_available=50,
        selected_candidates=candidates,
        status="ok",
        fallback_reason=None,
    )

    hits = qv4.r4_to_reader_hits(result)

    with pytest.raises(ValueError, match="budget QA-R4"):
        qv4.validate_r4_reader_contract(result, hits)


def test_validate_contract_rejects_changed_order():
    result = _result(reader_k=4)

    hits = qv4.r4_to_reader_hits(result)
    hits[0], hits[1] = hits[1], hits[0]

    with pytest.raises(ValueError):
        qv4.validate_r4_reader_contract(result, hits)


def test_run_reader_from_r4_disables_neighbor_expansion(monkeypatch):
    result = _result(reader_k=4)

    captured = {}

    def fake_tra_loi_theo_hang(
        cau_hoi,
        hits,
        so_dong=100,
        so_hang_vlm=5,
        bo_doc_anh=None,
        dung_vlm=True,
        mo_rong_lan_can=True,
    ):
        captured["cau_hoi"] = cau_hoi
        captured["hits"] = list(hits)
        captured["so_dong"] = so_dong
        captured["so_hang_vlm"] = so_hang_vlm
        captured["bo_doc_anh"] = bo_doc_anh
        captured["dung_vlm"] = dung_vlm
        captured["mo_rong_lan_can"] = mo_rong_lan_can

        return [
            (
                h,
                SimpleNamespace(
                    van_ban=f"answer-{i}",
                    do_tin=0.5,
                    nguon="vlm",
                    video_id=h.video_id,
                    frame_idx=h.frame_idx,
                ),
            )
            for i, h in enumerate(hits)
        ]

    monkeypatch.setattr(
        qv4,
        "tra_loi_theo_hang",
        fake_tra_loi_theo_hang,
    )

    outputs = qv4.run_reader_from_r4(
        cau_hoi="Có bao nhiêu người?",
        result=result,
        bo_doc_anh="FAKE_READER",
        dung_vlm=True,
    )

    assert len(outputs) == result.k_effective
    assert captured["so_dong"] == result.k_effective
    assert captured["so_hang_vlm"] == result.k_effective
    assert captured["mo_rong_lan_can"] is False
    assert captured["dung_vlm"] is True
    assert captured["bo_doc_anh"] == "FAKE_READER"

    expected_order = [
        c["frame_idx"]
        for c in result.selected_candidates
    ]
    actual_order = [
        h.frame_idx
        for h in captured["hits"]
    ]

    assert actual_order == expected_order


def test_run_reader_from_r4_uses_exact_r4_candidate_count(monkeypatch):
    candidates = tuple(
        _candidate(
            video_id=f"L28_V{i:03d}",
            n=i,
            frame_idx=1000 + i,
            pts_time=40.0 + i,
            score=0.7,
        )
        for i in range(1, 4)
    )

    result = _result(
        semantic_k=300,
        reader_k=8,
        candidates=candidates,
    )

    captured = {}

    def fake_tra_loi_theo_hang(
        cau_hoi,
        hits,
        so_dong=100,
        so_hang_vlm=5,
        bo_doc_anh=None,
        dung_vlm=True,
        mo_rong_lan_can=True,
    ):
        captured["n_hits"] = len(hits)
        captured["so_dong"] = so_dong
        captured["so_hang_vlm"] = so_hang_vlm

        return []

    monkeypatch.setattr(
        qv4,
        "tra_loi_theo_hang",
        fake_tra_loi_theo_hang,
    )

    qv4.run_reader_from_r4(
        cau_hoi="Câu hỏi test",
        result=result,
        dung_vlm=False,
    )

    assert captured["n_hits"] == 3
    assert captured["so_dong"] == 3
    assert captured["so_hang_vlm"] == 3


def test_run_qa_v4_passes_all_reader_answers_to_consensus(monkeypatch):
    result = _result(reader_k=4)

    fake_answers = [
        SimpleNamespace(
            van_ban="2",
            do_tin=0.9,
            nguon="vlm",
            video_id="L26_V001",
            frame_idx=100,
        ),
        SimpleNamespace(
            van_ban="hai",
            do_tin=0.8,
            nguon="vlm",
            video_id="L26_V002",
            frame_idx=200,
        ),
    ]

    fake_reader_outputs = [
        ("hit-1", fake_answers[0]),
        ("hit-2", fake_answers[1]),
    ]

    monkeypatch.setattr(
        qv4,
        "run_reader_from_r4",
        lambda **kwargs: fake_reader_outputs,
    )

    captured = {}

    fake_consensus = SimpleNamespace(
        final_answer="2",
        confidence=0.9,
    )

    def fake_choose_consensus(
        question,
        candidates,
        question_type=None,
    ):
        captured["question"] = question
        captured["candidates"] = list(candidates)
        captured["question_type"] = question_type
        return fake_consensus

    monkeypatch.setattr(
        qv4,
        "choose_consensus",
        fake_choose_consensus,
    )

    result_consensus = qv4.run_qa_v4(
        cau_hoi="Có bao nhiêu người?",
        result=result,
        dung_vlm=True,
    )

    assert result_consensus is fake_consensus
    assert captured["question"] == "Có bao nhiêu người?"
    assert captured["candidates"] == fake_answers


def test_run_qa_v4_final_answer_non_empty(monkeypatch):
    result = _result(reader_k=4)

    monkeypatch.setattr(
        qv4,
        "run_reader_from_r4",
        lambda **kwargs: [
            (
                "hit",
                SimpleNamespace(
                    van_ban="6",
                    do_tin=0.9,
                    nguon="vlm",
                    video_id="L21_V010",
                    frame_idx=19500,
                ),
            )
        ],
    )

    monkeypatch.setattr(
        qv4,
        "choose_consensus",
        lambda **kwargs: SimpleNamespace(
            final_answer="6",
            confidence=0.9,
        ),
    )

    consensus = qv4.run_qa_v4(
        cau_hoi="Có bao nhiêu người?",
        result=result,
    )

    assert isinstance(consensus.final_answer, str)
    assert consensus.final_answer.strip() != ""
