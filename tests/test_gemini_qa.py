from __future__ import annotations

import json
from types import SimpleNamespace


def _row(video_id: str, n: int, frame_idx: int, answer: str = "local"):
    hit = SimpleNamespace(
        video_id=video_id,
        n=n,
        frame_idx=frame_idx,
        pts_time=float(n),
        score=1.0,
        source="test",
    )
    return {
        "hit": hit,
        "video_id": video_id,
        "frame_ids": [frame_idx],
        "answer": answer,
        "answer_confidence": 0.2,
        "answer_source": "local",
        "answer_explanation": "local",
        "other_answers": [],
    }


def test_select_candidate_indices_balances_videos_and_neighbours():
    from aic2026.gemini_qa import select_candidate_indices

    rows = []
    for video in ("V1", "V2", "V3", "V4", "V5"):
        rows.extend(_row(video, n, n) for n in range(1, 6))

    indices = select_candidate_indices(rows, max_images=12)
    videos = [rows[index]["video_id"] for index in indices]
    assert len(indices) == 12
    assert set(videos) == {"V1", "V2", "V3", "V4"}
    assert all(videos.count(video) == 3 for video in set(videos))


def test_assess_uses_one_request_then_cache(tmp_path):
    from aic2026.gemini_qa import GeminiQAReader

    rows = [_row("V1", 1, 101), _row("V2", 2, 202)]
    for n in (1, 2):
        (tmp_path / f"{n}.jpg").write_bytes(b"fake-jpeg-" + bytes([n]))

    calls = []

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        result = {
            "items": [
                {
                    "candidate_id": 0,
                    "relevance": 0.9,
                    "answer": "màu xanh",
                    "confidence": 0.8,
                    "evidence": "áo trong ảnh",
                },
                {
                    "candidate_id": 1,
                    "relevance": 0.2,
                    "answer": "khong ro",
                    "confidence": 0.1,
                    "evidence": "không đủ bằng chứng",
                },
            ]
        }
        return {
            "candidates": [{
                "content": {"parts": [{"text": json.dumps(result)}]}
            }]
        }

    reader = GeminiQAReader(
        api_key="test-key",
        cache_dir=tmp_path / "cache",
        max_images=2,
        retries=0,
        transport=transport,
        image_path_resolver=lambda video, n: tmp_path / f"{n}.jpg",
        evidence_provider=lambda video, n, pts: ("OCR", "ASR"),
    )
    first, first_meta = reader.assess("Áo màu gì?", rows)
    second, second_meta = reader.assess("Áo màu gì?", rows)

    assert len(calls) == 1
    assert first[0].answer == "màu xanh"
    assert second[0].answer == "màu xanh"
    assert first_meta["cache_hit"] is False
    assert second_meta["cache_hit"] is True
    assert "test-key" not in json.dumps(first_meta)
    assert all(
        "test-key" not in path.read_text(encoding="utf-8")
        for path in (tmp_path / "cache").glob("*.json")
    )


def test_apply_assessments_preserves_exact_frame_answer_pair():
    from aic2026.gemini_qa import GeminiAssessment, apply_gemini_assessments

    rows = [_row("V1", 1, 101), _row("V2", 2, 202)]
    assessments = {
        0: GeminiAssessment(0, 0, 0.2, "đỏ", 0.7, "frame 101"),
        1: GeminiAssessment(1, 1, 0.9, "xanh", 0.9, "frame 202"),
    }
    output = apply_gemini_assessments(rows, assessments, rerank=True)

    assert [(row["frame_ids"][0], row["answer"]) for row in output] == [
        (202, "xanh"),
        (101, "đỏ"),
    ]


def test_gemini_abstention_keeps_nonempty_local_answer():
    from aic2026.gemini_qa import GeminiAssessment, apply_gemini_assessments

    rows = [_row("V1", 1, 101, answer="13")]
    assessments = {
        0: GeminiAssessment(0, 0, 0.5, "khong ro", 0.1, "thiếu chữ")
    }
    output = apply_gemini_assessments(rows, assessments)
    assert output[0]["answer"] == "13"
    assert output[0]["answer_source"] == "local"


def test_schema_compatibility_retry_does_not_need_network_retry_budget():
    from aic2026.gemini_qa import GeminiHTTPError, GeminiQAReader

    payload = {
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {"type": "OBJECT"},
        }
    }
    seen = []

    def transport(url, headers, current, timeout):
        seen.append(current)
        if len(seen) == 1:
            raise GeminiHTTPError(400, "responseSchema is unsupported")
        return {"ok": True}

    reader = GeminiQAReader(api_key="x", retries=0, transport=transport)
    assert reader._call(payload) == {"ok": True}
    assert len(seen) == 2
    assert "responseSchema" not in seen[1]["generationConfig"]


def test_provider_falls_back_to_local_rows(monkeypatch):
    import aic2026.ui.pipelines as pipelines
    from aic2026.gemini_qa import GeminiQAError

    rows = [_row("V1", 1, 101, answer="local-safe")]
    monkeypatch.setattr(
        pipelines,
        "run_qa_answer_rows",
        lambda *args, **kwargs: {"rows": rows, "latency_ms": 1.0},
    )

    class BrokenReader:
        model = "gemini-test"

        def assess(self, question, input_rows):
            raise GeminiQAError("429 quota")

    result = pipelines.run_qa_with_provider(
        "câu hỏi", [], provider="gemini", cloud_reader=BrokenReader()
    )
    assert result["provider"] == "local_fallback"
    assert result["rows"][0]["answer"] == "local-safe"
    assert "429" in result["provider_meta"]["fallback_reason"]


def test_qa_v4_adapter_preserves_r4_candidate_identity(monkeypatch):
    import aic2026.qa_answer as qa
    from scripts.qa_v4_pipeline import run_reader_from_r4

    captured = {}

    def fake_reader(question, hits, **kwargs):
        captured["hits"] = hits
        captured["kwargs"] = kwargs
        return [(hit, qa.DapAn("x", 1.0, "test")) for hit in hits]

    monkeypatch.setattr(qa, "tra_loi_theo_hang", fake_reader)
    result = SimpleNamespace(selected_candidates=(
        {"video_id": "V1", "frame_id": 123, "n": 7, "score": 0.8,
         "pts_time": 4.2},
    ))
    output = run_reader_from_r4(
        cau_hoi="x", result=result, bo_doc_anh=None, dung_vlm=False
    )

    assert len(output) == 1
    hit = captured["hits"][0]
    assert (hit.video_id, hit.frame_idx, hit.n) == ("V1", 123, 7)
    assert captured["kwargs"]["mo_rong_lan_can"] is False
