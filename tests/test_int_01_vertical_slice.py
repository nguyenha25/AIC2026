from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.int_01_vertical_slice import (
    adapt_reader_output,
    audit_candidate,
    audit_qa_r1_handoff,
    find_qa_r1_record,
    normalize_answer,
    review_vertical_slice,
)


def make_query() -> dict:
    return {
        "id": "12",
        "loai_truy_van": "hoi_dap",
        "cau_hoi": "Địa điểm trong biển hiệu tên gì?",
    }


def make_candidate() -> dict:
    return {
        "schema_version": "1.1",
        "query_id": "12",
        "event_id": None,
        "video_id": "L23_V024",
        "n": 57,
        "frame_idx": 6695,
        "pts_time": 223.17,
        "stage": "reader",
        "source_hits": ["clip_l", "ocr"],
        "source_ranks": {"clip_l": 18, "ocr": 3},
        "scores": {"clip_l": 0.31, "ocr": 0.84},
        "score_fused": 0.72,
        "semantic_coverage": 0.78,
        "window": {"start": 220.0, "end": 226.0},
        "rank_final": 1,
        "status": "ok",
        "error": None,
    }


def make_profile(candidate: dict | None = None, linked: bool = True) -> dict:
    record = {
        "query_id": "12",
        "reader_candidates": [candidate or make_candidate()],
    }
    if linked:
        record["query_plan_schema_version"] = "1.1"
        record["routing_weights_applied"] = {
            "clip_l": 0.2,
            "ocr": 0.7,
            "asr": 0.1,
            "caption": 0.0,
        }
    return {
        "schema_version": "1.1",
        "task": "QA-R1",
        "rrf": {"weight_clip_l": 0.2, "weight_ocr": 0.7},
        "query_records": [record],
    }


class TestInt01Contracts(unittest.TestCase):
    def test_normalize_answer_preserves_vietnamese_and_numbers(self):
        self.assertEqual(normalize_answer(" XÃ Giang Ly, số 2! "), "xã giang ly số 2")

    def test_find_qa_r1_record_keeps_query_id_alignment(self):
        record = find_qa_r1_record(make_profile(), "12")
        self.assertEqual(record["query_id"], "12")

    def test_find_qa_r1_record_rejects_missing_query(self):
        with self.assertRaises(KeyError):
            find_qa_r1_record(make_profile(), "99")

    def test_candidate_requires_integer_frame_and_provenance(self):
        candidate = make_candidate()
        candidate["frame_idx"] = "6695"
        candidate["source_hits"] = []
        candidate["source_ranks"] = {}

        codes = {issue.code for issue in audit_candidate(candidate, "12")}

        self.assertIn("QA_R1_INTEGER_FIELD", codes)
        self.assertIn("QA_R1_PROVENANCE_EMPTY", codes)

    def test_handoff_flags_profile_that_only_joins_by_query_id(self):
        plan = {
            "query_id": "12",
            "preferred_modalities": {"clip_l": 0.2, "ocr": 0.7, "asr": 0.1},
        }
        profile = make_profile(linked=False)
        profile["rrf"] = {"weight_clip_b": 1.0, "weight_clip_l": 0.2}
        record = profile["query_records"][0]

        codes = {issue.code for issue in audit_qa_r1_handoff(plan, profile, record)}

        self.assertIn("QA_R1_QUERY_PLAN_NOT_CONSUMED", codes)
        self.assertIn("QA_R1_ROUTING_NOT_APPLIED", codes)

    def test_handoff_downgrades_explicitly_unavailable_modality_to_p1(self):
        plan = {
            "query_id": "12",
            "preferred_modalities": {"clip_l": 0.2, "ocr": 0.7, "asr": 0.1},
        }
        profile = make_profile(linked=True)
        profile["rrf"] = {"weight_clip_b": 1.0, "weight_clip_l": 0.2}
        record = profile["query_records"][0]
        record["unavailable_modalities"] = ["ocr", "asr"]

        issues = audit_qa_r1_handoff(plan, profile, record)

        by_code = {issue.code: issue for issue in issues}
        self.assertIn("QA_R1_MODALITY_UNAVAILABLE", by_code)
        self.assertEqual(by_code["QA_R1_MODALITY_UNAVAILABLE"].severity, "P1")
        self.assertNotIn("QA_R1_ROUTING_NOT_APPLIED", by_code)

    def test_raw_qwen_text_becomes_safe_fallback_not_fake_confidence(self):
        answer, issues = adapt_reader_output(
            raw_output="Xã Giang Ly",
            plan={"query_id": "12", "answer_type": "short_text"},
            candidate=make_candidate(),
            latency_ms=12.5,
            model_id="fake-qwen",
        )

        self.assertEqual(answer["answer"], "khong ro")
        self.assertEqual(answer["answer_raw"], "Xã Giang Ly")
        self.assertEqual(answer["confidence"], 0.0)
        self.assertEqual(answer["status"], "fallback")
        self.assertTrue(answer["fallback_used"])
        self.assertIn("QA_V2_NO_CONFIDENCE", {issue.code for issue in issues})

    def test_structured_reader_output_creates_evidence_answer(self):
        answer, issues = adapt_reader_output(
            raw_output={
                "answer": "Xã Giang Ly",
                "confidence": 0.87,
                "confidence_method": "test_probability_v1",
            },
            plan={"query_id": "12", "answer_type": "short_text"},
            candidate=make_candidate(),
            latency_ms=12.5,
            model_id="reader-v3",
        )

        self.assertFalse(issues)
        self.assertEqual(answer["answer_normalized"], "xã giang ly")
        self.assertEqual(answer["video_id"], "L23_V024")
        self.assertEqual(answer["frame_idx"], 6695)
        self.assertEqual(answer["status"], "ok")
        self.assertFalse(answer["fallback_used"])

    def test_vertical_slice_preserves_candidate_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "057.jpg"
            image.write_bytes(b"fake-image")

            report = review_vertical_slice(
                query=make_query(),
                profile=make_profile(),
                image_resolver=lambda video_id, n: image,
                reader=lambda image_path, question: {
                    "answer": "Xã Giang Ly",
                    "confidence": 0.87,
                    "confidence_method": "test_probability_v1",
                },
                model_id="reader-v3",
            )

        self.assertTrue(report["reader_executed"])
        self.assertEqual(report["evidence_answer"]["video_id"], "L23_V024")
        self.assertEqual(
            report["evidence_answer"]["evidence"][1]["value"]["source_ranks"],
            {"clip_l": 18, "ocr": 3},
        )
        self.assertTrue(report["acceptance"]["candidate_provenance_present"])
        self.assertEqual(report["status"], "pass")


if __name__ == "__main__":
    unittest.main()
