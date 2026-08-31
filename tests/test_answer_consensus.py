import math
import inspect
import pytest

from scripts.answer_consensus import (
    Candidate,
    canonicalize_answer,
    choose_consensus,
    consensus_record,
    group_candidates,
    validate_confidence,
)


def c(answer, confidence=0.5, source="vlm"):
    return Candidate(answer, confidence, source)


def test_normalize_case_punctuation_and_spaces():
    assert canonicalize_answer(
        "  Đèo   Tà Pứa!!! ",
        question="Địa điểm này là đâu?",
        question_type="dia_diem",
    ) == "đèo tà pứa"


def test_normalization_keeps_vietnamese_accents():
    a = canonicalize_answer("má", question_type="khac")
    b = canonicalize_answer("ma", question_type="khac")
    assert a != b


@pytest.mark.parametrize(
    "answer",
    ["2", "2 người", "hai", "two"],
)
def test_count_answers_canonicalize_to_same_number(answer):
    assert canonicalize_answer(
        answer,
        question="Có bao nhiêu người?",
        question_type="dem",
    ) == "2"


def test_count_equivalents_are_grouped_together():
    groups = group_candidates(
        "Có bao nhiêu người?",
        [
            c("2", 0.6, "vlm"),
            c("hai", 0.5, "asr"),
            c("2 người", 0.7, "ocr"),
            c("3", 0.9, "vlm"),
        ],
        question_type="dem",
    )
    assert groups[0].canonical_answer == "2"
    assert groups[0].support == 3


def test_majority_support_wins_before_confidence():
    result = choose_consensus(
        "Tên địa điểm là gì?",
        [
            c("Tà Pứa", 0.55, "ocr"),
            c("Tà Pứa", 0.50, "asr"),
            c("Mộc Châu", 0.99, "vlm"),
        ],
        question_type="dia_diem",
    )
    assert result.canonical_answer == "tà pứa"
    assert result.support == 2


def test_confidence_breaks_support_tie():
    result = choose_consensus(
        "Đây là gì?",
        [
            c("A", 0.4, "asr"),
            c("B", 0.8, "asr"),
        ],
        question_type="khac",
    )
    assert result.canonical_answer == "b"


def test_source_priority_breaks_full_quality_tie_for_count():
    result = choose_consensus(
        "Có bao nhiêu người?",
        [
            c("2", 0.6, "vlm"),
            c("3", 0.6, "ocr"),
        ],
        question_type="dem",
    )
    assert result.canonical_answer == "2"


def test_source_priority_breaks_full_quality_tie_for_text():
    result = choose_consensus(
        "Dòng chữ trên hình là gì?",
        [
            c("Alpha", 0.6, "vlm"),
            c("Beta", 0.6, "ocr"),
        ],
        question_type="chu_tren_hinh",
    )
    assert result.canonical_answer == "beta"


def test_input_order_does_not_change_result():
    items = [
        c("Beta", 0.6, "ocr"),
        c("Alpha", 0.6, "vlm"),
        c("Gamma", 0.3, "asr"),
    ]
    r1 = choose_consensus(
        "Dòng chữ trên hình là gì?",
        items,
        question_type="chu_tren_hinh",
    )
    r2 = choose_consensus(
        "Dòng chữ trên hình là gì?",
        list(reversed(items)),
        question_type="chu_tren_hinh",
    )
    assert r1.final_answer == r2.final_answer
    assert r1.canonical_answer == r2.canonical_answer


def test_empty_candidate_does_not_become_winner():
    result = choose_consensus(
        "Đây là gì?",
        [
            c("", 1.0, "ocr"),
            c("xe đạp", 0.4, "vlm"),
        ],
        question_type="khac",
    )
    assert result.final_answer == "xe đạp"


def test_fallback_does_not_outvote_real_answer():
    result = choose_consensus(
        "Đây là gì?",
        [
            c("khong ro", 0.0, "du_phong"),
            c("khong ro", 0.0, "du_phong"),
            c("xe đạp", 0.2, "vlm"),
        ],
        question_type="khac",
    )
    assert result.final_answer == "xe đạp"
    assert result.used_fallback is False


def test_all_empty_uses_nonempty_fallback():
    result = choose_consensus(
        "Đây là gì?",
        [c("", 0.0, "ocr"), c("", 0.0, "asr")],
        question_type="khac",
    )
    assert result.final_answer.strip()
    assert result.used_fallback is True
    assert result.confidence == 0.0


def test_no_candidates_uses_nonempty_fallback():
    result = choose_consensus(
        "Có bao nhiêu người?",
        [],
        question_type="dem",
    )
    assert result.final_answer.strip()
    assert result.used_fallback is True


@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, math.nan, math.inf, -math.inf, "0.5", True, None],
)
def test_invalid_confidence_is_rejected(value):
    with pytest.raises(ValueError, match="confidence"):
        validate_confidence(value)


@pytest.mark.parametrize("value", [0, 0.0, 0.5, 1, 1.0])
def test_valid_confidence_is_accepted(value):
    assert validate_confidence(value) == float(value)


def test_dict_candidate_is_supported():
    result = choose_consensus(
        "Đây là gì?",
        [{"van_ban": "xe đạp", "do_tin": 0.7, "nguon": "vlm"}],
        question_type="khac",
    )
    assert result.final_answer == "xe đạp"


class FakeDapAn:
    def __init__(self):
        self.van_ban = "Jacquemus"
        self.do_tin = 0.8
        self.nguon = "ocr"
        self.frame_idx = 10
        self.video_id = "L00_V001"


def test_dapan_like_object_is_supported():
    result = choose_consensus(
        "Tên thương hiệu là gì?",
        [FakeDapAn()],
        question_type="chu_tren_hinh",
    )
    assert result.final_answer == "Jacquemus"


def test_result_contains_diagnostics():
    result = choose_consensus(
        "Có bao nhiêu người?",
        [
            c("2", 0.6, "vlm"),
            c("hai", 0.5, "asr"),
            c("3", 0.4, "ocr"),
        ],
        question_type="dem",
    )
    assert result.support == 2
    assert "vlm" in result.sources
    assert len(result.alternatives) == 1
    assert result.alternatives[0]["canonical_answer"] == "3"


def test_support_bonus_is_capped_at_one():
    result = choose_consensus(
        "Đây là gì?",
        [c("xe", 0.95, "asr"), c("xe", 0.95, "ocr")],
        question_type="khac",
    )
    assert result.confidence == 1.0


def test_consensus_record_has_required_fields_and_no_gt():
    row = consensus_record(
        query_id="23",
        video_id="L26_V001",
        question="Có bao nhiêu người?",
        candidates=[c("2", 0.7, "vlm")],
    )
    assert row["task"] == "QA-V4"
    assert row["query_id"] == "23"
    assert row["video_id"] == "L26_V001"
    assert row["final_answer"] == "2"
    assert row["candidate_count"] == 1
    assert "gt_answer" not in row


def test_choose_consensus_api_has_no_gt_answer_parameter():
    params = inspect.signature(choose_consensus).parameters
    assert "gt_answer" not in params
    assert "ground_truth" not in params
    assert "gt" not in params


def test_consensus_record_api_has_no_gt_answer_parameter():
    params = inspect.signature(consensus_record).parameters
    assert "gt_answer" not in params
    assert "ground_truth" not in params
    assert "gt" not in params
