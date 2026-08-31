"""
QA-V4 — Answer normalization + consensus/fusion.

Mục tiêu:
- Nhận nhiều đáp án ứng viên từ OCR / ASR / VLM / dự phòng.
- Chuẩn hóa theo loại câu hỏi.
- Gom các đáp án tương đương.
- Chọn đáp án cuối theo support -> confidence -> source priority.
- Không dùng ground-truth để quyết định winner.
- Luôn trả final_answer khác rỗng.

Module này không chạy retrieval và không chạy model.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from src.aic2026.qa_answer import (
    CHU_TREN_HINH,
    DEM,
    DIA_DIEM,
    DAP_AN_DU_PHONG,
    KHAC,
    MAU,
    THOI_GIAN,
    chuan_hoa_so,
    doan_du_phong,
    loai_cau_hoi,
)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

SOURCE_PRIORITY = {
    DEM: ("vlm", "ocr", "asr", "du_phong"),
    MAU: ("vlm", "ocr", "asr", "du_phong"),
    CHU_TREN_HINH: ("ocr", "asr", "vlm", "du_phong"),
    THOI_GIAN: ("ocr", "asr", "vlm", "du_phong"),
    DIA_DIEM: ("ocr", "asr", "vlm", "du_phong"),
    KHAC: ("asr", "ocr", "vlm", "du_phong"),
}

FALLBACK_SOURCE = "du_phong"
CONFIDENCE_TIE_EPSILON = 0.05


@dataclass(frozen=True)
class Candidate:
    answer: str
    confidence: float
    source: str
    frame_idx: int = -1
    video_id: str = ""


@dataclass(frozen=True)
class GroupSummary:
    canonical_answer: str
    representative_answer: str
    support: int
    sources: tuple[str, ...]
    mean_confidence: float
    max_confidence: float
    best_source_rank: int
    member_answers: tuple[str, ...]


@dataclass(frozen=True)
class ConsensusResult:
    final_answer: str
    canonical_answer: str
    confidence: float
    support: int
    sources: tuple[str, ...]
    alternatives: tuple[dict[str, Any], ...]
    used_fallback: bool
    question_type: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sources"] = list(self.sources)
        d["alternatives"] = list(self.alternatives)
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", (text or "").lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.replace("đ", "d")


def _basic_normalize(text: str) -> str:
    """
    NFC + lowercase + bỏ punctuation ngoại trừ ký tự chữ/số + gộp khoảng trắng.
    Giữ dấu tiếng Việt để tránh gộp nhầm nghĩa.
    """
    s = unicodedata.normalize("NFC", str(text or "")).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _valid_confidence(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    x = float(value)
    return math.isfinite(x) and 0.0 <= x <= 1.0


def validate_confidence(value: Any) -> float:
    if not _valid_confidence(value):
        raise ValueError(
            f"confidence phải là số hữu hạn trong [0, 1], nhận {value!r}"
        )
    return float(value)


def _source_rank(question_type: str, source: str) -> int:
    order = SOURCE_PRIORITY.get(question_type, SOURCE_PRIORITY[KHAC])
    try:
        return order.index(source)
    except ValueError:
        return len(order)


def _extract_candidate(item: Any) -> Candidate:
    """
    Hỗ trợ:
    - Candidate
    - aic2026.qa_answer.DapAn-like object
    - dict có answer/van_ban, confidence/do_tin, source/nguon
    """
    if isinstance(item, Candidate):
        return item

    if isinstance(item, dict):
        answer = item.get("answer", item.get("van_ban", ""))
        confidence = item.get("confidence", item.get("do_tin", 0.0))
        source = item.get("source", item.get("nguon", ""))
        frame_idx = item.get("frame_idx", -1)
        video_id = item.get("video_id", "")
    else:
        answer = getattr(item, "answer", getattr(item, "van_ban", ""))
        confidence = getattr(
            item, "confidence", getattr(item, "do_tin", 0.0)
        )
        source = getattr(item, "source", getattr(item, "nguon", ""))
        frame_idx = getattr(item, "frame_idx", -1)
        video_id = getattr(item, "video_id", "")

    return Candidate(
        answer=str(answer or "").strip(),
        confidence=validate_confidence(confidence),
        source=str(source or "").strip().lower(),
        frame_idx=int(frame_idx),
        video_id=str(video_id or ""),
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_COUNT_NOUNS = {
    "người", "nguoi", "cái", "cai", "chiếc", "chiec", "con",
    "bạn", "ban", "xe", "vật", "vat", "quả", "qua",
}


def canonicalize_answer(answer: str, question: str = "", question_type: str | None = None) -> str:
    """
    Canonical form để grouping, KHÔNG dùng GT.

    Với câu DEM:
      "two", "hai", "2 người" -> "2"

    Với loại khác:
      lowercase + punctuation/space normalization, vẫn giữ dấu tiếng Việt.
    """
    qtype = question_type or loai_cau_hoi(question)
    text = str(answer or "").strip()
    if not text:
        return ""

    if qtype == DEM:
        normalized_number = str(chuan_hoa_so(text)).strip()
        # chuan_hoa_so của repo trả "2" nếu tìm thấy số/chữ số.
        if re.fullmatch(r"\d+", normalized_number):
            return normalized_number

        # Bảo hiểm cho dạng "2 người" nếu implementation upstream thay đổi.
        m = re.search(r"\b\d+\b", text)
        if m:
            return m.group(0)

    return _basic_normalize(text)


def _is_fallback(candidate: Candidate) -> bool:
    if candidate.source == FALLBACK_SOURCE:
        return True
    return _basic_normalize(candidate.answer) == _basic_normalize(DAP_AN_DU_PHONG)


def _is_real_answer(candidate: Candidate) -> bool:
    return bool(candidate.answer.strip()) and not _is_fallback(candidate)


# ---------------------------------------------------------------------------
# Grouping + scoring
# ---------------------------------------------------------------------------

def group_candidates(
    question: str,
    candidates: Sequence[Any],
    *,
    question_type: str | None = None,
) -> list[GroupSummary]:
    qtype = question_type or loai_cau_hoi(question)
    prepared = [_extract_candidate(x) for x in candidates]

    # Nếu tồn tại đáp án thật, fallback không được tham gia vote.
    real = [c for c in prepared if _is_real_answer(c)]
    usable = real if real else [c for c in prepared if c.answer.strip()]

    buckets: dict[str, list[Candidate]] = defaultdict(list)
    for c in usable:
        canonical = canonicalize_answer(c.answer, question, qtype)
        if canonical:
            buckets[canonical].append(c)

    groups: list[GroupSummary] = []
    for canonical, members in buckets.items():
        # Representative: confidence cao hơn -> source ưu tiên hơn ->
        # answer lexicographic để deterministic.
        representative = sorted(
            members,
            key=lambda c: (
                -c.confidence,
                _source_rank(qtype, c.source),
                _basic_normalize(c.answer),
                c.answer,
            ),
        )[0]

        confidences = [c.confidence for c in members]
        unique_sources = tuple(sorted({c.source for c in members}))

        groups.append(
            GroupSummary(
                canonical_answer=canonical,
                representative_answer=representative.answer,
                support=len(members),
                sources=unique_sources,
                mean_confidence=sum(confidences) / len(confidences),
                max_confidence=max(confidences),
                best_source_rank=min(
                    _source_rank(qtype, c.source) for c in members
                ),
                member_answers=tuple(sorted(c.answer for c in members)),
            )
        )

    # Winner order:
    # 1) support cao hơn
    # 2) trong cùng support, chỉ coi confidence là thắng rõ nếu chênh > epsilon
    # 3) nếu confidence nằm trong epsilon của nhóm tốt nhất -> source priority
    # 4) max confidence
    # 5) canonical lexical tie-break => deterministic
    ranked: list[GroupSummary] = []
    remaining = list(groups)

    while remaining:
        max_support = max(g.support for g in remaining)
        same_support = [g for g in remaining if g.support == max_support]

        best_mean = max(g.mean_confidence for g in same_support)

        # Các nhóm đủ gần confidence tốt nhất được xem như hòa về confidence.
        contenders = [
            g
            for g in same_support
            if best_mean - g.mean_confidence <= CONFIDENCE_TIE_EPSILON
        ]

        winner = min(
            contenders,
            key=lambda g: (
                g.best_source_rank,
                -g.max_confidence,
                -g.mean_confidence,
                g.canonical_answer,
            ),
        )

        ranked.append(winner)
        remaining.remove(winner)

    return ranked


def choose_consensus(
    question: str,
    candidates: Sequence[Any],
    *,
    question_type: str | None = None,
) -> ConsensusResult:
    """
    Chọn final answer mà không nhận/không dùng gt_answer.
    """
    qtype = question_type or loai_cau_hoi(question)
    groups = group_candidates(question, candidates, question_type=qtype)

    if not groups:
        fallback = doan_du_phong(question)
        answer = str(fallback.van_ban or DAP_AN_DU_PHONG).strip()
        if not answer:
            answer = DAP_AN_DU_PHONG
        return ConsensusResult(
            final_answer=answer,
            canonical_answer=canonicalize_answer(answer, question, qtype),
            confidence=0.0,
            support=0,
            sources=(FALLBACK_SOURCE,),
            alternatives=(),
            used_fallback=True,
            question_type=qtype,
        )

    winner = groups[0]

    alternatives = tuple(
        {
            "answer": g.representative_answer,
            "canonical_answer": g.canonical_answer,
            "support": g.support,
            "sources": list(g.sources),
            "mean_confidence": g.mean_confidence,
        }
        for g in groups[1:5]
    )

    final = winner.representative_answer.strip()
    if not final:
        # Invariant cuối: không bao giờ trả đáp án rỗng.
        fallback = doan_du_phong(question)
        final = str(fallback.van_ban or DAP_AN_DU_PHONG).strip() or DAP_AN_DU_PHONG
        return ConsensusResult(
            final_answer=final,
            canonical_answer=canonicalize_answer(final, question, qtype),
            confidence=0.0,
            support=0,
            sources=(FALLBACK_SOURCE,),
            alternatives=alternatives,
            used_fallback=True,
            question_type=qtype,
        )

    # Confidence output chỉ là diagnostic.
    # Bonus support nhẹ, clamp 1.0.
    fused_conf = min(
        1.0,
        winner.max_confidence + 0.1 * max(0, winner.support - 1),
    )

    return ConsensusResult(
        final_answer=final,
        canonical_answer=winner.canonical_answer,
        confidence=fused_conf,
        support=winner.support,
        sources=winner.sources,
        alternatives=alternatives,
        used_fallback=False,
        question_type=qtype,
    )


def consensus_record(
    *,
    query_id: str,
    video_id: str,
    question: str,
    candidates: Sequence[Any],
) -> dict[str, Any]:
    """
    Diagnostic record cho contracts/answer_consensus.jsonl.

    Cố ý KHÔNG có gt_answer trong input contract để tránh leakage.
    """
    result = choose_consensus(question, candidates)
    return {
        "schema_version": "1.0",
        "task": "QA-V4",
        "query_id": str(query_id),
        "video_id": str(video_id),
        "question_vi": str(question),
        "final_answer": result.final_answer,
        "canonical_answer": result.canonical_answer,
        "confidence": result.confidence,
        "support": result.support,
        "sources": list(result.sources),
        "alternatives": list(result.alternatives),
        "used_fallback": result.used_fallback,
        "question_type": result.question_type,
        "candidate_count": len(candidates),
    }


__all__ = [
    "Candidate",
    "ConsensusResult",
    "GroupSummary",
    "SOURCE_PRIORITY",
    "canonicalize_answer",
    "choose_consensus",
    "consensus_record",
    "group_candidates",
    "validate_confidence",
]
