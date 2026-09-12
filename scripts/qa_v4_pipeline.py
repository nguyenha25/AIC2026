"""Stable R4 -> QA reader adapter used by EVAL-03.

This file restores the module imported by ``scripts.eval_03_qa``.  It performs
no retrieval and never reads ground truth; it only converts the exact R4
candidate prefix to the hit contract consumed by ``qa_answer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class R4ReaderHit:
    video_id: str
    n: int
    score: float
    frame_idx: int
    pts_time: float
    source: str


def _candidate_to_hit(candidate: dict[str, Any]) -> R4ReaderHit:
    video_id = str(candidate.get("video_id", "")).strip()
    if not video_id:
        raise ValueError("R4 candidate thiếu video_id")
    n = int(candidate["n"])
    frame_idx = int(candidate.get("frame_id", candidate.get("frame_idx")))
    pts_time = candidate.get("pts_time")
    if pts_time is None:
        try:
            from aic2026.frame_map import lookup

            _, pts_time = lookup(video_id, n)
        except Exception:
            pts_time = 0.0
    return R4ReaderHit(
        video_id=video_id,
        n=n,
        score=float(candidate.get("score", 0.0)),
        frame_idx=frame_idx,
        pts_time=float(pts_time),
        source=str(candidate.get("source", "qa_r4")),
    )


def run_reader_from_r4(
    *,
    cau_hoi: str,
    result,
    bo_doc_anh=None,
    dung_vlm: bool = True,
):
    """Return one independently generated answer for every selected R4 frame."""
    from aic2026.qa_answer import tra_loi_theo_hang

    candidates = list(result.selected_candidates)
    hits = [_candidate_to_hit(dict(candidate)) for candidate in candidates]
    return tra_loi_theo_hang(
        str(cau_hoi).strip(),
        hits,
        so_dong=len(hits),
        so_hang_vlm=len(hits),
        bo_doc_anh=bo_doc_anh,
        dung_vlm=bool(dung_vlm),
        mo_rong_lan_can=False,
    )


__all__ = ["R4ReaderHit", "run_reader_from_r4"]
