"""Adapters that connect the Streamlit UI to production pipelines.

The UI is intentionally kept thin: retrieval, QA answering, TRAKE alignment,
and submission formatting continue to live in their existing production
modules.  This file only normalizes free-form UI input/output.
"""

from __future__ import annotations

import csv
import io
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from aic2026.submit import KIS, QA, TRAKE, Answer, SubmissionBudget


SUPPORTED_TASKS = (KIS, QA, TRAKE)

_EVENT_PREFIX = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:(?:E\s*)?\d+\s*[.):\-]\s*|\(\d+\)\s*)",
    flags=re.IGNORECASE,
)
_EXPLICIT_EVENT_LINE = re.compile(
    r"^\s*(?:[-*•]\s*)?E\s*\d+\s*[.):\-]\s*(?P<text>.+?)\s*$",
    flags=re.IGNORECASE,
)


def ensure_repo_root_importable() -> Path:
    """Make the repository-level ``scripts`` package importable in Streamlit.

    ``streamlit run src/aic2026/ui/app.py`` may put only the app directory and
    ``src`` on ``sys.path``.  The production TRAKE runner lives in the sibling
    top-level ``scripts`` package, so resolve the repository from this file
    instead of depending on the terminal's current working directory.
    """

    repo_root = Path(__file__).resolve().parents[3]
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        raise ModuleNotFoundError(
            f"Không tìm thấy thư mục runner TRAKE: {scripts_dir}"
        )
    root_text = str(repo_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return repo_root


def normalize_task(value: str) -> str:
    """Return a canonical task name or fail before touching a pipeline."""

    task = str(value or "").strip().lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Dạng truy vấn không hỗ trợ: {value!r}")
    return task


def parse_trake_event_lines(text: str) -> list[str]:
    """Parse TRAKE events, tolerating an optional natural-language preamble."""

    raw_lines = [line for line in str(text or "").splitlines() if line.strip()]
    explicit_events = []
    for raw_line in raw_lines:
        match = _EXPLICIT_EVENT_LINE.match(raw_line)
        if match:
            explicit_events.append(match.group("text").strip())

    # Khi có ít nhất hai dòng E1/E2..., xem chúng là danh sách có chủ đích và
    # bỏ phần mô tả bối cảnh phía trước. Nếu không, vẫn hỗ trợ dạng (1), 2., ...
    events = explicit_events if len(explicit_events) >= 2 else [
        line
        for raw_line in raw_lines
        if (line := _EVENT_PREFIX.sub("", raw_line).strip())
    ]

    if len(events) < 2:
        raise ValueError("TRAKE cần ít nhất 2 sự kiện, mỗi sự kiện một dòng.")
    if len(events) > 12:
        raise ValueError("UI giới hạn 12 sự kiện TRAKE trong một truy vấn.")
    return events


def make_trake_inference_row(query_id: str, events: Sequence[str]) -> dict[str, Any]:
    """Build the runner contract without leaking ground truth into inference.

    ``scripts.run_trake_e2.run_one_query`` currently accepts the normalized dev
    row contract even though its frame fields are validation-only.  Zero-valued
    placeholders make that distinction explicit and are never consumed by the
    retrieval/alignment/refinement path.
    """

    qid = str(query_id or "").strip()
    if not qid:
        raise ValueError("Mã câu không được rỗng.")
    normalized_events = [str(event).strip() for event in events if str(event).strip()]
    if len(normalized_events) < 2:
        raise ValueError("TRAKE cần ít nhất 2 sự kiện.")

    return {
        "id": qid,
        "loai_truy_van": "chuoi_su_kien",
        "cac_giai_doan": [
            {
                "su_kien": event,
                "frame_start": 0,
                "frame_end": 0,
                "pts_time": 0.0,
            }
            for event in normalized_events
        ],
    }


def _frame_ids(item: dict[str, Any]) -> list[int]:
    values = item.get("frame_ids")
    if values is None:
        values = item.get("frame_idx")
    if isinstance(values, (int, float, str)):
        values = [values]
    if not isinstance(values, (list, tuple)):
        raise ValueError("Candidate thiếu frame_ids/frame_idx hợp lệ.")
    return [int(value) for value in values]


def answer_from_ui_item(task: str, item: dict[str, Any]) -> Answer:
    """Convert one UI/cart record to the single submission source of truth."""

    task = normalize_task(task)
    answer = item.get("answer") if task == QA else None
    return Answer(
        video_id=str(item.get("video_id", "")).strip(),
        frame_ids=_frame_ids(item),
        answer=None if answer is None else str(answer),
    )


def submission_csv_bytes(
    task: str,
    items: Iterable[dict[str, Any]],
    *,
    limit: int = 100,
) -> bytes:
    """Format ranked UI items as a headerless BTC-compatible CSV."""

    task = normalize_task(task)
    budget = SubmissionBudget(task=task, limit=int(limit))
    budget.extend(answer_from_ui_item(task, item) for item in items)

    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    for answer in budget.answers:
        writer.writerow(answer.to_row(task))
    return stream.getvalue().encode("utf-8")


def run_qa_answer_rows(
    question: str,
    hits: Sequence,
    *,
    use_vlm: bool,
    image_reader=None,
    max_rows: int = 100,
    vlm_rows: int = 5,
) -> dict[str, Any]:
    """Generate a non-empty answer for every ranked QA frame."""

    from aic2026.qa_answer import tra_loi_theo_hang

    started = time.perf_counter()
    pairs = tra_loi_theo_hang(
        str(question).strip(),
        list(hits),
        so_dong=int(max_rows),
        so_hang_vlm=int(vlm_rows),
        bo_doc_anh=image_reader,
        dung_vlm=bool(use_vlm),
        mo_rong_lan_can=True,
    )
    rows: list[dict[str, Any]] = []
    for hit, answer in pairs:
        rows.append(
            {
                "hit": hit,
                "video_id": str(hit.video_id),
                "frame_ids": [int(hit.frame_idx)],
                "answer": str(answer.van_ban),
                "answer_confidence": float(answer.do_tin),
                "answer_source": str(answer.nguon),
                "answer_explanation": str(answer.giai_thich),
                "other_answers": list(answer.ung_vien_khac),
            }
        )

    return {
        "rows": rows,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
    }


def run_trake_query(
    query_id: str,
    event_text: str,
    *,
    video_beam_size: int = 24,
    dense_rerank_top_k: int = 3,
    dense_rerank_margin: float = 0.015,
) -> dict[str, Any]:
    """Run the locked TR-R1 profile-union -> TR-R2 -> TR-E2 pipeline."""

    ensure_repo_root_importable()

    from aic2026.semantic.parser import RuleBasedParser
    from aic2026.trake_retrieval import TRR1Config
    from scripts.run_trake_e2 import run_one_query

    events = parse_trake_event_lines(event_text)
    row = make_trake_inference_row(query_id, events)
    config = TRR1Config(
        top_k=500,
        max_region_duration_seconds=10.0,
        region_merge_gap_seconds=2.0,
        min_region_duration_seconds=0.5,
        max_regions_per_event=10,
        video_consensus_weight=0.45,
        video_rrf_k=60.0,
    )

    started = time.perf_counter()
    result = run_one_query(
        row=row,
        parser_s1=RuleBasedParser(),
        trr1_config=config,
        video_beam_size=int(video_beam_size),
        dense_rerank_top_k=int(dense_rerank_top_k),
        dense_rerank_margin=float(dense_rerank_margin),
        trr1_profile_union=True,
    )
    result["ui_latency_ms"] = (time.perf_counter() - started) * 1000.0
    result["ui_config"] = {
        "video_beam_size": int(video_beam_size),
        "dense_rerank_top_k": int(dense_rerank_top_k),
        "dense_rerank_margin": float(dense_rerank_margin),
        "profile_union": True,
        "boundary_frozen": True,
    }
    return result


def dense_frame_path(data_root: Path, video_id: str, frame_idx: int) -> Path:
    """Return the canonical local dense-frame path used by TR-R2."""

    return (
        Path(data_root)
        / "derived"
        / "frames_dense"
        / str(video_id)
        / f"{int(frame_idx):06d}.jpg"
    )
