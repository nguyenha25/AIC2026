r"""Đánh giá Gemini Q&A trên dev AIC 2026.

Luồng end-to-end (candidate mode ``r4``):
    QA-R4 candidates -> local evidence rows -> Gemini assessment
    -> answer-only / rerank -> submission BTC -> scorer chính thức của repo.

Luồng chẩn đoán reader (candidate mode ``oracle_gt``):
    GT video/frame range -> keyframe oracle -> Gemini assessment -> scorer.
    Chế độ này không truyền GT answer vào reader, nhưng vẫn là diagnostic-only
    vì dùng vị trí GT để chọn ảnh.

Nguyên tắc chống leakage:
- GT không bao giờ được truyền vào reader, prompt hay reranker.
- GT chỉ được đọc trong hàm chấm sau khi output đã được tạo.
- Mỗi answer luôn gắn với đúng video/frame đã sinh ra nó.

Chạy smoke không gọi API:
    python -X utf8 -u -m scripts.eval_gemini_qa ^
      --providers none --limit 1

Chạy tune Gemini (12/18 câu, chia theo video):
    python -X utf8 -u -m scripts.eval_gemini_qa ^
      --dev D:\aic-data\dev\dev_questions_baseline_clean.jsonl ^
      --r4 D:\aic-data\runs\qa_r4_adaptive_candidates.jsonl ^
      --partition tune --holdout-size 6 ^
      --providers none gemini gemini_rerank ^
      --max-images 4 8 12

Chẩn đoán reader bằng frame GT (không phải điểm end-to-end):
    python -X utf8 -u -m scripts.eval_gemini_qa ^
      --candidate-mode oracle_gt --partition tune ^
      --providers none gemini gemini_rerank --max-images 4 8 12

Chấm lockbox sau khi đã khóa cấu hình:
    python -X utf8 -u -m scripts.eval_gemini_qa ^
      --partition holdout --holdout-size 6 ^
      --providers none gemini gemini_rerank ^
      --max-images 12
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

from aic2026.eval import K_THRESHOLDS, compute_final_score, compute_r_score  # noqa: E402
from aic2026.gemini_qa import (  # noqa: E402
    PROMPT_VERSION,
    GeminiQAError,
    GeminiQAReader,
    apply_gemini_assessments,
)
from aic2026.qa_answer import (  # noqa: E402
    BoDocAnh,
    DAP_AN_DU_PHONG,
    bien_the_dap_an,
)
from aic2026.semantic.parser import RuleBasedParser  # noqa: E402
from aic2026.submit import QA, Answer, SubmissionBudget, submission_filename  # noqa: E402
from aic2026.ui.pipelines import run_qa_answer_rows  # noqa: E402
from scripts.qa_v4_pipeline import R4ReaderHit  # noqa: E402


DEFAULT_DEV = Path(r"D:\aic-data\dev\dev_questions.jsonl")
DEFAULT_R4 = Path(r"D:\aic-data\runs\qa_r4_adaptive_candidates.jsonl")
DEFAULT_RUNS = Path(r"D:\aic-data\runs")
DEFAULT_FRAME_MAP = Path(
    os.getenv("DATA_ROOT", r"D:\aic-data")
) / "index" / "frame_map.parquet"
PROVIDERS = ("none", "local", "gemini", "gemini_rerank")
CANDIDATE_MODES = ("r4", "oracle_gt")


@dataclass(frozen=True)
class EvalConfig:
    config_id: str
    provider: str
    max_images: int | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL lỗi tại {path}, dòng {line_no}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}, dòng {line_no} phải là JSON object")
            rows.append(value)
    return rows


def load_qa_dev(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    parser = RuleBasedParser()
    for row in read_jsonl(path):
        if str(row.get("loai_truy_van", "")).strip() != "hoi_dap":
            continue
        query_id = str(row.get("id", "")).strip()
        question = str(row.get("cau_hoi", "")).strip()
        required = ("video_id", "frame_start", "frame_end", "cau_tra_loi")
        missing = [name for name in required if name not in row]
        if not query_id or not question or missing:
            if missing:
                raise ValueError(f"QA {query_id or '?'} thiếu trường GT: {missing}")
            continue
        plan = parser.parse_qa(query_id, question)
        output[query_id] = {
            "query_id": query_id,
            "question": question,
            "video_id": str(row["video_id"]),
            "intent": str(plan.intent),
            "answer_type": str(plan.answer_type),
            "gt": {
                "gt_video_id": str(row["video_id"]),
                "gt_frame_range": [int(row["frame_start"]), int(row["frame_end"])],
                "gt_answer": str(row["cau_tra_loi"]).strip(),
            },
        }
    if not output:
        raise ValueError(f"Không có dòng hoi_dap hợp lệ trong {path}")
    return output


def grouped_partition(
    dev: Mapping[str, Mapping[str, Any]],
    *,
    holdout_size: int,
    seed: str,
) -> dict[str, str]:
    """Chia tune/holdout xác định, không để cùng video ở hai phía."""
    if holdout_size <= 0 or holdout_size >= len(dev):
        raise ValueError("--holdout-size phải nằm giữa 1 và số câu QA - 1")

    groups: dict[str, list[str]] = {}
    for query_id, row in dev.items():
        groups.setdefault(str(row["video_id"]), []).append(str(query_id))

    ordered = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(
            f"{seed}\0{item[0]}".encode("utf-8")
        ).hexdigest(),
    )
    holdout_videos: set[str] = set()
    selected = 0
    for video_id, query_ids in ordered:
        if selected >= holdout_size:
            break
        holdout_videos.add(video_id)
        selected += len(query_ids)

    return {
        query_id: ("holdout" if str(row["video_id"]) in holdout_videos else "tune")
        for query_id, row in dev.items()
    }


def build_configs(providers: Sequence[str], max_images: Sequence[int]) -> list[EvalConfig]:
    output: list[EvalConfig] = []
    seen: set[str] = set()
    for provider in providers:
        if provider not in PROVIDERS:
            raise ValueError(f"Provider không hỗ trợ: {provider}")
        if provider.startswith("gemini"):
            for count in max_images:
                config_id = f"{provider}_img{int(count)}"
                if config_id not in seen:
                    output.append(EvalConfig(config_id, provider, int(count)))
                    seen.add(config_id)
        elif provider not in seen:
            output.append(EvalConfig(provider, provider, None))
            seen.add(provider)
    return output


def candidate_to_hit(candidate: Mapping[str, Any]) -> R4ReaderHit:
    video_id = str(candidate.get("video_id", "")).strip()
    if not video_id:
        raise ValueError("R4 candidate thiếu video_id")
    n = int(candidate["n"])
    frame_value = candidate.get("frame_id", candidate.get("frame_idx"))
    if frame_value is None:
        raise ValueError(f"R4 candidate {video_id} n={n} thiếu frame_id/frame_idx")
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
        frame_idx=int(frame_value),
        pts_time=float(pts_time),
        source=str(candidate.get("source", "qa_r4")),
    )


def load_r4(path: Path) -> dict[str, list[R4ReaderHit]]:
    output: dict[str, list[R4ReaderHit]] = {}
    for row in read_jsonl(path):
        query_id = str(row.get("query_id", row.get("id", ""))).strip()
        if not query_id:
            continue
        candidates = row.get("selected_candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"R4 query={query_id} thiếu selected_candidates list")
        output[query_id] = [candidate_to_hit(item) for item in candidates]
    if not output:
        raise ValueError(f"Không có QA-R4 record hợp lệ trong {path}")
    return output


def load_frame_map_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Đọc frame map một lần và nhóm keyframe theo video cho oracle reader."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy frame map: {path}")
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("oracle_gt cần pandas để đọc frame_map.parquet") from exc

    frame_map = pd.read_parquet(
        path,
        columns=["video_id", "n", "frame_idx", "pts_time"],
    )
    output: dict[str, list[dict[str, Any]]] = {}
    for video_id, n, frame_idx, pts_time in zip(
        frame_map["video_id"],
        frame_map["n"],
        frame_map["frame_idx"],
        frame_map["pts_time"],
    ):
        output.setdefault(str(video_id), []).append({
            "video_id": str(video_id),
            "n": int(n),
            "frame_idx": int(frame_idx),
            "pts_time": float(pts_time),
        })
    for rows in output.values():
        rows.sort(key=lambda row: (int(row["frame_idx"]), int(row["n"])))
    return output


def _coverage_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    midpoint: float,
) -> list[Mapping[str, Any]]:
    """Tạo prefix lồng nhau phủ đều: giữa -> hai đầu -> các khoảng trống."""
    remaining = list(rows)
    if not remaining:
        return []
    first = min(
        remaining,
        key=lambda row: (
            abs(int(row["frame_idx"]) - midpoint),
            int(row["frame_idx"]),
        ),
    )
    selected = [first]
    remaining.remove(first)
    while remaining:
        next_row = max(
            remaining,
            key=lambda row: (
                min(
                    abs(int(row["frame_idx"]) - int(chosen["frame_idx"]))
                    for chosen in selected
                ),
                -abs(int(row["frame_idx"]) - midpoint),
                -int(row["frame_idx"]),
            ),
        )
        selected.append(next_row)
        remaining.remove(next_row)
    return selected


def build_oracle_hits(
    query: Mapping[str, Any],
    frame_map_index: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    limit: int = 50,
) -> list[R4ReaderHit]:
    """Chọn keyframe từ GT location nhưng không đọc hoặc sao chép GT answer.

    Keyframe nằm trong khoảng GT luôn đứng trước. Prefix được sắp theo coverage
    để các grid 4/8/12 ảnh là các tập lồng nhau và đều nhìn thấy đầu/giữa/cuối.
    Nếu frame map không có keyframe trong đoạn GT, các frame gần biên nhất được
    giữ làm chẩn đoán nhưng ``frame_hit`` vẫn là false khi chấm.
    """
    if limit <= 0:
        raise ValueError("oracle limit phải > 0")
    gt = query["gt"]
    video_id = str(gt["gt_video_id"])
    start, end = (int(value) for value in gt["gt_frame_range"])
    if start > end:
        start, end = end, start
    video_rows = list(frame_map_index.get(video_id, ()))
    if not video_rows:
        raise ValueError(f"Frame map không có video GT {video_id}")

    midpoint = (start + end) / 2.0
    inside = [
        row for row in video_rows
        if start <= int(row["frame_idx"]) <= end
    ]
    ordered_inside = _coverage_order(inside, midpoint=midpoint)
    outside = [
        row for row in video_rows
        if not (start <= int(row["frame_idx"]) <= end)
    ]
    outside.sort(key=lambda row: (
        min(
            abs(int(row["frame_idx"]) - start),
            abs(int(row["frame_idx"]) - end),
        ),
        abs(int(row["frame_idx"]) - midpoint),
        int(row["frame_idx"]),
    ))
    selected = (ordered_inside + outside)[: int(limit)]
    return [
        R4ReaderHit(
            video_id=video_id,
            n=int(row["n"]),
            score=max(0.0, 1.0 - rank * 1e-6),
            frame_idx=int(row["frame_idx"]),
            pts_time=float(row["pts_time"]),
            source=(
                "oracle_gt"
                if start <= int(row["frame_idx"]) <= end
                else "oracle_context"
            ),
        )
        for rank, row in enumerate(selected)
    ]


def apply_gemini_eval_rows(
    rows: Sequence[Mapping[str, Any]],
    assessments: Mapping[int, Any],
    *,
    rerank: bool,
    reader_only: bool,
) -> list[dict[str, Any]]:
    """Áp assessment; oracle reader-only không giữ câu trả lời local ẩn.

    Ở mode R4, hành vi giữ nguyên production hybrid: các row không được Gemini
    xem vẫn giữ câu trả lời local. Ở oracle, chỉ row thực sự được Gemini xem
    mới được chấm và cả câu trả lời abstention cũng được giữ nguyên là abstain.
    """
    if not reader_only:
        return apply_gemini_assessments(rows, assessments, rerank=rerank)

    output: list[tuple[int, dict[str, Any], Any]] = []
    for row_index, assessment in assessments.items():
        index = int(row_index)
        if not 0 <= index < len(rows):
            continue
        row = dict(rows[index])
        row["gemini_relevance"] = float(assessment.relevance)
        row["gemini_confidence"] = float(assessment.confidence)
        row["gemini_evidence"] = str(assessment.evidence)
        row["answer"] = str(assessment.answer).strip() or DAP_AN_DU_PHONG
        row["answer_confidence"] = float(assessment.confidence)
        row["answer_source"] = "gemini"
        row["answer_explanation"] = str(assessment.evidence) or (
            "Gemini multimodal trên đúng candidate oracle"
        )
        output.append((index, row, assessment))

    if rerank:
        output.sort(
            key=lambda item: (
                0.70 * float(item[2].relevance) + 0.30 / (1.0 + item[0]),
                -item[0],
            ),
            reverse=True,
        )
    else:
        output.sort(key=lambda item: item[0])
    return [row for _, row, _ in output]


def rows_to_budget(
    question: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int,
    variants_per_frame: int,
) -> SubmissionBudget:
    budget = SubmissionBudget(task=QA, limit=int(max_rows))
    for row in rows:
        hit = row["hit"]
        answer = str(row.get("answer") or DAP_AN_DU_PHONG).strip()
        variants = bien_the_dap_an(answer, question)[: max(1, variants_per_frame)]
        for variant in variants:
            budget.add(Answer(
                video_id=str(hit.video_id),
                frame_ids=[int(hit.frame_idx)],
                answer=str(variant).strip() or DAP_AN_DU_PHONG,
            ))
    return budget


def budget_dicts(budget: SubmissionBudget) -> list[dict[str, Any]]:
    return [
        {
            "video_id": str(answer.video_id),
            "frame_id": int(answer.frame_ids[0]),
            "answer": str(answer.answer or DAP_AN_DU_PHONG),
        }
        for answer in budget.answers
    ]


def first_rank(values: Iterable[bool]) -> int | None:
    for rank, matched in enumerate(values, start=1):
        if matched:
            return rank
    return None


def score_rows(
    *,
    config: EvalConfig,
    query: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    provider_meta: Mapping[str, Any],
    elapsed_ms: float,
    max_rows: int,
    variants_per_frame: int,
    submission_path: Path,
) -> dict[str, Any]:
    question = str(query["question"])
    gt = dict(query["gt"])
    budget = rows_to_budget(
        question,
        rows,
        max_rows=max_rows,
        variants_per_frame=variants_per_frame,
    )
    if not budget.answers:
        raise ValueError("Không tạo được dòng submission nào")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    budget.write(submission_path)
    submissions = budget_dicts(budget)

    scores = compute_final_score(gt, submissions, QA)
    ceiling_submissions = [
        {**row, "answer": gt["gt_answer"]}
        for row in submissions
    ]
    ceiling_scores = compute_final_score(gt, ceiling_submissions, QA)

    gt_video = str(gt["gt_video_id"])
    start, end = (int(value) for value in gt["gt_frame_range"])
    row_video_flags = [str(row["video_id"]) == gt_video for row in rows]
    row_frame_flags = [
        str(row["video_id"]) == gt_video
        and start <= int(row["frame_ids"][0]) <= end
        for row in rows
    ]
    submission_correct_flags = [
        compute_r_score(QA, gt, submitted) > 0
        for submitted in submissions
    ]
    gt_frame_rows = [row for row, is_gt in zip(rows, row_frame_flags) if is_gt]
    correct_gt_frame_rows = sum(
        compute_r_score(
            QA,
            gt,
            {
                "video_id": str(row["video_id"]),
                "frame_id": int(row["frame_ids"][0]),
                "answer": str(row.get("answer") or ""),
            },
        ) > 0
        for row in gt_frame_rows
    )

    video_rank = first_rank(row_video_flags)
    frame_rank = first_rank(row_frame_flags)
    correct_rank = first_rank(submission_correct_flags)
    if video_rank is None:
        failure_type = "retrieval_video_miss"
    elif frame_rank is None:
        failure_type = "retrieval_frame_miss"
    elif correct_rank is None:
        failure_type = "reader_miss"
    else:
        failure_type = "ok"

    actual_provider = str(provider_meta.get("provider", config.provider))
    fallback_reason = provider_meta.get("fallback_reason")
    return {
        "schema_version": "1.0",
        "query_id": str(query["query_id"]),
        "question": question,
        "intent": str(query["intent"]),
        "answer_type": str(query["answer_type"]),
        "gt_video_id": gt_video,
        "gt_frame_range": [start, end],
        "gt_answer": str(gt["gt_answer"]),
        "config_id": config.config_id,
        "requested_provider": config.provider,
        "actual_provider": actual_provider,
        "max_images": config.max_images,
        "selected_images": int(provider_meta.get("selected_images", 0) or 0),
        "selected_videos": int(provider_meta.get("selected_videos", 0) or 0),
        "cache_hit": bool(provider_meta.get("cache_hit", False)),
        "api_calls": int(provider_meta.get("api_calls", 0) or 0),
        "json_recovery": provider_meta.get("json_recovery"),
        "fallback_reason": fallback_reason,
        "provider_fallback": bool(fallback_reason) or actual_provider == "local_fallback",
        "latency_ms": float(elapsed_ms),
        "row_count": len(rows),
        "submission_rows": len(submissions),
        "video_hit": video_rank is not None,
        "video_rank": video_rank,
        "frame_hit": frame_rank is not None,
        "frame_rank": frame_rank,
        "correct_answer_hit": correct_rank is not None,
        "first_correct_submission_rank": correct_rank,
        "gt_frame_rows": len(gt_frame_rows),
        "correct_gt_frame_rows": correct_gt_frame_rows,
        "answer_accuracy_given_gt_frame": (
            correct_gt_frame_rows / len(gt_frame_rows) if gt_frame_rows else None
        ),
        **{key: float(value) for key, value in scores.items()},
        "retrieval_ceiling": float(ceiling_scores["final_score"]),
        "reader_loss": max(
            0.0,
            float(ceiling_scores["final_score"]) - float(scores["final_score"]),
        ),
        "failure_type": failure_type,
        "answers": [
            {
                "rank": rank,
                "video_id": str(row["video_id"]),
                "frame_idx": int(row["frame_ids"][0]),
                "answer": str(row.get("answer") or ""),
                "source": str(row.get("answer_source") or ""),
                "confidence": float(row.get("answer_confidence", 0.0) or 0.0),
                "gemini_relevance": row.get("gemini_relevance"),
            }
            for rank, row in enumerate(rows, start=1)
        ],
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def provider_failure_category(reason: Any) -> str | None:
    """Rút gọn lỗi provider thành nhãn ổn định cho CSV/console."""
    text = str(reason or "").casefold()
    if not text:
        return None
    if "http 429" in text or "quota" in text or "rate limit" in text:
        return "quota_exceeded"
    if "json" in text:
        return "invalid_json"
    if "không có ảnh" in text or "missing image" in text:
        return "missing_image"
    if "http 401" in text or "http 403" in text or "api key" in text:
        return "authentication"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    return "provider_error"


def summarize(records: Sequence[Mapping[str, Any]], config: EvalConfig) -> dict[str, Any]:
    rows = [row for row in records if row.get("config_id") == config.config_id]
    status_ok = [row for row in rows if row.get("status") == "ok"]
    is_cloud = config.provider.startswith("gemini")
    provider_valid = [
        row for row in status_ok
        if not (is_cloud and bool(row.get("provider_fallback")))
    ]
    metric_rows = [
        row for row in provider_valid
        if bool(row.get("oracle_eligible", True))
    ]
    latencies = [float(row["latency_ms"]) for row in metric_rows]
    gt_frame_total = sum(int(row.get("gt_frame_rows", 0)) for row in metric_rows)
    correct_gt_total = sum(
        int(row.get("correct_gt_frame_rows", 0)) for row in metric_rows
    )

    def mean(
        name: str,
        source: Sequence[Mapping[str, Any]] = metric_rows,
    ) -> float | None:
        values = [float(row[name]) for row in source if row.get(name) is not None]
        return statistics.fmean(values) if values else None

    fallback_queries = sum(
        bool(row.get("provider_fallback")) for row in status_ok
    )
    excluded_oracle_gaps = len(provider_valid) - len(metric_rows)

    return {
        "config_id": config.config_id,
        "candidate_mode": (
            str(rows[0].get("candidate_mode", "r4")) if rows else None
        ),
        "diagnostic_only": (
            bool(rows[0].get("diagnostic_only", False)) if rows else False
        ),
        "provider": config.provider,
        "max_images": config.max_images,
        "queries": len(rows),
        "status_ok_queries": len(status_ok),
        "successful_queries": len(provider_valid),
        "gemini_valid_queries": len(provider_valid) if is_cloud else None,
        "metric_queries": len(metric_rows),
        "oracle_eligible_queries": sum(
            bool(row.get("oracle_eligible", True)) for row in provider_valid
        ),
        "excluded_oracle_gap_queries": excluded_oracle_gaps,
        "error_queries": len(rows) - len(status_ok),
        "mean_final_score": mean("final_score"),
        "mean_final_score_all_status_ok": mean("final_score", status_ok),
        **{f"mean_R@{k}": mean(f"R@{k}") for k in K_THRESHOLDS},
        "mean_retrieval_ceiling": mean("retrieval_ceiling"),
        "mean_reader_loss": mean("reader_loss"),
        "mean_delta_vs_none": mean("delta_vs_none"),
        "video_recall": (
            sum(bool(row.get("video_hit")) for row in metric_rows) / len(metric_rows)
            if metric_rows else None
        ),
        "frame_recall": (
            sum(bool(row.get("frame_hit")) for row in metric_rows) / len(metric_rows)
            if metric_rows else None
        ),
        "answer_hit_rate": (
            sum(bool(row.get("correct_answer_hit")) for row in metric_rows)
            / len(metric_rows)
            if metric_rows else None
        ),
        "answer_accuracy_given_gt_frame": (
            correct_gt_total / gt_frame_total if gt_frame_total else None
        ),
        "fallback_queries": fallback_queries,
        "json_retry_queries": sum(
            row.get("json_recovery") == "retry" for row in status_ok
        ),
        "cache_hit_queries": sum(
            bool(row.get("cache_hit")) for row in status_ok
        ),
        "api_calls": sum(int(row.get("api_calls", 0)) for row in status_ok),
        "evaluation_valid": (
            len(rows) > 0
            and len(rows) == len(status_ok)
            and fallback_queries == 0
            and len(metric_rows) > 0
        ),
        "latency_mean_ms": statistics.fmean(latencies) if latencies else None,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A/B benchmark local/Gemini QA bằng scorer BTC của repo."
    )
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--r4", type=Path, default=DEFAULT_R4)
    parser.add_argument("--frame-map", type=Path, default=DEFAULT_FRAME_MAP)
    parser.add_argument(
        "--candidate-mode",
        choices=CANDIDATE_MODES,
        default="r4",
        help=(
            "r4 = đánh giá end-to-end; oracle_gt = dùng GT location để "
            "chẩn đoán reader, không phải điểm hệ thống."
        ),
    )
    parser.add_argument(
        "--oracle-candidates",
        type=int,
        default=50,
        help="Số keyframe oracle tối đa trước bước chọn --max-images.",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--providers", nargs="+", choices=PROVIDERS,
        default=["none", "gemini", "gemini_rerank"],
    )
    parser.add_argument("--max-images", nargs="+", type=int, default=[12])
    parser.add_argument("--partition", choices=("all", "tune", "holdout"), default="all")
    parser.add_argument("--holdout-size", type=int, default=6)
    parser.add_argument("--split-seed", default="aic2026-qa-v1")
    parser.add_argument("--query-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--variants-per-frame", type=int, default=1)
    parser.add_argument("--local-vlm-rows", type=int, default=5)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Mặc định dùng cache riêng trong run để đo cold-start công bằng.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit phải > 0")
    if not (1 <= args.max_rows <= 100):
        raise ValueError("--max-rows phải nằm trong [1, 100]")
    if args.variants_per_frame <= 0:
        raise ValueError("--variants-per-frame phải > 0")
    if args.local_vlm_rows < 0:
        raise ValueError("--local-vlm-rows phải >= 0")
    if not (1 <= args.oracle_candidates <= 100):
        raise ValueError("--oracle-candidates phải nằm trong [1, 100]")
    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds phải >= 0")
    if any(value <= 0 or value > 12 for value in args.max_images):
        raise ValueError("Mỗi --max-images phải nằm trong [1, 12]")

    configs = build_configs(args.providers, args.max_images)
    dev_all = load_qa_dev(args.dev)
    partition_map = (
        {query_id: "all" for query_id in dev_all}
        if args.partition == "all"
        else grouped_partition(
            dev_all, holdout_size=args.holdout_size, seed=args.split_seed
        )
    )
    selected_ids = [
        query_id
        for query_id in dev_all
        if args.partition == "all" or partition_map[query_id] == args.partition
    ]
    if args.query_id:
        wanted = {str(value).strip() for value in args.query_id}
        selected_ids = [query_id for query_id in selected_ids if query_id in wanted]
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]
    if not selected_ids:
        raise ValueError("Partition/bộ lọc không còn câu QA nào")

    if args.candidate_mode == "r4":
        candidates_by_query = load_r4(args.r4)
        missing_r4 = [
            query_id for query_id in selected_ids
            if query_id not in candidates_by_query
        ]
        if missing_r4:
            raise ValueError(f"Thiếu QA-R4 cho query: {missing_r4}")
    else:
        frame_map_index = load_frame_map_index(args.frame_map)
        candidates_by_query = {
            query_id: build_oracle_hits(
                dev_all[query_id],
                frame_map_index,
                limit=args.oracle_candidates,
            )
            for query_id in selected_ids
        }

    started = datetime.now()
    stamp = started.strftime("%Y-%m-%d_%H%M%S")
    run_dir = args.runs_dir / (
        f"{stamp}_EVAL-GEMINI-QA_{args.candidate_mode}_{args.partition}"
    )
    report_dir = run_dir / "report"
    submissions_dir = run_dir / "submissions"
    cache_root = args.cache_dir or run_dir / "cache"
    for path in (report_dir, submissions_dir, cache_root):
        path.mkdir(parents=True, exist_ok=True)

    need_cloud = any(config.provider.startswith("gemini") for config in configs)
    readers: dict[int, GeminiQAReader] = {}
    if need_cloud:
        for max_images in sorted({int(c.max_images) for c in configs if c.max_images}):
            reader = GeminiQAReader(
                max_images=max_images,
                cache_dir=cache_root / f"images_{max_images}",
            )
            if not reader.configured:
                raise RuntimeError(
                    "Thiếu GEMINI_API_KEY. Evaluator dừng thay vì âm thầm "
                    "chấm local_fallback như một cấu hình Gemini."
                )
            readers[max_images] = reader

    local_reader = None
    if any(config.provider == "local" for config in configs):
        print("Đang nạp BLIP local...")
        local_reader = BoDocAnh()
        local_reader._nap()

    print("=" * 88)
    print("EVAL-GEMINI-QA — DEV ABLATION")
    print("=" * 88)
    print(f"DEV       : {args.dev}")
    print(f"Candidates: {args.candidate_mode}")
    if args.candidate_mode == "r4":
        print(f"QA-R4     : {args.r4}")
    else:
        print(f"Frame map : {args.frame_map}")
        print("CẢNH BÁO   : oracle_gt là DIAGNOSTIC-ONLY, không phải điểm end-to-end")
    print(f"Partition : {args.partition} ({len(selected_ids)} câu)")
    print(f"Configs   : {', '.join(config.config_id for config in configs)}")
    print(f"Run dir   : {run_dir}")
    print()

    records: list[dict[str, Any]] = []
    unique_api_calls = 0
    for query_number, query_id in enumerate(selected_ids, start=1):
        query = dev_all[query_id]
        question = str(query["question"])
        hits = candidates_by_query[query_id]
        oracle_eligible = (
            args.candidate_mode != "oracle_gt"
            or any(str(getattr(hit, "source", "")) == "oracle_gt" for hit in hits)
        )

        base_started = time.perf_counter()
        base = run_qa_answer_rows(
            question,
            hits,
            use_vlm=False,
            image_reader=None,
            max_rows=args.max_rows,
            vlm_rows=0,
            # QA-R4 đã chốt prefix reader; không thay đổi contract bằng n±2.
            expand_neighbors=False,
        )
        base_elapsed = (time.perf_counter() - base_started) * 1000.0
        base_rows = list(base["rows"])

        generated: dict[str, tuple[list[dict[str, Any]], dict[str, Any], float]] = {}
        if any(config.provider == "none" for config in configs):
            generated["none"] = (
                base_rows,
                {"provider": "none", "fallback_reason": None},
                base_elapsed,
            )

        if any(config.provider == "local" for config in configs):
            local_started = time.perf_counter()
            local = run_qa_answer_rows(
                question,
                hits,
                use_vlm=True,
                image_reader=local_reader,
                max_rows=args.max_rows,
                vlm_rows=args.local_vlm_rows,
                expand_neighbors=False,
            )
            generated["local"] = (
                list(local["rows"]),
                {
                    "provider": "local",
                    "model": getattr(local_reader, "ten_mo_hinh", "BLIP-VQA"),
                    "fallback_reason": None,
                },
                (time.perf_counter() - local_started) * 1000.0,
            )

        for max_images, reader in readers.items():
            cloud_started = time.perf_counter()
            try:
                assessments, meta = reader.assess(question, base_rows)
                cloud_elapsed = base_elapsed + (
                    time.perf_counter() - cloud_started
                ) * 1000.0
                unique_api_calls += int(meta.get("api_calls", 0) or 0)
                if "gemini" in args.providers:
                    generated[f"gemini_img{max_images}"] = (
                        apply_gemini_eval_rows(
                            base_rows,
                            assessments,
                            rerank=False,
                            reader_only=args.candidate_mode == "oracle_gt",
                        ),
                        {**meta, "provider": "gemini"},
                        cloud_elapsed,
                    )
                if "gemini_rerank" in args.providers:
                    generated[f"gemini_rerank_img{max_images}"] = (
                        apply_gemini_eval_rows(
                            base_rows,
                            assessments,
                            rerank=True,
                            reader_only=args.candidate_mode == "oracle_gt",
                        ),
                        {**meta, "provider": "gemini_rerank"},
                        cloud_elapsed,
                    )
            except GeminiQAError as exc:
                cloud_elapsed = base_elapsed + (
                    time.perf_counter() - cloud_started
                ) * 1000.0
                failed_calls = 2 if "retry thất bại" in str(exc) else 1
                unique_api_calls += failed_calls
                meta = {
                    "provider": "local_fallback",
                    "model": reader.model,
                    "selected_images": 0,
                    "selected_videos": 0,
                    "cache_hit": False,
                    "api_calls": failed_calls,
                    "json_recovery": None,
                    "fallback_reason": str(exc),
                }
                for provider in ("gemini", "gemini_rerank"):
                    if provider in args.providers:
                        generated[f"{provider}_img{max_images}"] = (
                            base_rows, meta, cloud_elapsed
                        )
            if args.delay_seconds and readers:
                time.sleep(args.delay_seconds)

        for config in configs:
            rows, meta, elapsed = generated[config.config_id]
            submission_path = (
                submissions_dir
                / config.config_id
                / submission_filename(query_id, QA)
            )
            try:
                record = score_rows(
                    config=config,
                    query=query,
                    rows=rows,
                    provider_meta=meta,
                    elapsed_ms=elapsed,
                    max_rows=args.max_rows,
                    variants_per_frame=args.variants_per_frame,
                    submission_path=submission_path,
                )
                record["partition"] = partition_map[query_id]
                record["candidate_mode"] = args.candidate_mode
                record["diagnostic_only"] = args.candidate_mode == "oracle_gt"
                record["oracle_eligible"] = oracle_eligible
                if args.candidate_mode == "oracle_gt" and not oracle_eligible:
                    record["failure_type"] = "oracle_keyframe_gap"
                record["status"] = "ok"
                record["error"] = None
            except Exception as exc:
                record = {
                    "schema_version": "1.0",
                    "query_id": query_id,
                    "question": question,
                    "intent": query["intent"],
                    "answer_type": query["answer_type"],
                    "config_id": config.config_id,
                    "requested_provider": config.provider,
                    "max_images": config.max_images,
                    "partition": partition_map[query_id],
                    "candidate_mode": args.candidate_mode,
                    "diagnostic_only": args.candidate_mode == "oracle_gt",
                    "oracle_eligible": oracle_eligible,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": elapsed,
                    "provider_fallback": bool(meta.get("fallback_reason")),
                }
            record["provider_failure_category"] = provider_failure_category(
                record.get("fallback_reason")
            )
            records.append(record)
            score_text = (
                f"{record['final_score']:.3f}"
                if record.get("final_score") is not None else "ERROR"
            )
            print(
                f"[{query_number:02d}/{len(selected_ids):02d}] {query_id} | "
                f"{config.config_id:<24} | score={score_text} | "
                f"failure={record.get('failure_type', record.get('error'))}"
            )

    none_scores = {
        str(record["query_id"]): float(record["final_score"])
        for record in records
        if record.get("config_id") == "none"
        and record.get("status") == "ok"
        and record.get("final_score") is not None
    }
    for record in records:
        baseline_score = none_scores.get(str(record.get("query_id")))
        record["baseline_none_score"] = baseline_score
        record["delta_vs_none"] = (
            float(record["final_score"]) - baseline_score
            if baseline_score is not None and record.get("final_score") is not None
            else None
        )

    per_query_path = report_dir / "per_query.jsonl"
    with per_query_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    failures = [
        record for record in records
        if record.get("status") != "ok"
        or record.get("failure_type") != "ok"
        or record.get("provider_fallback")
    ]
    compact_failure_fields = (
        "query_id",
        "config_id",
        "candidate_mode",
        "status",
        "failure_type",
        "provider_fallback",
        "provider_failure_category",
        "fallback_reason",
        "oracle_eligible",
        "video_hit",
        "frame_hit",
        "final_score",
        "error",
    )
    with (report_dir / "failures.jsonl").open("w", encoding="utf-8") as stream:
        for record in failures:
            compact = {key: record.get(key) for key in compact_failure_fields}
            stream.write(json.dumps(compact, ensure_ascii=False) + "\n")

    fallback_columns = (
        "query_id",
        "config_id",
        "requested_provider",
        "actual_provider",
        "max_images",
        "provider_failure_category",
        "fallback_reason",
    )
    fallback_rows = [
        {key: record.get(key) for key in fallback_columns}
        for record in records
        if record.get("provider_fallback")
    ]
    fallback_path = report_dir / "fallbacks.csv"
    if fallback_rows:
        write_csv(fallback_path, fallback_rows)
    else:
        fallback_path.write_text(
            ",".join(fallback_columns) + "\n",
            encoding="utf-8-sig",
        )

    summaries = [summarize(records, config) for config in configs]
    write_csv(report_dir / "ablation.csv", summaries)
    intent_summaries: list[dict[str, Any]] = []
    intents = sorted({str(row.get("intent")) for row in records})
    for config in configs:
        for intent in intents:
            subset = [row for row in records if str(row.get("intent")) == intent]
            item = summarize(subset, config)
            item["intent"] = intent
            intent_summaries.append(item)
    write_csv(report_dir / "by_intent.csv", intent_summaries)
    valid_summaries = [
        row for row in summaries
        if row.get("mean_final_score") is not None
        and bool(row.get("evaluation_valid"))
    ]
    winner = max(
        valid_summaries,
        key=lambda row: (
            float(row["mean_final_score"]),
            -float(row.get("latency_mean_ms") or float("inf")),
        ),
        default=None,
    )
    summary = {
        "schema_version": "1.0",
        "task": "EVAL-GEMINI-QA",
        "partition": args.partition,
        "candidate_mode": args.candidate_mode,
        "diagnostic_only": args.candidate_mode == "oracle_gt",
        "query_count": len(selected_ids),
        "configs": summaries,
        "by_intent": intent_summaries,
        "winner_by_mean_final_score": winner,
        "unique_api_calls_in_this_run": unique_api_calls,
        "prompt_version": PROMPT_VERSION,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "schema_version": "1.0",
        "task": "EVAL-GEMINI-QA",
        "dev": {"path": str(args.dev), "sha256": sha256_file(args.dev)},
        "candidate_mode": args.candidate_mode,
        "diagnostic_only": args.candidate_mode == "oracle_gt",
        "r4": (
            {"path": str(args.r4), "sha256": sha256_file(args.r4)}
            if args.candidate_mode == "r4" else None
        ),
        "frame_map": (
            {"path": str(args.frame_map), "sha256": sha256_file(args.frame_map)}
            if args.candidate_mode == "oracle_gt" else None
        ),
        "partition": args.partition,
        "partition_map": partition_map,
        "holdout_size_requested": args.holdout_size,
        "split_seed": args.split_seed,
        "selected_query_ids": selected_ids,
        "configs": [config.__dict__ for config in configs],
        "max_rows": args.max_rows,
        "variants_per_frame": args.variants_per_frame,
        "local_vlm_rows": args.local_vlm_rows,
        "oracle_candidates": (
            args.oracle_candidates if args.candidate_mode == "oracle_gt" else None
        ),
        "cache_dir": str(cache_root),
        "gemini_model": (
            next(iter(readers.values())).model if readers else None
        ),
        "prompt_version": PROMPT_VERSION,
        "api_key_recorded": False,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("TÓM TẮT")
    for row in summaries:
        score = row.get("mean_final_score")
        score_text = f"{score:.4f}" if score is not None else "N/A"
        print(
            f"- {row['config_id']}: score={score_text}, "
            f"video_recall={row.get('video_recall')}, "
            f"frame_recall={row.get('frame_recall')}, "
            f"valid={row.get('successful_queries')}/{row.get('queries')}, "
            f"oracle_gap={row.get('excluded_oracle_gap_queries')}, "
            f"fallback={row.get('fallback_queries')}, "
            f"api_calls={row.get('api_calls')}"
        )
    print(f"Báo cáo: {report_dir}")
    return 0 if valid_summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
