"""
EVAL-03 (phần Q&A) — chạy end-to-end từ QA-R4 -> Reader -> QA-V4 consensus
-> submission đúng cặp frame-answer -> scorer BTC -> báo cáo cuối.

Phạm vi:
- CHỈ Q&A. Không chạy/không sửa TRAKE.
- Không dùng GT để sinh/rerank đáp án.
- GT chỉ được dùng SAU KHI đã tạo submission để chấm điểm.
- Giữ đúng cặp frame-answer: không clone một consensus answer lên mọi frame.
- Dùng scorer hiện có của repo để tính R@1/5/20/50/100 và Final Score.

Ví dụ:
    python -X utf8 -u -m scripts.eval_03_qa ^
      --r4 D:\aic-data\runs\qa_r4_adaptive_candidates.jsonl ^
      --dev D:\aic-data\dev\dev_questions_baseline_clean.jsonl

Chạy nhanh 1 câu:
    python -X utf8 -u -m scripts.eval_03_qa --limit 1

Không dùng VLM:
    python -X utf8 -u -m scripts.eval_03_qa --no-vlm

So paired với baseline_clean đã chấm trước đó:
    python -X utf8 -u -m scripts.eval_03_qa ^
      --baseline-csv D:\aic-data\derived\eval\baseline_clean_scoring_results.csv

Output:
    DATA_ROOT/runs/<timestamp>_EVAL-03_QA/
      contracts/eval_03_qa_results.jsonl
      submissions/query-<id>-qa.csv
      report/eval_03_qa_scores.csv
      report/eval_03_qa_summary.json
      manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from aic2026.qa_answer import (
    BoDocAnh,
    DAP_AN_DU_PHONG,
    bien_the_dap_an,
)
from scripts.adaptive_k import AdaptiveKResult
from scripts.answer_consensus import (
    canonicalize_answer,
    choose_consensus,
)
from scripts.qa_v4_pipeline import run_reader_from_r4

from src.aic2026.eval import K_THRESHOLDS, compute_final_score
from src.aic2026.submit import (
    QA,
    Answer,
    SubmissionBudget,
    submission_filename,
)


DEFAULT_R4 = Path(r"D:\aic-data\runs\qa_r4_adaptive_candidates.jsonl")
DEFAULT_DEV = Path(r"D:\aic-data\dev\dev_questions_baseline_clean.jsonl")
DEFAULT_RUNS = Path(r"D:\aic-data\runs")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL lỗi tại {path}, dòng {line_no}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{path}, dòng {line_no} phải là JSON object."
                )
            rows.append(obj)
    return rows


def _load_qa_dev(path: Path) -> dict[str, dict[str, Any]]:
    """
    Đọc schema dev thật của repo:
      id, loai_truy_van, cau_hoi, video_id,
      frame_start, frame_end, cau_tra_loi
    """
    out: dict[str, dict[str, Any]] = {}

    for row in _read_jsonl(path):
        if row.get("loai_truy_van") != "hoi_dap":
            continue

        query_id = str(row.get("id", "")).strip()
        question = str(row.get("cau_hoi", "")).strip()
        if not query_id or not question:
            continue

        required = (
            "video_id",
            "frame_start",
            "frame_end",
            "cau_tra_loi",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(
                f"QA {query_id} thiếu trường GT bắt buộc: {missing}"
            )

        out[query_id] = {
            "query_id": query_id,
            "question": question,
            "raw": row,
            "gt": {
                "gt_video_id": str(row["video_id"]),
                "gt_frame_range": [
                    int(row["frame_start"]),
                    int(row["frame_end"]),
                ],
                "gt_answer": str(row["cau_tra_loi"]).strip(),
            },
        }

    if not out:
        raise ValueError(f"Không tìm thấy câu hoi_dap nào trong {path}")

    return out


def _to_r4_result(record: dict[str, Any]) -> AdaptiveKResult:
    return AdaptiveKResult(
        query_id=str(record["query_id"]),
        k_requested=int(record["k_requested"]),
        reader_k_requested=int(record["reader_k_requested"]),
        k_effective=int(record["k_effective"]),
        k_available=int(record["k_available"]),
        selected_candidates=tuple(record["selected_candidates"]),
        status=str(record["status"]),
        fallback_reason=record.get("fallback_reason"),
    )


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]

    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _submission_dicts(budget: SubmissionBudget) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for answer in budget.answers:
        rows.append(
            {
                "video_id": str(answer.video_id),
                "frame_id": int(answer.frame_ids[0]),
                "answer": str(answer.answer or DAP_AN_DU_PHONG),
            }
        )
    return rows


def _retrieval_ceiling_submissions(
    submissions: list[dict[str, Any]],
    gt_answer: str,
) -> list[dict[str, Any]]:
    """
    Giữ nguyên video/frame/rank của output hệ thống, chỉ thay answer bằng GT.
    Đây là retrieval ceiling để tách phần điểm reader làm mất.
    """
    return [
        {
            "video_id": row["video_id"],
            "frame_id": row["frame_id"],
            "answer": gt_answer,
        }
        for row in submissions
    ]


def _rank_reader_outputs(
    question: str,
    reader_outputs: list[tuple[Any, Any]],
    consensus: Any,
    *,
    consensus_rerank: bool,
) -> list[tuple[Any, Any]]:
    """
    Không clone consensus answer lên frame khác.

    Nếu bật consensus_rerank:
      - frame nào TỰ sinh answer có canonical form trùng consensus thì được
        đưa lên trước;
      - trong mỗi nhóm vẫn giữ nguyên thứ tự retrieval QA-R4.

    Vì vậy mỗi dòng vẫn là evidence của chính frame đó.
    """
    if not consensus_rerank:
        return list(reader_outputs)

    winner = str(getattr(consensus, "canonical_answer", "") or "")
    if not winner:
        return list(reader_outputs)

    decorated: list[tuple[int, int, tuple[Any, Any]]] = []
    for original_rank, pair in enumerate(reader_outputs):
        _, answer = pair
        text = str(getattr(answer, "van_ban", "") or "")
        canonical = canonicalize_answer(text, question=question)
        match = canonical == winner
        decorated.append((0 if match else 1, original_rank, pair))

    decorated.sort(key=lambda x: (x[0], x[1]))
    return [pair for _, _, pair in decorated]


def _build_submission(
    question: str,
    ranked_reader_outputs: list[tuple[Any, Any]],
    *,
    max_rows: int,
    variants_per_frame: int,
) -> SubmissionBudget:
    """
    Tạo submission giữ đúng frame-answer.

    Mỗi frame chỉ nhận:
      answer mà Reader sinh TỪ frame đó
      + các biến thể chuẩn hóa của CHÍNH answer đó.

    Không lấy consensus final_answer gắn hàng loạt lên toàn bộ frame.
    """
    budget = SubmissionBudget(task=QA, limit=max_rows)

    for hit, dap_an in ranked_reader_outputs:
        answer_text = str(
            getattr(dap_an, "van_ban", "") or DAP_AN_DU_PHONG
        ).strip()
        if not answer_text:
            answer_text = DAP_AN_DU_PHONG

        variants = bien_the_dap_an(answer_text, question)
        variants = variants[: max(1, variants_per_frame)]

        for variant in variants:
            budget.add(
                Answer(
                    video_id=str(hit.video_id),
                    frame_ids=[int(hit.frame_idx)],
                    answer=str(variant).strip() or DAP_AN_DU_PHONG,
                )
            )

    return budget


def _read_baseline_scores(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy baseline CSV: {path}")

    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"query_id", "final_score"}.issubset(fields):
            raise ValueError(
                "Baseline CSV phải có ít nhất 2 cột: "
                "query_id, final_score"
            )

        for row in reader:
            query_id = str(row.get("query_id", "")).strip()
            score = str(row.get("final_score", "")).strip()
            if query_id and score:
                out[query_id] = float(score)

    return out


def _write_score_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "query_id",
        *[f"R@{k}" for k in K_THRESHOLDS],
        "final_score",
        "retrieval_ceiling",
        "reader_loss",
        "baseline_score",
        "delta_vs_baseline",
        "submission_rows",
        "reader_frames",
        "fallback_frames",
        "latency_ms",
        "status",
        "error",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: row.get(key) for key in fieldnames}
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "EVAL-03 Q&A: QA-R4 -> Reader -> QA-V4 -> "
            "submission -> scorer BTC."
        )
    )
    parser.add_argument("--r4", type=Path, default=DEFAULT_R4)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help=(
            "CSV baseline_clean đã chấm, có query_id và final_score. "
            "Nếu không truyền, script KHÔNG dùng mốc 0.0444 để giả làm "
            "baseline sạch."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help="Có thể truyền nhiều lần.",
    )
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument(
        "--no-consensus-rerank",
        action="store_true",
        help=(
            "Không đưa frame có answer đồng thuận lên trước; "
            "giữ nguyên thứ tự QA-R4."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Tối đa số dòng submission cho mỗi query.",
    )
    parser.add_argument(
        "--variants-per-frame",
        type=int,
        default=4,
        help=(
            "Số biến thể tối đa của CHÍNH answer từng frame. "
            "Không làm mất invariant frame-answer."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit phải > 0.")
    if not (1 <= args.max_rows <= 100):
        raise ValueError("--max-rows phải nằm trong [1, 100].")
    if args.variants_per_frame <= 0:
        raise ValueError("--variants-per-frame phải > 0.")

    started = datetime.now()
    stamp = started.strftime("%Y-%m-%d_%H%M%S")
    run_dir = args.runs_dir / f"{stamp}_EVAL-03_QA"
    contracts_dir = run_dir / "contracts"
    submissions_dir = run_dir / "submissions"
    report_dir = run_dir / "report"

    for path in (contracts_dir, submissions_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    contracts_path = contracts_dir / "eval_03_qa_results.jsonl"
    scores_path = report_dir / "eval_03_qa_scores.csv"
    summary_path = report_dir / "eval_03_qa_summary.json"
    manifest_path = run_dir / "manifest.json"

    dev = _load_qa_dev(args.dev)
    r4_records = _read_jsonl(args.r4)

    if args.query_id:
        wanted = {str(x).strip() for x in args.query_id}
        r4_records = [
            row
            for row in r4_records
            if str(row.get("query_id", "")).strip() in wanted
        ]

    if args.limit is not None:
        r4_records = r4_records[: args.limit]

    if not r4_records:
        raise ValueError("Không có QA-R4 record nào để chạy EVAL-03.")

    baseline = _read_baseline_scores(args.baseline_csv)

    use_vlm = not args.no_vlm
    consensus_rerank = not args.no_consensus_rerank

    print("=" * 84)
    print("EVAL-03 — Q&A END-TO-END")
    print("=" * 84)
    print(f"R4 input          : {args.r4}")
    print(f"DEV sạch          : {args.dev}")
    print(f"Queries           : {len(r4_records)}")
    print(f"VLM               : {use_vlm}")
    print(f"Consensus rerank  : {consensus_rerank}")
    print(f"Variants/frame    : {args.variants_per_frame}")
    print(f"Baseline paired   : {args.baseline_csv or 'N/A'}")
    print(f"Run dir           : {run_dir}")
    print()

    reader = None
    if use_vlm:
        print("Đang nạp BLIP Reader...")
        reader = BoDocAnh()
        reader._nap()
        print("BLIP Reader đã nạp.")
        print()

    score_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []

    with contracts_path.open("w", encoding="utf-8") as contracts_file:
        for idx, raw_r4 in enumerate(r4_records, start=1):
            query_id = str(raw_r4.get("query_id", "")).strip()
            t0 = time.perf_counter()

            try:
                if query_id not in dev:
                    raise KeyError(
                        f"Không tìm thấy query_id={query_id!r} trong DEV QA."
                    )

                dev_row = dev[query_id]
                question = dev_row["question"]
                gt = dev_row["gt"]
                result = _to_r4_result(raw_r4)

                # 1) Reader chạy RIÊNG từng frame, giữ cặp frame-answer.
                reader_outputs = list(
                    run_reader_from_r4(
                        cau_hoi=question,
                        result=result,
                        bo_doc_anh=reader,
                        dung_vlm=use_vlm,
                    )
                )

                # 2) QA-V4 consensus chỉ dùng để lấy tín hiệu đồng thuận.
                #    Không dùng GT; không clone final_answer lên frame khác.
                answer_candidates = [
                    dap_an for _, dap_an in reader_outputs
                ]
                consensus = choose_consensus(
                    question=question,
                    candidates=answer_candidates,
                )

                # 3) Có thể ưu tiên những frame TỰ có answer trùng consensus.
                ranked = _rank_reader_outputs(
                    question,
                    reader_outputs,
                    consensus,
                    consensus_rerank=consensus_rerank,
                )

                # 4) Tạo submission đúng pair frame-answer.
                budget = _build_submission(
                    question,
                    ranked,
                    max_rows=args.max_rows,
                    variants_per_frame=args.variants_per_frame,
                )
                if len(budget) == 0:
                    raise ValueError(
                        "Reader không tạo được dòng submission nào."
                    )

                submission_path = (
                    submissions_dir
                    / submission_filename(query_id, QA)
                )
                budget.write(submission_path)

                submissions = _submission_dicts(budget)

                # 5) Chấm đúng scorer của repo.
                scores = compute_final_score(gt, submissions, QA)

                # 6) Retrieval ceiling: giữ nguyên frame/rank,
                #    điền GT answer chỉ để đo trần retrieval.
                ceiling_submissions = _retrieval_ceiling_submissions(
                    submissions,
                    gt["gt_answer"],
                )
                ceiling_scores = compute_final_score(
                    gt,
                    ceiling_submissions,
                    QA,
                )
                ceiling = float(ceiling_scores["final_score"])
                final_score = float(scores["final_score"])
                reader_loss = max(0.0, ceiling - final_score)

                baseline_score = baseline.get(query_id)
                delta = (
                    final_score - baseline_score
                    if baseline_score is not None
                    else None
                )

                latency_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

                fallback_frames = sum(
                    1
                    for _, d in reader_outputs
                    if str(getattr(d, "nguon", "")) == "du_phong"
                )

                row = {
                    "query_id": query_id,
                    **scores,
                    "retrieval_ceiling": ceiling,
                    "reader_loss": reader_loss,
                    "baseline_score": baseline_score,
                    "delta_vs_baseline": delta,
                    "submission_rows": len(budget),
                    "reader_frames": len(reader_outputs),
                    "fallback_frames": fallback_frames,
                    "latency_ms": latency_ms,
                    "status": "ok",
                    "error": None,
                }
                score_rows.append(row)

                contract = {
                    "schema_version": "1.1",
                    "task": "EVAL-03-QA",
                    "query_id": query_id,
                    "question": question,
                    "r4": {
                        "k_requested": result.k_requested,
                        "reader_k_requested": result.reader_k_requested,
                        "k_effective": result.k_effective,
                        "k_available": result.k_available,
                        "status": result.status,
                        "fallback_reason": result.fallback_reason,
                    },
                    "reader_outputs": [
                        {
                            "rank_r4": rank,
                            "video_id": str(hit.video_id),
                            "n": int(hit.n),
                            "frame_idx": int(hit.frame_idx),
                            "pts_time": float(hit.pts_time),
                            "retrieval_score": float(hit.score),
                            "answer": str(dap_an.van_ban),
                            "confidence": float(dap_an.do_tin),
                            "source": str(dap_an.nguon),
                            "explanation": str(
                                getattr(dap_an, "giai_thich", "")
                            ),
                        }
                        for rank, (hit, dap_an) in enumerate(
                            reader_outputs,
                            start=1,
                        )
                    ],
                    "consensus": _jsonable(consensus),
                    "submission_file": submission_path.name,
                    "submission_rows": submissions,
                    "score": scores,
                    "retrieval_ceiling": ceiling_scores,
                    "reader_loss": reader_loss,
                    "latency_ms": latency_ms,
                    "status": "ok",
                    "error": None,
                }
                contract_rows.append(contract)

                contracts_file.write(
                    json.dumps(
                        contract,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                contracts_file.flush()

                print(
                    f"[{idx:02d}/{len(r4_records):02d}] "
                    f"{query_id} | "
                    f"score={final_score:.3f} | "
                    f"ceiling={ceiling:.3f} | "
                    f"loss={reader_loss:.3f} | "
                    f"rows={len(budget)} | "
                    f"{latency_ms:.0f} ms"
                )

            except Exception as exc:
                latency_ms = (
                    time.perf_counter() - t0
                ) * 1000.0
                error = f"{type(exc).__name__}: {exc}"

                row = {
                    "query_id": query_id,
                    **{f"R@{k}": None for k in K_THRESHOLDS},
                    "final_score": None,
                    "retrieval_ceiling": None,
                    "reader_loss": None,
                    "baseline_score": baseline.get(query_id),
                    "delta_vs_baseline": None,
                    "submission_rows": 0,
                    "reader_frames": 0,
                    "fallback_frames": 0,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": error,
                }
                score_rows.append(row)

                contract = {
                    "schema_version": "1.1",
                    "task": "EVAL-03-QA",
                    "query_id": query_id,
                    "status": "error",
                    "error": error,
                    "latency_ms": latency_ms,
                }
                contract_rows.append(contract)
                contracts_file.write(
                    json.dumps(
                        contract,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                contracts_file.flush()

                print(
                    f"[{idx:02d}/{len(r4_records):02d}] "
                    f"{query_id} | ERROR | {error}"
                )

    _write_score_csv(scores_path, score_rows)

    ok_rows = [r for r in score_rows if r["status"] == "ok"]
    error_rows = [r for r in score_rows if r["status"] != "ok"]

    if not ok_rows:
        print()
        print("Không có query nào chấm thành công.")
        return 1

    metric_means = {
        f"mean_R@{k}": (
            sum(float(r[f"R@{k}"]) for r in ok_rows)
            / len(ok_rows)
        )
        for k in K_THRESHOLDS
    }

    final_score = (
        sum(float(r["final_score"]) for r in ok_rows)
        / len(ok_rows)
    )
    retrieval_ceiling = (
        sum(float(r["retrieval_ceiling"]) for r in ok_rows)
        / len(ok_rows)
    )
    reader_loss = retrieval_ceiling - final_score

    latencies = [
        float(r["latency_ms"])
        for r in ok_rows
        if r.get("latency_ms") is not None
    ]

    paired = [
        r
        for r in ok_rows
        if r.get("baseline_score") is not None
    ]
    baseline_mean = (
        sum(float(r["baseline_score"]) for r in paired) / len(paired)
        if paired
        else None
    )
    paired_eval_mean = (
        sum(float(r["final_score"]) for r in paired) / len(paired)
        if paired
        else None
    )
    paired_delta = (
        paired_eval_mean - baseline_mean
        if baseline_mean is not None
        and paired_eval_mean is not None
        else None
    )

    summary = {
        "schema_version": "1.1",
        "task": "EVAL-03-QA",
        "scope": "qa_only_no_trake",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "inputs": {
            "r4": str(args.r4),
            "dev": str(args.dev),
            "baseline_csv": (
                str(args.baseline_csv)
                if args.baseline_csv is not None
                else None
            ),
        },
        "config": {
            "use_vlm": use_vlm,
            "consensus_rerank": consensus_rerank,
            "max_rows": args.max_rows,
            "variants_per_frame": args.variants_per_frame,
        },
        "queries_total": len(score_rows),
        "queries_ok": len(ok_rows),
        "queries_error": len(error_rows),
        **metric_means,
        "final_score": final_score,
        "retrieval_ceiling": retrieval_ceiling,
        "reader_loss": reader_loss,
        "fallback_frames_total": sum(
            int(r["fallback_frames"]) for r in ok_rows
        ),
        "latency_ms": {
            "mean": (
                sum(latencies) / len(latencies)
                if latencies
                else None
            ),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "paired_baseline": {
            "available": bool(paired),
            "queries": len(paired),
            "baseline_mean": baseline_mean,
            "eval_mean": paired_eval_mean,
            "delta": paired_delta,
            "improved_queries": sum(
                1
                for r in paired
                if float(r["delta_vs_baseline"]) > 0.0
            ),
            "equal_queries": sum(
                1
                for r in paired
                if float(r["delta_vs_baseline"]) == 0.0
            ),
            "worse_queries": sum(
                1
                for r in paired
                if float(r["delta_vs_baseline"]) < 0.0
            ),
        },
        "outputs": {
            "contracts_jsonl": str(contracts_path),
            "scores_csv": str(scores_path),
            "submissions_dir": str(submissions_dir),
            "summary_json": str(summary_path),
        },
        "notes": [
            (
                "Final Score dùng scorer hiện có của repo; "
                "không dùng MiniLM proxy của QA-V4 batch làm điểm cuối."
            ),
            (
                "Retrieval ceiling giữ nguyên video/frame/rank và chỉ "
                "thay answer bằng GT sau khi submission đã được sinh."
            ),
            (
                "Consensus không được clone answer lên mọi frame; "
                "mỗi frame giữ answer/evidence riêng."
            ),
            (
                "Mốc 0.0444 không tự động được dùng làm baseline_clean. "
                "Muốn paired comparison phải truyền --baseline-csv."
            ),
        ],
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "1.1",
        "run_id": run_dir.name,
        "task": "EVAL-03-QA",
        "created_at": started.isoformat(timespec="seconds"),
        "offline_only": True,
        "r4_input": str(args.r4),
        "dev_input": str(args.dev),
        "use_vlm": use_vlm,
        "consensus_rerank": consensus_rerank,
        "max_rows": args.max_rows,
        "variants_per_frame": args.variants_per_frame,
        "artifacts": {
            "contracts": str(contracts_path.relative_to(run_dir)),
            "scores": str(scores_path.relative_to(run_dir)),
            "summary": str(summary_path.relative_to(run_dir)),
            "submissions": str(submissions_dir.relative_to(run_dir)),
        },
    }
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 84)
    print("EVAL-03 Q&A SUMMARY")
    print("=" * 84)
    print(
        f"Queries OK/Error   : "
        f"{len(ok_rows)} / {len(error_rows)}"
    )
    for k in K_THRESHOLDS:
        print(
            f"Mean R@{k:<3}        : "
            f"{summary[f'mean_R@{k}']:.4f}"
        )
    print(f"FINAL SCORE        : {final_score:.4f}")
    print(f"Retrieval ceiling  : {retrieval_ceiling:.4f}")
    print(f"Reader loss        : {reader_loss:.4f}")

    if latencies:
        print(
            f"Latency p50/p95    : "
            f"{_percentile(latencies, 0.50):.0f} / "
            f"{_percentile(latencies, 0.95):.0f} ms"
        )

    if paired:
        print(
            f"Paired baseline    : "
            f"{baseline_mean:.4f} -> {paired_eval_mean:.4f} "
            f"(delta {paired_delta:+.4f}, n={len(paired)})"
        )
    else:
        print(
            "Paired baseline    : N/A "
            "(chưa truyền --baseline-csv)"
        )

    print(f"Scores CSV         : {scores_path}")
    print(f"Contracts JSONL    : {contracts_path}")
    print(f"Submissions        : {submissions_dir}")
    print(f"Summary JSON       : {summary_path}")
    print(f"Manifest           : {manifest_path}")
    print("=" * 84)

    return 0 if not error_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
