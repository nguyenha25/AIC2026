from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from aic2026.qa_answer import BoDocAnh
from scripts.adaptive_k import AdaptiveKResult
from scripts.qa_v4_pipeline import run_qa_v4


DEFAULT_R4 = Path(r"D:\aic-data\runs\qa_r4_adaptive_candidates.jsonl")
DEFAULT_DEV = Path(r"D:\aic-data\dev\dev_questions.jsonl")
DEFAULT_RUNS = Path(r"D:\aic-data\runs")
DEFAULT_SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    out: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL lỗi tại {path}, dòng {line_number}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"{path}, dòng {line_number} phải là JSON object."
                )

            out.append(obj)

    return out


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_gt_answer(record: dict[str, Any]) -> str | None:
    """
    Cố gắng lấy đáp án GT từ một số schema thường gặp.
    Nếu dev không chứa đáp án, trả None thay vì đoán.
    """
    direct_fields = (
        "dap_an",
        "dap_an_dung",
        "cau_tra_loi",
        "answer",
        "gt_answer",
        "ground_truth_answer",
    )

    for field in direct_fields:
        value = record.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)

    answers = record.get("answers")

    if isinstance(answers, str) and answers.strip():
        return answers.strip()

    if isinstance(answers, list):
        for value in answers:
            if isinstance(value, str) and value.strip():
                return value.strip()

            if isinstance(value, dict):
                for key in ("text", "answer", "dap_an"):
                    x = value.get(key)
                    if isinstance(x, str) and x.strip():
                        return x.strip()

    return None


def _load_qa_dev(
    dev_path: Path,
) -> dict[str, dict[str, Any]]:
    qa: dict[str, dict[str, Any]] = {}

    for record in _read_jsonl(dev_path):
        if record.get("loai_truy_van") != "hoi_dap":
            continue

        query_id = str(record.get("id", "")).strip()
        question = str(record.get("cau_hoi", "")).strip()

        if not query_id or not question:
            continue

        qa[query_id] = {
            "query_id": query_id,
            "question": question,
            "gt_answer": _extract_gt_answer(record),
            "raw": record,
        }

    if not qa:
        raise ValueError(
            f"Không tìm thấy câu hoi_dap nào trong {dev_path}"
        )

    return qa


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


def _consensus_to_dict(consensus: Any) -> dict[str, Any]:
    if is_dataclass(consensus):
        return asdict(consensus)

    if isinstance(consensus, dict):
        return dict(consensus)

    data: dict[str, Any] = {}

    for key in (
        "final_answer",
        "canonical_answer",
        "confidence",
        "support",
        "sources",
        "alternatives",
        "used_fallback",
        "question_type",
    ):
        if hasattr(consensus, key):
            data[key] = getattr(consensus, key)

    if not data:
        data["value"] = str(consensus)

    return data


def _final_answer(consensus: Any) -> str:
    if hasattr(consensus, "final_answer"):
        return str(consensus.final_answer).strip()

    if isinstance(consensus, dict):
        for key in ("final_answer", "answer", "van_ban"):
            value = consensus.get(key)
            if value is not None:
                return str(value).strip()

    return str(consensus).strip()


def _exact_match(prediction: str, gt_answer: str | None) -> bool | None:
    if gt_answer is None:
        return None

    return _normalize_text(prediction) == _normalize_text(gt_answer)


class SemanticMatcher:
    def __init__(
        self,
        model_name: str,
        threshold: float,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.threshold = float(threshold)

    def score(
        self,
        prediction: str,
        gt_answer: str | None,
    ) -> tuple[float | None, bool | None]:
        if gt_answer is None:
            return None, None

        if not prediction.strip() or not gt_answer.strip():
            return 0.0, False

        vectors = self.model.encode(
            [prediction, gt_answer],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        score = float(vectors[0] @ vectors[1])

        return score, score >= self.threshold


def _selected_candidates_for_log(
    result: AdaptiveKResult,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    for rank, candidate in enumerate(
        result.selected_candidates,
        start=1,
    ):
        frame_idx = candidate.get(
            "frame_idx",
            candidate.get("frame_id"),
        )

        selected.append(
            {
                "rank": rank,
                "video_id": candidate.get("video_id"),
                "n": candidate.get("n"),
                "frame_idx": frame_idx,
                "pts_time": candidate.get("pts_time"),
                "score": candidate.get("score"),
            }
        )

    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "QA-V4 batch: QA-R4 -> Reader -> consensus -> "
            "JSONL + summary."
        )
    )

    parser.add_argument(
        "--r4",
        type=Path,
        default=DEFAULT_R4,
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=DEFAULT_DEV,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help=(
            "Có thể truyền nhiều lần. Nếu bỏ trống, chạy toàn bộ QA "
            "có trong QA-R4."
        ),
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
    )
    parser.add_argument(
        "--semantic-model",
        type=str,
        default=DEFAULT_SEMANTIC_MODEL,
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.80,
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    started = datetime.now()
    run_stamp = started.strftime("%Y-%m-%d_%H%M")
    run_dir = args.runs_dir / f"{run_stamp}_QA-V4_batch"
    contracts_dir = run_dir / "contracts"
    report_dir = run_dir / "report"

    contracts_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = contracts_dir / "qa_v4_results.jsonl"
    summary_json = report_dir / "qa_v4_summary.json"

    dev_map = _load_qa_dev(args.dev)
    r4_records = _read_jsonl(args.r4)

    if args.query_id:
        wanted = {str(x).strip() for x in args.query_id}
        r4_records = [
            row
            for row in r4_records
            if str(row.get("query_id", "")).strip() in wanted
        ]

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit phải > 0.")
        r4_records = r4_records[: args.limit]

    if not r4_records:
        raise ValueError("Không có QA-R4 record nào để chạy.")

    use_vlm = not args.no_vlm
    use_semantic = not args.no_semantic

    print("=" * 78)
    print("QA-V4 BATCH — QA-R4 -> READER -> CONSENSUS")
    print("=" * 78)
    print(f"R4       : {args.r4}")
    print(f"DEV      : {args.dev}")
    print(f"Queries  : {len(r4_records)}")
    print(f"VLM      : {use_vlm}")
    print(f"Semantic : {use_semantic}")
    print(f"Run dir  : {run_dir}")
    print()

    reader = None

    if use_vlm:
        print("Đang nạp BLIP Reader...")
        reader = BoDocAnh()
        reader._nap()
        print("BLIP Reader đã nạp.")
        print()

    semantic_matcher = None

    if use_semantic:
        print("Đang nạp semantic matcher...")
        semantic_matcher = SemanticMatcher(
            model_name=args.semantic_model,
            threshold=args.semantic_threshold,
        )
        print("Semantic matcher đã nạp.")
        print()

    results: list[dict[str, Any]] = []

    with output_jsonl.open("w", encoding="utf-8") as out:
        for index, raw_r4 in enumerate(
            r4_records,
            start=1,
        ):
            result = _to_r4_result(raw_r4)
            query_id = str(result.query_id)

            dev = dev_map.get(query_id)

            if dev is None:
                row = {
                    "query_id": query_id,
                    "status": "error",
                    "error": "Không tìm thấy QA tương ứng trong dev.",
                }
                out.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                out.flush()
                results.append(row)
                print(
                    f"[{index:02d}/{len(r4_records):02d}] "
                    f"{query_id}: ERROR — thiếu dev QA"
                )
                continue

            question = dev["question"]
            gt_answer = dev["gt_answer"]

            t0 = time.perf_counter()

            try:
                consensus = run_qa_v4(
                    cau_hoi=question,
                    result=result,
                    bo_doc_anh=reader,
                    dung_vlm=use_vlm,
                )

                latency_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

                prediction = _final_answer(consensus)
                exact = _exact_match(
                    prediction,
                    gt_answer,
                )

                semantic_score = None
                semantic_match = None

                if semantic_matcher is not None:
                    (
                        semantic_score,
                        semantic_match,
                    ) = semantic_matcher.score(
                        prediction,
                        gt_answer,
                    )

                row = {
                    "query_id": query_id,
                    "question": question,
                    "gt_answer": gt_answer,
                    "prediction": prediction,
                    "exact_match": exact,
                    "semantic_score": semantic_score,
                    "semantic_match": semantic_match,
                    "semantic_threshold": (
                        args.semantic_threshold
                        if semantic_matcher is not None
                        else None
                    ),
                    "latency_ms": latency_ms,
                    "r4": {
                        "k_requested": result.k_requested,
                        "reader_k_requested": (
                            result.reader_k_requested
                        ),
                        "k_effective": result.k_effective,
                        "k_available": result.k_available,
                        "status": result.status,
                        "fallback_reason": (
                            result.fallback_reason
                        ),
                    },
                    "selected_candidates": (
                        _selected_candidates_for_log(result)
                    ),
                    "consensus": (
                        _consensus_to_dict(consensus)
                    ),
                    "status": "ok",
                    "error": None,
                }

                pred_show = prediction.replace(
                    "\n",
                    " ",
                )[:70]

                print(
                    f"[{index:02d}/{len(r4_records):02d}] "
                    f"{query_id} | "
                    f"k={result.k_effective} | "
                    f"{latency_ms:.0f} ms | "
                    f"exact={exact} | "
                    f"sem={semantic_match} | "
                    f"{pred_show}"
                )

            except Exception as exc:
                latency_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

                row = {
                    "query_id": query_id,
                    "question": question,
                    "gt_answer": gt_answer,
                    "prediction": None,
                    "exact_match": None,
                    "semantic_score": None,
                    "semantic_match": None,
                    "latency_ms": latency_ms,
                    "r4": {
                        "k_requested": result.k_requested,
                        "reader_k_requested": (
                            result.reader_k_requested
                        ),
                        "k_effective": result.k_effective,
                        "k_available": result.k_available,
                        "status": result.status,
                    },
                    "selected_candidates": (
                        _selected_candidates_for_log(result)
                    ),
                    "consensus": None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

                print(
                    f"[{index:02d}/{len(r4_records):02d}] "
                    f"{query_id} | ERROR | "
                    f"{type(exc).__name__}: {exc}"
                )

            out.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            results.append(row)

    ok_rows = [
        row
        for row in results
        if row.get("status") == "ok"
    ]
    error_rows = [
        row
        for row in results
        if row.get("status") != "ok"
    ]

    exact_rows = [
        row
        for row in ok_rows
        if row.get("exact_match") is not None
    ]
    semantic_rows = [
        row
        for row in ok_rows
        if row.get("semantic_match") is not None
    ]

    exact_hits = sum(
        1
        for row in exact_rows
        if row["exact_match"] is True
    )
    semantic_hits = sum(
        1
        for row in semantic_rows
        if row["semantic_match"] is True
    )

    latencies = [
        float(row["latency_ms"])
        for row in ok_rows
        if row.get("latency_ms") is not None
    ]

    summary = {
        "task": "QA-V4",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "r4_input": str(args.r4),
        "dev_input": str(args.dev),
        "queries_total": len(results),
        "queries_ok": len(ok_rows),
        "queries_error": len(error_rows),
        "queries_with_gt": len(exact_rows),
        "exact_correct": exact_hits,
        "exact_accuracy": (
            exact_hits / len(exact_rows)
            if exact_rows
            else None
        ),
        "semantic_evaluated": len(semantic_rows),
        "semantic_correct": semantic_hits,
        "semantic_accuracy": (
            semantic_hits / len(semantic_rows)
            if semantic_rows
            else None
        ),
        "semantic_threshold": (
            args.semantic_threshold
            if semantic_matcher is not None
            else None
        ),
        "semantic_model": (
            args.semantic_model
            if semantic_matcher is not None
            else None
        ),
        "avg_latency_ms": (
            sum(latencies) / len(latencies)
            if latencies
            else None
        ),
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_json),
    }

    summary_json.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("QA-V4 SUMMARY")
    print("=" * 78)
    print(
        f"Processed        : "
        f"{len(results)}"
    )
    print(
        f"OK / Error       : "
        f"{len(ok_rows)} / {len(error_rows)}"
    )
    print(
        f"GT available     : "
        f"{len(exact_rows)}"
    )

    if exact_rows:
        print(
            f"Exact            : "
            f"{exact_hits}/{len(exact_rows)} "
            f"({100.0 * exact_hits / len(exact_rows):.1f}%)"
        )
    else:
        print("Exact            : N/A — dev không có GT answer")

    if semantic_rows:
        print(
            f"Semantic         : "
            f"{semantic_hits}/{len(semantic_rows)} "
            f"({100.0 * semantic_hits / len(semantic_rows):.1f}%)"
        )
    else:
        print("Semantic         : N/A")

    if latencies:
        print(
            f"Avg latency      : "
            f"{sum(latencies) / len(latencies):.0f} ms/query"
        )

    print(f"Results JSONL    : {output_jsonl}")
    print(f"Summary JSON     : {summary_json}")
    print("=" * 78)

    return 0 if not error_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
