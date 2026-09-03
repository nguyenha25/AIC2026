from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aic2026.qa_answer import BoDocAnh
from scripts.adaptive_k import AdaptiveKResult
from scripts.qa_v4_pipeline import run_qa_v4


DEFAULT_R4 = Path(r"D:\aic-data\runs\qa_r4_adaptive_candidates.jsonl")
DEFAULT_DEV = Path(r"D:\aic-data\dev\dev_questions.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL lỗi tại {path}, dòng {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}, dòng {line_number} phải là JSON object."
                )

            records.append(record)

    if not records:
        raise ValueError(f"File không có record nào: {path}")

    return records


def _to_r4_result(record: dict[str, Any]) -> AdaptiveKResult:
    required = (
        "query_id",
        "k_requested",
        "reader_k_requested",
        "k_effective",
        "k_available",
        "selected_candidates",
        "status",
    )

    missing = [key for key in required if key not in record]
    if missing:
        raise KeyError(
            "R4 record thiếu field: " + ", ".join(missing)
        )

    selected = record["selected_candidates"]

    if not isinstance(selected, list):
        raise TypeError("selected_candidates phải là JSON array.")

    return AdaptiveKResult(
        query_id=str(record["query_id"]),
        k_requested=int(record["k_requested"]),
        reader_k_requested=int(record["reader_k_requested"]),
        k_effective=int(record["k_effective"]),
        k_available=int(record["k_available"]),
        selected_candidates=tuple(selected),
        status=str(record["status"]),
        fallback_reason=record.get("fallback_reason"),
    )


def _load_question_map(dev_path: Path) -> dict[str, str]:
    records = _read_jsonl(dev_path)

    result: dict[str, str] = {}

    for record in records:
        if record.get("loai_truy_van") != "hoi_dap":
            continue

        query_id = str(record.get("id", "")).strip()
        question = str(record.get("cau_hoi", "")).strip()

        if query_id and question:
            result[query_id] = question

    if not result:
        raise ValueError(
            f"Không tìm thấy câu hoi_dap nào trong: {dev_path}"
        )

    return result


def _pick_r4_record(
    records: list[dict[str, Any]],
    query_id: str | None,
) -> dict[str, Any]:
    if query_id is None:
        return records[0]

    wanted = str(query_id).strip()

    for record in records:
        if str(record.get("query_id", "")).strip() == wanted:
            return record

    raise KeyError(
        f"Không tìm thấy query_id={wanted!r} trong R4 output."
    )


def _final_answer_text(consensus: Any) -> str:
    for field in (
        "final_answer",
        "van_ban",
        "answer",
    ):
        value = getattr(consensus, field, None)
        if value is not None:
            text = str(value).strip()
            if text:
                return text

    if isinstance(consensus, dict):
        for field in (
            "final_answer",
            "van_ban",
            "answer",
        ):
            value = consensus.get(field)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text

    return str(consensus).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test QA-V4: đọc đúng output QA-R4 rồi chạy "
            "Reader + consensus cho một câu QA."
        )
    )

    parser.add_argument(
        "--r4",
        type=Path,
        default=DEFAULT_R4,
        help=f"QA-R4 JSONL (default: {DEFAULT_R4})",
    )

    parser.add_argument(
        "--dev",
        type=Path,
        default=DEFAULT_DEV,
        help=f"Dev questions JSONL (default: {DEFAULT_DEV})",
    )

    parser.add_argument(
        "--query-id",
        type=str,
        default=None,
        help=(
            "Query QA cần chạy. Nếu bỏ trống, dùng record đầu tiên "
            "trong QA-R4 output."
        ),
    )

    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Không gọi BLIP; chỉ dùng fallback/non-VLM reader.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    print("=" * 72)
    print("QA-V4 SMOKE — QA-R4 -> READER -> CONSENSUS")
    print("=" * 72)
    print(f"R4  : {args.r4}")
    print(f"DEV : {args.dev}")
    print()

    r4_records = _read_jsonl(args.r4)
    question_map = _load_question_map(args.dev)

    raw_r4 = _pick_r4_record(
        r4_records,
        args.query_id,
    )
    r4_result = _to_r4_result(raw_r4)

    query_id = str(r4_result.query_id)

    if query_id not in question_map:
        raise KeyError(
            f"DEV không có câu hoi_dap query_id={query_id!r}."
        )

    question = question_map[query_id]

    print(f"Query ID           : {query_id}")
    print(f"Question           : {question}")
    print(f"Semantic K         : {r4_result.k_requested}")
    print(f"Reader requested K : {r4_result.reader_k_requested}")
    print(f"Effective K        : {r4_result.k_effective}")
    print(f"Available K        : {r4_result.k_available}")
    print(f"R4 status          : {r4_result.status}")
    print()

    print("Selected frames:")
    for rank, candidate in enumerate(
        r4_result.selected_candidates,
        start=1,
    ):
        print(
            f"  #{rank:02d} "
            f"video={candidate.get('video_id')} "
            f"n={candidate.get('n')} "
            f"frame_idx={candidate.get('frame_idx')} "
            f"pts_time={candidate.get('pts_time')} "
            f"score={candidate.get('score')}"
        )

    print()

    use_vlm = not args.no_vlm

    reader = None
    if use_vlm:
        print("Đang nạp BLIP Reader...")
        reader = BoDocAnh()
        # Nạp chủ động để lỗi model/runtime xuất hiện trước khi chạy pipeline.
        reader._nap()
        print("BLIP Reader đã nạp.")
        print()

    consensus = run_qa_v4(
        cau_hoi=question,
        result=r4_result,
        bo_doc_anh=reader,
        dung_vlm=use_vlm,
    )

    final_answer = _final_answer_text(consensus)

    print("=" * 72)
    print("QA-V4 RESULT")
    print("=" * 72)
    print(f"Query ID     : {query_id}")
    print(f"Final answer : {final_answer}")
    print()
    print("Consensus object:")
    print(consensus)
    print("=" * 72)

    if not final_answer:
        raise RuntimeError("QA-V4 trả final answer rỗng.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
