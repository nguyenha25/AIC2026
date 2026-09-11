"""Xuất ranked TR-R2/TR-E2 candidates thành CSV TRAKE đúng chuẩn BTC.

Input mặc định là JSON do ``scripts.run_trake_e2`` sinh. Mỗi candidate video
được giữ một dòng tốt nhất; thứ tự candidate trong JSON là thứ tự xếp hạng.
Không dùng GT và không đổi frame_idx sang ``n`` hay timestamp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aic2026.paths import RUNS_DIR, SUBMISSIONS_DIR
from aic2026.submit import Answer, SubmissionBudget, TRAKE, submission_filename


DEFAULT_INPUT = RUNS_DIR / "trake_e2_ranked_results.json"
DEFAULT_OUTPUT_DIR = SUBMISSIONS_DIR / "trake_e2_ranked"


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy kết quả TR-E2: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, list):
        raise ValueError("Kết quả TR-E2 phải là JSON array")

    return [dict(row) for row in value]


def candidate_answer(
    candidate: dict[str, Any],
    *,
    expected_events: int,
) -> Answer | None:
    video_id = str(candidate.get("video_id", "")).strip()
    raw_frame_idx = candidate.get("frame_idx")

    if not video_id or not isinstance(raw_frame_idx, list):
        return None

    frame_idx = [int(value) for value in raw_frame_idx]

    if len(frame_idx) != expected_events:
        return None

    if any(left >= right for left, right in zip(frame_idx, frame_idx[1:])):
        return None

    return Answer(video_id=video_id, frame_ids=frame_idx)


def export_one(
    result: dict[str, Any],
    output_dir: Path,
    *,
    max_answers: int,
) -> tuple[Path, int, int]:
    query_id = str(result.get("query_id", "")).strip()
    events = result.get("events")
    candidates = result.get("ranked_candidates")

    if not query_id:
        raise ValueError("Kết quả thiếu query_id")

    if not isinstance(events, list) or not events:
        raise ValueError(f"Query {query_id}: thiếu events")

    if not isinstance(candidates, list):
        raise ValueError(f"Query {query_id}: thiếu ranked_candidates")

    budget = SubmissionBudget(task=TRAKE, limit=max_answers)
    rejected = 0

    for candidate in candidates:
        answer = candidate_answer(
            dict(candidate),
            expected_events=len(events),
        )

        if answer is None:
            rejected += 1
            continue

        budget.add(answer)

    if not budget.answers:
        raise ValueError(f"Query {query_id}: không có candidate hợp lệ")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / submission_filename(query_id, TRAKE)
    budget.write(path)
    return path, len(budget.answers), rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Xuất ranked TR-E2 candidates thành CSV TRAKE."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-answers", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 1 <= args.max_answers <= 100:
        raise ValueError("--max-answers phải nằm trong [1, 100]")

    results = load_results(args.input)
    total_rows = 0
    total_rejected = 0

    for result in results:
        path, rows, rejected = export_one(
            result,
            args.output_dir,
            max_answers=args.max_answers,
        )
        total_rows += rows
        total_rejected += rejected
        print(f"[OK] {path.name}: {rows} dòng, loại {rejected}")

    print(
        f"DONE: {len(results)} query, {total_rows} dòng, "
        f"loại {total_rejected} candidate không hợp lệ"
    )
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
