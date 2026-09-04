"""
scripts/export_tr_r1_candidates.py
=====================================

Chuyển `tr_r1_benchmark.json` (output của `scripts/benchmark_tr_r1.py`)
thành `tr_r1_candidates.jsonl` — đúng contract PHẲNG (một dòng JSON /
event) mà `scripts/benchmark_tr_r2.py` cần làm input.

Vì sao cần file này
--------------------
`benchmark_tr_r2.py` đọc:

    INPUT_PATH = RUNS_DIR / "tr_r1_candidates.jsonl"

nhưng `benchmark_tr_r1.py` chỉ ghi ra `tr_r1_benchmark.json` với schema
LỒNG NHAU khác hẳn:

    { "queries": [ { "query_id", "events": [ { "event_id", "gt",
                                                 "result", ... } ] } ] }

Không có script nào nối 2 bước này trước đây — TR-R2 benchmark hiện
KHÔNG THỂ chạy vì thiếu input. Script này lấp đúng khoảng trống đó.

Không cần transform gì phức tạp: mỗi `event["result"]` trong
tr_r1_benchmark.json đã CHÍNH XÁC là output của `trr1_result_to_dict()`
(event_id/text/relation/regions) — khớp thẳng với sub-schema `tr_r1`
mà `result_from_record()` bên benchmark_tr_r2.py cần. Tương tự,
`event["gt"]` đã có đủ video_id/frame_start/frame_end/fps/pts_time/
start_time/end_time — khớp thẳng schema `gt` mà
`get_gt_events_from_records()` cần. Script chỉ FLATTEN, không tính
toán lại gì.

Chạy:
    python -m scripts.export_tr_r1_candidates

Hoặc chỉ định path khác:
    python -m scripts.export_tr_r1_candidates --benchmark path.json --output path.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aic2026.paths import RUNS_DIR

DEFAULT_BENCHMARK_PATH = RUNS_DIR / "tr_r1_benchmark.json"
DEFAULT_OUTPUT_PATH = RUNS_DIR / "tr_r1_candidates.jsonl"


def export(
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> int:
    """
    Đọc tr_r1_benchmark.json, ghi tr_r1_candidates.jsonl (atomic write
    qua .tmp + replace, cùng convention với benchmark_tr_r1.py).

    Trả về số record đã ghi.

    Raises:
        FileNotFoundError nếu chưa chạy benchmark_tr_r1.py.
        ValueError nếu schema tr_r1_benchmark.json không đúng kỳ vọng
        (fail loud thay vì ghi ra file rỗng/thiếu field im lặng).
    """

    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Chưa có {benchmark_path}. "
            "Chạy scripts/benchmark_tr_r1.py trước."
        )

    with benchmark_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data.get("queries")

    if not isinstance(queries, list):
        raise ValueError(
            f"{benchmark_path}: thiếu 'queries' hoặc sai kiểu "
            f"(got {type(queries).__name__})."
        )

    if not queries:
        raise ValueError(
            f"{benchmark_path}: 'queries' rỗng — không có gì để export."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    record_count = 0
    query_count = 0

    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as out:

            for query in queries:

                query_id = query.get("query_id")

                if not query_id:
                    raise ValueError(
                        "Một query trong tr_r1_benchmark.json thiếu "
                        "'query_id'."
                    )

                events = query.get("events")

                if not isinstance(events, list):
                    raise ValueError(
                        f"Query {query_id!r} thiếu 'events' hoặc sai kiểu."
                    )

                if not events:
                    raise ValueError(
                        f"Query {query_id!r} có 'events' rỗng."
                    )

                query_count += 1

                for event in events:

                    required = {
                        "event_id",
                        "event_index",
                        "text",
                        "gt",
                        "result",
                    }

                    missing = required - set(event)

                    if missing:
                        raise ValueError(
                            f"Query {query_id!r}, event "
                            f"{event.get('event_id')!r}: thiếu field "
                            + ", ".join(sorted(missing))
                        )

                    record: dict[str, Any] = {
                        "query_id": query_id,
                        "event_id": event["event_id"],
                        "event_index": event["event_index"],
                        "text": event["text"],
                        "relation": event.get("relation"),
                        # event["result"] đã đúng schema
                        # trr1_result_to_dict() — khớp thẳng "tr_r1".
                        "tr_r1": event["result"],
                        # event["gt"] đã có đủ field cần cho evaluation.
                        "gt": event["gt"],
                    }

                    out.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

                    record_count += 1

    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    tmp_path.replace(output_path)

    print(
        f"Đã export {record_count} event record từ {query_count} query "
        f"-> {output_path}"
    )

    return record_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export tr_r1_benchmark.json -> tr_r1_candidates.jsonl "
            "(input contract của benchmark_tr_r2.py)."
        )
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help=f"Input (default: {DEFAULT_BENCHMARK_PATH})",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output (default: {DEFAULT_OUTPUT_PATH})",
    )

    args = parser.parse_args()

    export(benchmark_path=args.benchmark, output_path=args.output)


if __name__ == "__main__":
    main()