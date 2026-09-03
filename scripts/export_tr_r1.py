"""
Export TR-R1 benchmark artifact -> TR-R2 input.

Input:
    D:/aic-data/runs/tr_r1_benchmark.json

Output:
    D:/aic-data/runs/tr_r1_candidates.jsonl

Không chạy model.
Không chạy FAISS.
Không chạy lại TR-R1.

Script chỉ lấy kết quả TR-R1 đã được benchmark và đóng gói
thành 1 JSON object / event để TR-R2 sử dụng.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aic2026.paths import RUNS_DIR


INPUT_PATH = RUNS_DIR / "tr_r1_benchmark.json"
OUTPUT_PATH = RUNS_DIR / "tr_r1_candidates.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy file TR-R1 benchmark: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"File không phải JSON object: {path}"
        )

    return data


def export_candidates(
    benchmark: dict[str, Any],
) -> int:
    queries = benchmark.get("queries")

    if not isinstance(queries, list):
        raise ValueError(
            "benchmark JSON thiếu field 'queries'"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = OUTPUT_PATH.with_suffix(
        OUTPUT_PATH.suffix + ".tmp"
    )

    count = 0

    with tmp_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for query in queries:
            if not isinstance(query, dict):
                raise ValueError(
                    "Một query trong benchmark không phải object"
                )

            query_id = str(
                query.get("query_id", "")
            )

            video_id = query.get(
                "video_id"
            )

            events = query.get(
                "events"
            )

            if not isinstance(events, list):
                raise ValueError(
                    f"Query {query_id!r} thiếu field 'events'"
                )

            for event in events:
                if not isinstance(event, dict):
                    raise ValueError(
                        f"Query {query_id!r}: event không phải object"
                    )

                result = event.get(
                    "result"
                )

                if not isinstance(result, dict):
                    raise ValueError(
                        f"Query {query_id!r}, "
                        f"event {event.get('event_id')!r}: "
                        "thiếu result"
                    )

                # -------------------------------------------------------
                # Đây là artifact dành cho TR-R2.
                #
                # Giữ:
                #   - thông tin event
                #   - GT để diagnostic/benchmark offline
                #   - toàn bộ TR-R1 result
                #
                # result chứa:
                #   video_id
                #   start_time
                #   end_time
                #   score
                #   hits
                #
                # Không sửa / rerank / truncate ở đây.
                # -------------------------------------------------------

                record = {
                    "query_id": query_id,
                    "video_id": video_id,

                    "event_id": event.get(
                        "event_id"
                    ),

                    "event_index": event.get(
                        "event_index"
                    ),

                    "text": event.get(
                        "text"
                    ),

                    "relation": event.get(
                        "relation"
                    ),

                    "entities": event.get(
                        "entities",
                        [],
                    ),

                    "actions": event.get(
                        "actions",
                        [],
                    ),

                    # GT giữ lại để benchmark/debug.
                    # TR-R2 production không cần dùng field này.
                    "gt": event.get(
                        "gt"
                    ),

                    # Diagnostic của TR-R1.
                    "tr_r1_diagnostic": {
                        "overlap": event.get(
                            "overlap"
                        ),
                        "video_match": event.get(
                            "video_match"
                        ),
                        "candidate_frame_hit": event.get(
                            "candidate_frame_hit"
                        ),
                        "best_iou": event.get(
                            "best_iou"
                        ),
                        "best_overlap_rank": event.get(
                            "best_overlap_rank"
                        ),
                    },

                    # ===================================================
                    # OUTPUT CHÍNH CỦA TR-R1
                    # ===================================================
                    "tr_r1": result,
                }

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                count += 1

    tmp_path.replace(
        OUTPUT_PATH
    )

    return count


def main() -> None:
    print("=" * 72)
    print("EXPORT TR-R1 -> TR-R2")
    print("=" * 72)

    print(
        f"INPUT  : {INPUT_PATH}"
    )

    print(
        f"OUTPUT : {OUTPUT_PATH}"
    )

    print()

    benchmark = load_json(
        INPUT_PATH
    )

    count = export_candidates(
        benchmark
    )

    print(
        f"Đã export: {count} events"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print(
        "TR-R1 artifact đã sẵn sàng cho TR-R2."
    )


if __name__ == "__main__":
    main()