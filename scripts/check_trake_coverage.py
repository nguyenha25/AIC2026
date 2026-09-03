"""Kiểm tra parser có giữ đủ event/action TRAKE trong dev set."""

import json

from aic2026.paths import DEV_DIR
from aic2026.semantic.parser import RuleBasedParser


def main() -> None:
    parser = RuleBasedParser()
    input_path = DEV_DIR / "dev_questions.jsonl"

    total_queries = 0
    source_events = 0
    parsed_events = 0
    empty_actions = 0
    empty_examples = []

    with input_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("loai_truy_van") != "chuoi_su_kien":
                continue

            total_queries += 1
            events_text = [
                str(stage["su_kien"]).strip()
                for stage in record.get("cac_giai_doan", [])
                if isinstance(stage, dict) and str(stage.get("su_kien", "")).strip()
            ]
            source_events += len(events_text)

            plan = parser.parse_trake(str(record["id"]), events_text)
            parsed_events += len(plan.events)

            if len(plan.events) != len(events_text):
                raise ValueError(
                    f"Query {record['id']}: nguồn có {len(events_text)} event, "
                    f"parser trả {len(plan.events)} event."
                )

            for event in plan.events:
                if not event.actions:
                    empty_actions += 1
                    if len(empty_examples) < 10:
                        empty_examples.append((str(record["id"]), event.text))

    action_coverage = (
        (parsed_events - empty_actions) / parsed_events
        if parsed_events
        else 0.0
    )

    print(f"Tổng số query TRAKE          : {total_queries}")
    print(f"Event nguồn / event parser   : {source_events} / {parsed_events}")
    print(f"Event không bóc được action  : {empty_actions}")
    print(f"Action coverage              : {action_coverage:.1%}")
    print("Lưu ý                        : TR-R1 hiện truy vấn bằng event.text;")
    print("                               actions rỗng không trực tiếp làm mất recall.")
    if empty_examples:
        print("Ví dụ:")
        for query_id, text in empty_examples:
            print(f" - query={query_id}: {text}")


if __name__ == "__main__":
    main()
