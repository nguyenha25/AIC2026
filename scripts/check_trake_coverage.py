# check_trake_coverage.py — chạy tại repo root: python -m scripts.check_trake_coverage
import json
from aic2026.semantic.parser import RuleBasedParser

parser = RuleBasedParser()
input_path = r"D:\aic-data\dev\dev_questions.jsonl"

total, empty_actions = 0, 0
empty_examples = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("loai_truy_van") != "chuoi_su_kien":
            continue
        events_text = [g["su_kien"] for g in record.get("cac_giai_doan", [])]
        if not events_text:
            continue
        plan = parser.parse_trake(str(record["id"]), events_text)
        for e in plan.events:
            total += 1
            if not e.actions:
                empty_actions += 1
                if len(empty_examples) < 5:
                    empty_examples.append(e.text)

print(f"Tổng số event TRAKE: {total}")
print(f"Số event KHÔNG bóc được action: {empty_actions} ({empty_actions/total*100:.1f}%)")
print("Ví dụ:")
for ex in empty_examples:
    print(" -", ex)