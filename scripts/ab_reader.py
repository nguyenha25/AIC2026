"""
QA-V3 — A/B công bằng BLIP vs Qwen trên cùng reader-oracle records.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from src.aic2026.paths import RUNS_DIR

BLIP_GLOB = "*QA-V1_blip_oracle"
QWEN_GLOB = "*QA-V2_qwen_oracle"
BLIP_REL = Path("contracts") / "qa_v1_reader_oracle.jsonl"
QWEN_REL = Path("contracts") / "qa_v2_qwen_oracle.jsonl"

MIN_COMPARABLE_FOR_RECOMMENDATION = 10


def doc_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON lỗi tại {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Record tại {path}:{line_no} không phải object")
            rows.append(obj)
    return rows


def ghi_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ghi_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def tim_run_moi_nhat(runs_dir: Path, pattern: str, rel_file: Path) -> Path:
    candidates = [
        p for p in runs_dir.iterdir()
        if p.is_dir() and p.match(pattern) and (p / rel_file).exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy run khớp {pattern!r} có file {rel_file} trong {runs_dir}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def khoa_record(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("query_id", "")), str(row.get("video_id", ""))


def chi_muc(rows, label):
    out = {}
    for row in rows:
        key = khoa_record(row)
        if not all(key):
            raise ValueError(f"{label}: record thiếu query_id/video_id: {row}")
        if key in out:
            raise ValueError(f"{label}: trùng record cho query/video {key}")
        out[key] = row
    return out


def cung_oracle_frame(a, b):
    return (
        a.get("oracle_frame_idx") == b.get("oracle_frame_idx")
        and a.get("oracle_n") == b.get("oracle_n")
    )


def _bool_int(value):
    if type(value) is not bool:
        raise ValueError(
            f"Giá trị boolean không hợp lệ: {value!r}; yêu cầu bool thật sự."
        )
    return 1 if value else 0


def _require_bool_field(row: dict[str, Any], field: str, label: str) -> bool:
    value = row.get(field)
    if type(value) is not bool:
        raise ValueError(
            f"{label}: {field} phải là bool, nhận {value!r} "
            f"({type(value).__name__})."
        )
    return value


def _valid_latency(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    x = float(value)
    return math.isfinite(x) and x >= 0.0


def _check_same_field(
    blip: dict[str, Any],
    qwen: dict[str, Any],
    field: str,
) -> None:
    if blip.get(field) != qwen.get(field):
        raise ValueError(
            f"BLIP/Qwen khác {field} cho {khoa_record(blip)}: "
            f"{blip.get(field)!r} != {qwen.get(field)!r}"
        )


def _pair_status_ok(row: dict[str, Any]) -> bool:
    return (
        row.get("same_oracle_frame") is True
        and row.get("blip", {}).get("status") == "ok"
        and row.get("qwen", {}).get("status") == "ok"
    )


def so_sanh_chat_luong(blip, qwen):
    b = (
        _bool_int(blip.get("semantic_match")),
        _bool_int(blip.get("exact_match")),
    )
    q = (
        _bool_int(qwen.get("semantic_match")),
        _bool_int(qwen.get("exact_match")),
    )
    if q > b:
        return "qwen"
    if b > q:
        return "blip"
    return "tie"


def ghep_records(blip_rows, qwen_rows):
    bidx = chi_muc(blip_rows, "BLIP")
    qidx = chi_muc(qwen_rows, "Qwen")

    bkeys = set(bidx)
    qkeys = set(qidx)
    common = sorted(bkeys & qkeys)

    aligned = []
    frame_mismatch = 0

    for key in common:
        b = bidx[key]
        q = qidx[key]

        # Fair A/B contract: cùng query/video phải cùng question, GT và image.
        for field in ("question_vi", "gt_answer", "image_path_rel"):
            _check_same_field(b, q, field)

        # Boolean metrics phải là bool thật sự.
        _require_bool_field(b, "exact_match", "BLIP")
        _require_bool_field(b, "semantic_match", "BLIP")
        _require_bool_field(q, "exact_match", "Qwen")
        _require_bool_field(q, "semantic_match", "Qwen")

        same_frame = cung_oracle_frame(b, q)
        if not same_frame:
            frame_mismatch += 1

        b_latency = b.get("latency_ms")
        q_latency = q.get("latency_ms")

        # Chỉ tạo speed ratio nếu cả hai latency đều hợp lệ và BLIP > 0.
        speed_ratio = None
        if _valid_latency(b_latency) and _valid_latency(q_latency):
            if float(b_latency) > 0:
                speed_ratio = float(q_latency) / float(b_latency)

        pair_valid = (
            same_frame
            and b.get("status") == "ok"
            and q.get("status") == "ok"
        )

        aligned.append({
            "schema_version": "1.1",
            "task": "QA-V3",
            "query_id": b.get("query_id"),
            "video_id": b.get("video_id"),
            "question_vi": b.get("question_vi"),
            "gt_answer": b.get("gt_answer"),
            "gt_frame_start": b.get("gt_frame_start"),
            "gt_frame_end": b.get("gt_frame_end"),
            "oracle_frame_idx": b.get("oracle_frame_idx"),
            "oracle_n": b.get("oracle_n"),
            "image_path_rel": b.get("image_path_rel"),
            "same_oracle_frame": same_frame,
            "pair_valid_for_metrics": pair_valid,
            "intent": b.get("intent"),
            "blip": {
                "model": b.get("model"),
                "pred_answer": b.get("pred_answer"),
                "exact_match": b.get("exact_match"),
                "semantic_match": b.get("semantic_match"),
                "latency_ms": b_latency,
                "status": b.get("status"),
            },
            "qwen": {
                "model": q.get("model"),
                "pred_answer": q.get("pred_answer"),
                "exact_match": q.get("exact_match"),
                "semantic_match": q.get("semantic_match"),
                "latency_ms": q_latency,
                "status": q.get("status"),
            },
            "quality_winner": (
                so_sanh_chat_luong(b, q)
                if pair_valid
                else None
            ),
            "qwen_vs_blip_latency_ratio": (
                speed_ratio
                if pair_valid
                else None
            ),
        })

    diagnostics = {
        "blip_records": len(blip_rows),
        "qwen_records": len(qwen_rows),
        "aligned_records": len(aligned),
        "only_blip": [
            {"query_id": k[0], "video_id": k[1]}
            for k in sorted(bkeys - qkeys)
        ],
        "only_qwen": [
            {"query_id": k[0], "video_id": k[1]}
            for k in sorted(qkeys - bkeys)
        ],
        "oracle_frame_mismatch": frame_mismatch,
    }
    return aligned, diagnostics


def percentile(values, p):
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])

    xs = sorted(float(x) for x in values)
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def thong_ke_model(rows, model_key):
    # Nếu row là output A/B đầy đủ thì cả pair phải hợp lệ.
    # Nếu row là fixture/unit-test cũ chỉ có một model, giữ tương thích cũ.
    valid = []
    for r in rows:
        if not r.get("same_oracle_frame"):
            continue

        model = r.get(model_key, {})
        if model.get("status") != "ok":
            continue

        if "blip" in r and "qwen" in r and not _pair_status_ok(r):
            continue

        valid.append(r)

    total = len(valid)

    exact = sum(
        _bool_int(r[model_key].get("exact_match"))
        for r in valid
    )
    semantic = sum(
        _bool_int(r[model_key].get("semantic_match"))
        for r in valid
    )

    latencies = [
        float(r[model_key]["latency_ms"])
        for r in valid
        if _valid_latency(r[model_key].get("latency_ms"))
    ]

    return {
        "n": total,
        "exact_match": exact,
        "exact_rate": exact / total if total else None,
        "semantic_match": semantic,
        "semantic_rate": semantic / total if total else None,
        "avg_latency_ms": mean(latencies) if latencies else None,
        "median_latency_ms": median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "latency_n": len(latencies),
    }


def chon_khuyen_nghi(blip_stats, qwen_stats):
    b_n = int(blip_stats.get("n") or 0)
    q_n = int(qwen_stats.get("n") or 0)

    if min(b_n, q_n) < MIN_COMPARABLE_FOR_RECOMMENDATION:
        return {
            "recommended_reader": None,
            "reason": (
                f"Chưa đủ {MIN_COMPARABLE_FOR_RECOMMENDATION} record hợp lệ "
                f"để kết luận reader (BLIP n={b_n}, Qwen n={q_n})."
            ),
        }

    b_sem = blip_stats.get("semantic_rate")
    q_sem = qwen_stats.get("semantic_rate")
    b_exact = blip_stats.get("exact_rate")
    q_exact = qwen_stats.get("exact_rate")

    if None in (b_sem, q_sem, b_exact, q_exact):
        return {
            "recommended_reader": None,
            "reason": "Không đủ record hợp lệ để kết luận.",
        }

    if q_sem > b_sem:
        return {
            "recommended_reader": "qwen",
            "reason": "Qwen có Semantic Match cao hơn trên cùng oracle holdout.",
        }
    if b_sem > q_sem:
        return {
            "recommended_reader": "blip",
            "reason": "BLIP có Semantic Match cao hơn trên cùng oracle holdout.",
        }

    if q_exact > b_exact:
        return {
            "recommended_reader": "qwen",
            "reason": "Semantic Match hòa, Qwen có Exact Match cao hơn.",
        }
    if b_exact > q_exact:
        return {
            "recommended_reader": "blip",
            "reason": "Semantic Match hòa, BLIP có Exact Match cao hơn.",
        }

    b_lat = blip_stats.get("avg_latency_ms")
    q_lat = qwen_stats.get("avg_latency_ms")
    if _valid_latency(b_lat) and _valid_latency(q_lat):
        winner = "blip" if float(b_lat) <= float(q_lat) else "qwen"
        return {
            "recommended_reader": winner,
            "reason": (
                "Chất lượng hòa; chọn reader có latency trung bình thấp hơn."
            ),
        }

    return {
        "recommended_reader": None,
        "reason": "Chất lượng hòa và thiếu latency hợp lệ để tie-break.",
    }


def tong_hop(aligned, diagnostics):
    # Một record chỉ được dùng cho A/B nếu:
    # - cùng oracle frame
    # - BLIP status=ok
    # - Qwen status=ok
    comparable = [
        r for r in aligned
        if _pair_status_ok(r)
    ]

    blip_stats = thong_ke_model(comparable, "blip")
    qwen_stats = thong_ke_model(comparable, "qwen")

    wins = Counter(
        r.get("quality_winner")
        for r in comparable
        if r.get("quality_winner") in {"blip", "qwen", "tie"}
    )

    ratios = [
        float(r["qwen_vs_blip_latency_ratio"])
        for r in comparable
        if _valid_latency(r.get("qwen_vs_blip_latency_ratio"))
    ]

    missing_side = bool(
        diagnostics.get("only_blip")
        or diagnostics.get("only_qwen")
    )
    frame_mismatch = int(diagnostics.get("oracle_frame_mismatch", 0)) > 0

    recommendation = chon_khuyen_nghi(blip_stats, qwen_stats)

    reasons = []
    if missing_side:
        reasons.append("Hai phía không có cùng đầy đủ record.")
    if frame_mismatch:
        reasons.append("Có record không dùng cùng oracle frame.")
    if len(comparable) < MIN_COMPARABLE_FOR_RECOMMENDATION:
        reasons.append(
            f"Chỉ có {len(comparable)} record hợp lệ, "
            f"nhỏ hơn ngưỡng {MIN_COMPARABLE_FOR_RECOMMENDATION}."
        )

    inconclusive = bool(reasons)

    # Missing record / frame mismatch luôn vô hiệu hóa kết luận model,
    # kể cả nếu phần giao nhau đủ n.
    if missing_side or frame_mismatch:
        recommendation = {
            "recommended_reader": None,
            "reason": " ".join(reasons),
        }

    # Small sample đã được chon_khuyen_nghi() chặn; đảm bảo nhất quán.
    if len(comparable) < MIN_COMPARABLE_FOR_RECOMMENDATION:
        recommendation = {
            "recommended_reader": None,
            "reason": " ".join(reasons),
        }

    return {
        "schema_version": "1.1",
        "task": "QA-V3",
        "comparison": "BLIP vs Qwen on identical oracle frames",
        "diagnostics": diagnostics,
        "comparable_records": len(comparable),
        "conclusion": "inconclusive" if inconclusive else "conclusive",
        "blip": blip_stats,
        "qwen": qwen_stats,
        "quality_wins": {
            "blip": wins.get("blip", 0),
            "qwen": wins.get("qwen", 0),
            "tie": wins.get("tie", 0),
        },
        "latency": {
            "avg_qwen_vs_blip_ratio": mean(ratios) if ratios else None,
            "interpretation": (
                "Ví dụ ratio=100 nghĩa là Qwen chậm hơn BLIP khoảng 100 lần."
            ),
        },
        "recommendation": recommendation,
        "warning": (
            "Cỡ mẫu nhỏ hoặc dữ liệu hai phía chưa đầy đủ; "
            "chỉ dùng descriptive metrics, không kết luận reader."
            if inconclusive
            else None
        ),
    }


def git_commit_ngan():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def tao_run_dir(runs_dir):
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base = runs_dir / f"{stamp}_QA-V3_reader_ab"
    if not base.exists():
        base.mkdir(parents=True)
        return base

    i = 2
    while True:
        candidate = runs_dir / f"{stamp}_QA-V3_reader_ab_{i}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        i += 1


def parse_args():
    p = argparse.ArgumentParser(
        description="QA-V3 — A/B BLIP vs Qwen trên cùng oracle frames"
    )
    p.add_argument("--blip-records", type=Path, default=None)
    p.add_argument("--qwen-records", type=Path, default=None)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    runs_dir = args.runs_dir

    blip_path = args.blip_records
    if blip_path is None:
        blip_path = (
            tim_run_moi_nhat(runs_dir, BLIP_GLOB, BLIP_REL) / BLIP_REL
        )

    qwen_path = args.qwen_records
    if qwen_path is None:
        qwen_path = (
            tim_run_moi_nhat(runs_dir, QWEN_GLOB, QWEN_REL) / QWEN_REL
        )

    print("=" * 88)
    print("QA-V3 — BLIP vs QWEN READER A/B")
    print("=" * 88)
    print(f"BLIP records : {blip_path}")
    print(f"Qwen records : {qwen_path}")

    blip_rows = doc_jsonl(blip_path)
    qwen_rows = doc_jsonl(qwen_path)
    aligned, diagnostics = ghep_records(blip_rows, qwen_rows)

    if not aligned:
        raise RuntimeError("Không có record chung giữa QA-V1 và QA-V2.")

    summary = tong_hop(aligned, diagnostics)

    run_dir = tao_run_dir(runs_dir)
    records_path = run_dir / "contracts" / "reader_ab.jsonl"
    summary_path = run_dir / "report" / "reader_ab_summary.json"
    manifest_path = run_dir / "manifest.json"

    ghi_jsonl(records_path, aligned)
    ghi_json(summary_path, summary)
    ghi_json(
        manifest_path,
        {
            "schema_version": "1.1",
            "task": "QA-V3",
            "run_id": run_dir.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit_ngan(),
            "inputs": {
                "blip_records": str(blip_path),
                "qwen_records": str(qwen_path),
            },
            "records": len(aligned),
            "comparable_records": summary["comparable_records"],
            "conclusion": summary["conclusion"],
            "recommended_reader": summary["recommendation"][
                "recommended_reader"
            ],
            "outputs": {
                "records": str(records_path),
                "summary": str(summary_path),
            },
        },
    )

    b = summary["blip"]
    q = summary["qwen"]
    w = summary["quality_wins"]
    ratio = summary["latency"]["avg_qwen_vs_blip_ratio"]
    rec = summary["recommendation"]

    def pct(x):
        return "-" if x is None else f"{x * 100:.1f}%"

    def num(x):
        return "-" if x is None else f"{x:.0f}"

    print()
    print("-" * 88)
    print(
        f"{'model':<10} {'n':>4} {'exact':>12} {'semantic':>12} "
        f"{'avg ms':>14} {'p95 ms':>14}"
    )
    print("-" * 88)
    print(
        f"{'BLIP':<10} {b['n']:>4} {pct(b['exact_rate']):>12} "
        f"{pct(b['semantic_rate']):>12} {num(b['avg_latency_ms']):>14} "
        f"{num(b['p95_latency_ms']):>14}"
    )
    print(
        f"{'Qwen':<10} {q['n']:>4} {pct(q['exact_rate']):>12} "
        f"{pct(q['semantic_rate']):>12} {num(q['avg_latency_ms']):>14} "
        f"{num(q['p95_latency_ms']):>14}"
    )

    print()
    print(
        f"Quality wins       : BLIP={w['blip']} | "
        f"Qwen={w['qwen']} | tie={w['tie']}"
    )
    if ratio is not None:
        print(f"Qwen/BLIP latency  : {ratio:.1f}x")

    print(f"Kết luận           : {summary['conclusion']}")
    print(f"Khuyến nghị        : {rec['recommended_reader']}")
    print(f"Lý do              : {rec['reason']}")

    if summary.get("warning"):
        print(f"CẢNH BÁO           : {summary['warning']}")

    print()
    print(f"Records            : {records_path}")
    print(f"Summary            : {summary_path}")
    print(f"Manifest           : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
