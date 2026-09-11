"""Ablation nhỏ, chạy trong một process để chọn trọng số sparse TR-R2.

Đây là công cụ tuning offline trên dev GT. Nó không sửa trọng số production;
người chạy chỉ chốt cấu hình sau khi đọc JSON kết quả.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts import benchmark_tr_r2 as benchmark_module


CONFIGS: dict[str, dict[str, float]] = {
    "compat": {
        "region": 0.0,
        "video": 0.0,
        "order": 0.0,
        "span": 0.0,
    },
    "region_only": {
        "region": 0.08,
        "video": 0.0,
        "order": 0.0,
        "span": 0.0,
    },
    "video_only": {
        "region": 0.0,
        "video": 0.04,
        "order": 0.0,
        "span": 0.0,
    },
    "order_only": {
        "region": 0.0,
        "video": 0.0,
        "order": 0.10,
        "span": 0.0,
    },
    "span_only": {
        "region": 0.0,
        "video": 0.0,
        "order": 0.0,
        "span": 0.015,
    },
    "span_calibrated": {
        "region": 0.0,
        "video": 0.0,
        "order": 0.0,
        "span": 0.041,
    },
    "conservative_combo": {
        "region": 0.02,
        "video": 0.01,
        "order": 0.025,
        "span": 0.005,
    },
    "optimized_v1": {
        "region": 0.08,
        "video": 0.04,
        "order": 0.10,
        "span": 0.015,
    },
}


def _apply_config(config: dict[str, float], beam_size: int) -> None:
    benchmark_module.VIDEO_BEAM_SIZE = int(beam_size)
    benchmark_module.REGION_PRIOR_WEIGHT = float(config["region"])
    benchmark_module.VIDEO_PRIOR_WEIGHT = float(config["video"])
    benchmark_module.ORDER_GAIN_WEIGHT = float(config["order"])
    benchmark_module.SPAN_PENALTY_WEIGHT = float(config["span"])


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, int, float]:
    nonzero_weights = sum(
        float(value) != 0.0
        for value in row["weights"].values()
    )
    return (
        float(row["video_accuracy"]),
        float(row["conditional_accuracy_given_beam"]),
        -nonzero_weights,
        -float(row["latency_p50_ms"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation sparse TR-R2: baseline, từng feature và combo nhỏ."
    )
    parser.add_argument("--video-beam-size", type=int, default=12)
    parser.add_argument(
        "--config",
        action="append",
        choices=tuple(CONFIGS),
        help="Chỉ chạy config được chọn; có thể lặp lại. Mặc định chạy tất cả.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=benchmark_module.RUNS_DIR / "tr_r2_sparse_ablation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.video_beam_size <= 0:
        raise ValueError("--video-beam-size phải > 0")

    names = args.config or list(CONFIGS)
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}

    for index, name in enumerate(names, start=1):
        config = CONFIGS[name]
        _apply_config(config, args.video_beam_size)
        print("=" * 72, flush=True)
        print(f"[ABLATION] {index}/{len(names)} config={name}", flush=True)
        print(f"weights={config}", flush=True)

        result = benchmark_module.benchmark_sparse_video_selection()
        summary = result["summary"]
        details[name] = result
        rows.append(
            {
                "name": name,
                "weights": dict(config),
                "beam_recall": float(summary["beam_recall"]),
                "video_accuracy": float(summary["video_accuracy"]),
                "conditional_accuracy_given_beam": float(
                    summary["conditional_accuracy_given_beam"]
                ),
                "video_correct_queries": int(summary["video_correct_queries"]),
                "beam_hits": int(summary["beam_hits"]),
                "num_queries": int(summary["num_queries"]),
                "latency_p50_ms": float(summary["latency_ms"]["p50"]),
                "latency_p95_ms": float(summary["latency_ms"]["p95"]),
            }
        )

    ranked = sorted(rows, key=_ranking_key, reverse=True)
    compat = next((row for row in rows if row["name"] == "compat"), None)
    best = ranked[0]
    improves_compat = bool(
        compat is not None
        and float(best["video_accuracy"]) > float(compat["video_accuracy"])
    )

    report = {
        "task": "TR-R2 sparse scorer ablation",
        "warning": (
            "GT chỉ dùng để tuning offline. Dev có 12 query nên không tự động "
            "đổi production defaults; cần xác nhận trên tập holdout."
        ),
        "video_beam_size": int(args.video_beam_size),
        "ranking_rule": (
            "video_accuracy, conditional_accuracy, ít trọng số khác 0 hơn, "
            "rồi latency_p50"
        ),
        "recommended_config": best["name"],
        "improves_compat_video_accuracy": improves_compat,
        "ranking": ranked,
        "details": details,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("name                 correct  cond_acc  p50_ms")
    for row in ranked:
        print(
            f"{row['name']:<20} "
            f"{row['video_correct_queries']:>2}/{row['num_queries']:<2}   "
            f"{row['conditional_accuracy_given_beam']:.4f}   "
            f"{row['latency_p50_ms']:.2f}"
        )
    print()
    print(f"Recommended (dev): {best['name']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
