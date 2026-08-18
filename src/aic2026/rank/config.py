"""Cấu hình tham số xếp hạng và khử trùng lặp."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class RankConfig:
    k_rrf: int = 60
    w_visual: float = 1.0
    w_ocr: float = 0.5
    w_asr: float = 0.3
    dedupe_window_seconds: float = 10.0
    top_k_candidates: int = 100

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> RankConfig:
        p = Path(config_path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        rank_sec = data.get("rank", {})
        return cls(
            k_rrf=int(rank_sec.get("k_rrf", 60)),
            w_visual=float(rank_sec.get("w_visual", 1.0)),
            w_ocr=float(rank_sec.get("w_ocr", 0.5)),
            w_asr=float(rank_sec.get("w_asr", 0.3)),
            dedupe_window_seconds=float(rank_sec.get("dedupe_window_seconds", 10.0)),
            top_k_candidates=int(rank_sec.get("top_k_candidates", 100)),
        )