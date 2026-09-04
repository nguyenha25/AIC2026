"""
EVAL-02 — Retrieval / Adaptive-K / TRAKE accuracy-latency Pareto
=================================================================

Mục tiêu
--------
Ablation đúng một biến/lần cho:

* Q&A reader budget: fixed K={4,8,12} vs QA-R4 adaptive K.
* TRAKE: sparse coarse hit (T1) vs dense-local + DP của TR-R2 (T2).
* Tổng hợp Recall@50/100/300/500 từ N-02 cùng latency breakdown từ QA-R1.
* Tìm Pareto front accuracy-latency; không tự sửa retrieval/reader.

Script này cố ý không import FAISS/Qwen/TR-R2 ở module import time. Nhờ vậy unit
-test chạy data-free. Chỉ khi bật --run-trake mới import pipeline TRAKE thật.

Các input runtime chính
-----------------------
--dev            dev_holdout/baseline_clean JSONL có GT.
--s4             QueryPlan/S4 JSONL có semantic_k_hint.
--r3             QA-R3 JSON/JSONL có candidates (Top-50 coverage/diversity).
--qa-r1-report   QA-R1 manifest có query_records[*].latency_ms.
--n02-report     N-02 profile có Recall@50/100/300/500.
--vlm-latency    tùy chọn JSONL: query_id, config_id, latency_ms.
--qa-score-json  tùy chọn JSON map config_id -> QA score end-to-end.
--run-trake      chạy TR-R1 -> TR-R2 thật trên các câu TRAKE trong --dev.

Output
------
DATA_ROOT/runs/eval_02_<timestamp>/
    manifest.json
    qa_ablation.json
    trake_ablation.json
    pareto.json
    REPORT_EVAL_02.md
"""

from __future__ import annotations

import argparse
import bisect
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FIXED_READER_K = (4, 8, 12)
DEFAULT_ADAPTIVE_POLICY = {100: 4, 300: 8, 500: 12}
RETRIEVAL_KS = (50, 100, 300, 500)


# ---------------------------------------------------------------------------
# Generic IO / statistics
# ---------------------------------------------------------------------------


def _qid(record: Mapping[str, Any]) -> str:
    value = record.get("query_id", record.get("id"))
    if value is None:
        raise KeyError("record thiếu query_id/id")
    value = str(value)
    if not value:
        raise ValueError("query_id/id không được rỗng")
    return value


def load_records(path: Path | str) -> list[dict[str, Any]]:
    """Đọc JSON array/object hoặc JSONL thành list[dict]."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_no, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL lỗi dòng {line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"dòng {line_no} phải là JSON object")
            records.append(item)
        return records

    if isinstance(document, list):
        if not all(isinstance(x, dict) for x in document):
            raise ValueError("JSON array phải chỉ chứa object")
        return list(document)

    if isinstance(document, dict):
        # Các manifest hiện tại thường bọc records trong query_records.
        if isinstance(document.get("query_records"), list):
            rows = document["query_records"]
            if not all(isinstance(x, dict) for x in rows):
                raise ValueError("query_records phải chỉ chứa object")
            return list(rows)
        return [document]

    raise ValueError("input phải là JSON object/array hoặc JSONL")


def percentile(values: Sequence[float], q: float) -> float | None:
    """Percentile tuyến tính tương thích cách hiểu p50/p95 thông thường."""
    if not values:
        return None
    if not 0 <= q <= 100:
        raise ValueError("q phải nằm trong [0,100]")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    xs = [float(v) for v in values]
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs) if xs else None,
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "max": max(xs) if xs else None,
    }


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Q&A — fixed K vs adaptive K trên CÙNG R3 pool
# ---------------------------------------------------------------------------


def candidate_frame_idx(candidate: Mapping[str, Any]) -> int | None:
    value = candidate.get("frame_idx", candidate.get("frame_id"))
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def qa_candidate_matches_gt(candidate: Mapping[str, Any], gt: Mapping[str, Any]) -> bool:
    if str(candidate.get("video_id", "")) != str(gt.get("video_id", "")):
        return False
    frame_idx = candidate_frame_idx(candidate)
    if frame_idx is None:
        return False
    try:
        start = int(gt["frame_start"])
        end = int(gt["frame_end"])
    except (KeyError, TypeError, ValueError):
        return False
    return start <= frame_idx <= end


def first_qa_gt_rank(
    candidates: Sequence[Mapping[str, Any]], gt: Mapping[str, Any]
) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if qa_candidate_matches_gt(candidate, gt):
            return rank
    return None


def _adaptive_k(semantic_k_hint: Any, policy: Mapping[int, int]) -> int:
    if isinstance(semantic_k_hint, bool):
        raise ValueError("semantic_k_hint không được là bool")
    try:
        semantic_k = int(semantic_k_hint)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"semantic_k_hint không hợp lệ: {semantic_k_hint!r}") from exc
    if semantic_k not in policy:
        raise ValueError(
            f"semantic_k_hint={semantic_k} không có trong policy {sorted(policy)}"
        )
    reader_k = int(policy[semantic_k])
    if reader_k <= 0:
        raise ValueError("reader_k phải > 0")
    return reader_k


def _qa_indices(
    gt_records: Iterable[Mapping[str, Any]],
    r3_records: Iterable[Mapping[str, Any]],
    s4_records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    gt: dict[str, dict[str, Any]] = {}
    for row in gt_records:
        loai = row.get("loai_truy_van")
        task = row.get("task")
        if loai is not None:
            if loai != "hoi_dap":
                continue
        elif task is not None and task != "qa":
            continue
        gt[_qid(row)] = dict(row)

    r3: dict[str, dict[str, Any]] = {_qid(row): dict(row) for row in r3_records}
    s4: dict[str, dict[str, Any]] = {}
    for row in s4_records:
        if row.get("task") not in (None, "qa"):
            continue
        s4[_qid(row)] = dict(row)
    return gt, r3, s4


def load_vlm_latency(path: Path | str | None) -> dict[tuple[str, str], float]:
    """JSONL tùy chọn: {query_id, config_id, latency_ms}."""
    if path is None:
        return {}
    result: dict[tuple[str, str], float] = {}
    for row in load_records(path):
        qid = _qid(row)
        config_id = str(row.get("config_id", ""))
        if not config_id:
            raise ValueError("VLM latency record thiếu config_id")
        value = row.get("latency_ms")
        if value is None:
            raise ValueError("VLM latency record thiếu latency_ms")
        result[(qid, config_id)] = float(value)
    return result


def load_qa_r1_per_query_latency(path: Path | str | None) -> dict[str, dict[str, float]]:
    """
    Đọc QA-R1 manifest. Trả latency theo query.

    Scope mapping:
      faiss_ms  = source_retrieval
      rerank_ms = video_fusion + top300 + top50

    Đây là scope minh bạch theo field thực tế của QA-R1; không gọi top12 là VLM.
    """
    if path is None:
        return {}
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = doc.get("query_records", [])
    if not isinstance(rows, list):
        raise ValueError("QA-R1 report thiếu query_records list")
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        qid = _qid(row)
        lat = row.get("latency_ms", {})
        if not isinstance(lat, Mapping):
            continue
        faiss_ms = float(lat.get("source_retrieval", 0.0))
        rerank_ms = sum(
            float(lat.get(name, 0.0))
            for name in ("video_fusion", "top300", "top50")
        )
        result[qid] = {
            "faiss_ms": faiss_ms,
            "rerank_ms": rerank_ms,
            "r1_total_ms": float(lat.get("total", faiss_ms + rerank_ms)),
        }
    return result


def evaluate_qa_reader_k(
    *,
    gt_records: Iterable[Mapping[str, Any]],
    r3_records: Iterable[Mapping[str, Any]],
    s4_records: Iterable[Mapping[str, Any]],
    fixed_k: Sequence[int] = FIXED_READER_K,
    adaptive_policy: Mapping[int, int] = DEFAULT_ADAPTIVE_POLICY,
    r1_latency_by_query: Mapping[str, Mapping[str, float]] | None = None,
    vlm_latency_by_query_config: Mapping[tuple[str, str], float] | None = None,
    vlm_ms_per_frame: float | None = None,
) -> dict[str, Any]:
    """Ablation reader budget trên chính cùng một R3 candidate pool."""
    gt, r3, s4 = _qa_indices(gt_records, r3_records, s4_records)
    common = [qid for qid in gt if qid in r3 and qid in s4]
    if not common:
        raise ValueError("không có QA query chung giữa GT, R3 và S4")

    r1_latency_by_query = r1_latency_by_query or {}
    vlm_latency_by_query_config = vlm_latency_by_query_config or {}

    configs: list[tuple[str, Any]] = [(f"fixed_{int(k)}", int(k)) for k in fixed_k]
    configs.append(("adaptive", None))

    output: dict[str, Any] = {
        "n_queries": len(common),
        "query_ids": common,
        "same_candidate_pool": True,
        "configs": {},
    }

    for config_id, k_fixed in configs:
        hits = 0
        rr_sum = 0.0
        k_effective_values: list[int] = []
        selector_ms: list[float] = []
        faiss_ms: list[float] = []
        rerank_ms: list[float] = []
        vlm_ms: list[float] = []
        total_ms: list[float] = []
        per_query: list[dict[str, Any]] = []

        for qid in common:
            row_r3 = r3[qid]
            candidates = row_r3.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError(f"R3 query {qid} candidates phải là list")

            if config_id == "adaptive":
                if "semantic_k_hint" not in s4[qid]:
                    raise KeyError(f"S4 query {qid} thiếu semantic_k_hint")
                k_requested = _adaptive_k(s4[qid]["semantic_k_hint"], adaptive_policy)
            else:
                k_requested = int(k_fixed)

            select_start = time.perf_counter()
            selected = candidates[:k_requested]
            select_elapsed = (time.perf_counter() - select_start) * 1000.0

            k_eff = len(selected)
            rank = first_qa_gt_rank(selected, gt[qid])
            if rank is not None:
                hits += 1
                rr_sum += 1.0 / rank

            k_effective_values.append(k_eff)
            selector_ms.append(select_elapsed)

            base = r1_latency_by_query.get(qid, {})
            q_faiss = base.get("faiss_ms")
            q_rerank = base.get("rerank_ms")
            if q_faiss is not None:
                faiss_ms.append(float(q_faiss))
            if q_rerank is not None:
                rerank_ms.append(float(q_rerank))

            q_vlm = vlm_latency_by_query_config.get((qid, config_id))
            vlm_source = "measured"
            if q_vlm is None and vlm_ms_per_frame is not None:
                q_vlm = float(vlm_ms_per_frame) * k_eff
                vlm_source = "estimated_linear_per_frame"
            if q_vlm is not None:
                vlm_ms.append(float(q_vlm))

            pieces = [select_elapsed]
            if q_faiss is not None:
                pieces.append(float(q_faiss))
            if q_rerank is not None:
                pieces.append(float(q_rerank))
            if q_vlm is not None:
                pieces.append(float(q_vlm))
            # Total chỉ có nghĩa là đầy đủ khi có retrieval + VLM.
            q_total = sum(pieces) if (q_faiss is not None and q_rerank is not None and q_vlm is not None) else None
            if q_total is not None:
                total_ms.append(q_total)

            per_query.append(
                {
                    "query_id": qid,
                    "k_requested": k_requested,
                    "k_effective": k_eff,
                    "gt_rank_in_selected": rank,
                    "hit": rank is not None,
                    "selector_ms": select_elapsed,
                    "faiss_ms": q_faiss,
                    "rerank_ms": q_rerank,
                    "vlm_ms": q_vlm,
                    "vlm_latency_source": vlm_source if q_vlm is not None else None,
                    "total_ms": q_total,
                }
            )

        n = len(common)
        output["configs"][config_id] = {
            "config_id": config_id,
            "hits": hits,
            "reader_recall": hits / n,
            "mrr": rr_sum / n,
            "reader_k": {
                "min": min(k_effective_values),
                "max": max(k_effective_values),
                "mean": statistics.fmean(k_effective_values),
                "total_frames": sum(k_effective_values),
            },
            "latency_ms": {
                "faiss": summarize(faiss_ms),
                "rerank": summarize(rerank_ms),
                "selector": summarize(selector_ms),
                "vlm": summarize(vlm_ms),
                "total": summarize(total_ms),
                "total_complete": len(total_ms) == n,
            },
            "per_query": per_query,
        }

    return output


# ---------------------------------------------------------------------------
# Retrieval profile (N-02) — Recall@50/100/300/500
# ---------------------------------------------------------------------------


def extract_n02_retrieval_profile(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = doc.get("metrics", {})
    recalls = metrics.get("recalls", {})
    out_recalls: dict[str, float | None] = {}
    for k in RETRIEVAL_KS:
        value = recalls.get(f"Recall@{k}")
        out_recalls[str(k)] = float(value) if value is not None else None
    latency = metrics.get("latency_ms", {})
    return {
        "source_task": doc.get("task"),
        "retrieval_method": doc.get("retrieval_method"),
        "num_queries": doc.get("dataset_info", {}).get("num_queries"),
        "recall_at_k": out_recalls,
        "latency_ms": {
            "p50": latency.get("p50"),
            "p95": latency.get("p95"),
        },
        "locked_config": doc.get("locked_config"),
        "status": doc.get("status"),
    }


# ---------------------------------------------------------------------------
# TRAKE metrics
# ---------------------------------------------------------------------------


def _distance_to_interval(frame_idx: int, start: int, end: int) -> int:
    if start <= frame_idx <= end:
        return 0
    if frame_idx < start:
        return start - frame_idx
    return frame_idx - end


def score_trake_prediction(
    *,
    predicted_video_id: str,
    predicted_frame_ids: Sequence[int],
    gt: Mapping[str, Any],
) -> dict[str, Any]:
    events = gt.get("cac_giai_doan")
    if not isinstance(events, list) or not events:
        raise ValueError("GT TRAKE thiếu cac_giai_doan")

    n_events = len(events)
    if len(predicted_frame_ids) != n_events:
        raise ValueError(
            f"prediction có {len(predicted_frame_ids)} frame nhưng GT có {n_events} event"
        )

    video_correct = str(predicted_video_id) == str(gt.get("video_id"))
    if not video_correct:
        return {
            "video_correct": False,
            "event_rscore": 0.0,
            "correct_events": 0,
            "total_events": n_events,
            "frame_errors": [None] * n_events,
        }

    correct = 0
    errors: list[int] = []
    for frame_idx, event in zip(predicted_frame_ids, events):
        start = int(event["frame_start"])
        end = int(event["frame_end"])
        frame_idx = int(frame_idx)
        dist = _distance_to_interval(frame_idx, start, end)
        errors.append(dist)
        if dist == 0:
            correct += 1

    return {
        "video_correct": True,
        "event_rscore": correct / n_events,
        "correct_events": correct,
        "total_events": n_events,
        "frame_errors": errors,
    }


def aggregate_trake(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_queries": 0,
            "video_recall": None,
            "mean_event_rscore": None,
            "correct_events": 0,
            "total_events": 0,
            "mean_frame_error": None,
            "latency_ms": summarize([]),
        }

    n = len(rows)
    correct_video = sum(1 for row in rows if row.get("video_correct"))
    rscores = [float(row["event_rscore"]) for row in rows]
    correct_events = sum(int(row["correct_events"]) for row in rows)
    total_events = sum(int(row["total_events"]) for row in rows)
    frame_errors = [
        int(error)
        for row in rows
        for error in row.get("frame_errors", [])
        if error is not None
    ]
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    return {
        "n_queries": n,
        "video_recall": correct_video / n,
        "mean_event_rscore": statistics.fmean(rscores),
        "correct_events": correct_events,
        "total_events": total_events,
        "mean_frame_error": statistics.fmean(frame_errors) if frame_errors else None,
        "latency_ms": summarize(latencies),
    }


@dataclass
class _FrameLookup:
    by_video: dict[str, tuple[list[float], list[int]]]

    def nearest_frame(self, video_id: str, pts_time: float) -> int:
        if video_id not in self.by_video:
            raise KeyError(f"frame_map không có video {video_id}")
        times, frame_ids = self.by_video[video_id]
        pos = bisect.bisect_left(times, float(pts_time))
        if pos <= 0:
            return frame_ids[0]
        if pos >= len(times):
            return frame_ids[-1]
        before = pos - 1
        after = pos
        if abs(times[before] - pts_time) <= abs(times[after] - pts_time):
            return frame_ids[before]
        return frame_ids[after]


def _load_frame_lookup(frame_map_path: Path | str) -> _FrameLookup:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("--run-trake cần pandas để đọc frame_map.parquet") from exc

    df = pd.read_parquet(frame_map_path, columns=["video_id", "pts_time", "frame_idx"])
    by_video: dict[str, tuple[list[float], list[int]]] = {}
    for video_id, group in df.groupby("video_id", sort=False):
        g = group.sort_values("pts_time")
        by_video[str(video_id)] = (
            [float(x) for x in g["pts_time"].tolist()],
            [int(x) for x in g["frame_idx"].tolist()],
        )
    return _FrameLookup(by_video=by_video)


def _trake_events(gt: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = gt.get("cac_giai_doan")
    if not isinstance(raw, list) or not raw:
        raise ValueError("TRAKE query thiếu cac_giai_doan")
    events = []
    for i, event in enumerate(raw, start=1):
        text = event.get("su_kien", event.get("text"))
        if not text:
            raise ValueError(f"TRAKE event {i} thiếu su_kien/text")
        events.append(
            {
                "event_id": f"E{i}",
                "text": str(text),
                "relation": "start" if i == 1 else f"after:E{i-1}",
            }
        )
    return events


def _events_regions_dict(tr_r1_results: Sequence[Any]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for result in tr_r1_results:
        output[str(result.event_id)] = [
            {
                "video_id": region.video_id,
                "start_time": float(region.start_time),
                "end_time": float(region.end_time),
            }
            for region in result.regions
        ]
    return output

def _choose_sparse_video(
    events_regions: Mapping[str, Sequence[Mapping[str, Any]]],
    choose_rrf: Any,
) -> tuple[str, dict[str, Any]]:
    """
    Ưu tiên video có region ở TẤT CẢ event.

    Nếu tồn tại intersection, vẫn dùng chính chon_video_rrf của TR-R2,
    nhưng chỉ trên các video full-coverage.

    Nếu không có intersection, trả video RRF gốc + metadata partial;
    caller quyết định không chấm T1 thay vì bịa frame cho event bị thiếu.
    """
    if not events_regions:
        raise ValueError("events_regions rỗng")

    video_sets: dict[str, set[str]] = {}

    for event_id, regions in events_regions.items():
        videos = {
            str(region["video_id"])
            for region in regions
            if region.get("video_id") is not None
        }
        video_sets[str(event_id)] = videos

    if not video_sets:
        raise ValueError("không có event region")

    sets = list(video_sets.values())
    common_videos = set.intersection(*sets) if sets else set()

    if common_videos:
        filtered = {
            event_id: [
                region
                for region in regions
                if str(region["video_id"]) in common_videos
            ]
            for event_id, regions in events_regions.items()
        }

        video_id = str(choose_rrf(filtered))

        return video_id, {
            "full_event_coverage": True,
            "common_video_count": len(common_videos),
            "common_videos": sorted(common_videos),
            "covered_events": sorted(events_regions),
            "missing_events": [],
        }

    # Diagnostic fallback: vẫn xem RRF gốc muốn chọn video nào.
    video_id = str(choose_rrf(events_regions))

    covered_events = []
    missing_events = []

    for event_id, regions in events_regions.items():
        found = any(
            str(region.get("video_id")) == video_id
            for region in regions
        )
        if found:
            covered_events.append(str(event_id))
        else:
            missing_events.append(str(event_id))

    return video_id, {
        "full_event_coverage": False,
        "common_video_count": 0,
        "common_videos": [],
        "covered_events": covered_events,
        "missing_events": missing_events,
    }


def _sparse_frames(
    tr_r1_results: Sequence[Any], video_id: str, frame_lookup: _FrameLookup
) -> list[int]:
    frame_ids: list[int] = []
    for result in tr_r1_results:
        matching = [r for r in result.regions if str(r.video_id) == str(video_id)]
        if not matching:
            raise ValueError(f"event {result.event_id} không có region cho video đã chọn {video_id}")
        region = matching[0]
        hits = list(region.hits)
        if not hits:
            midpoint = (float(region.start_time) + float(region.end_time)) / 2.0
            frame_ids.append(frame_lookup.nearest_frame(video_id, midpoint))
            continue
        hit = max(hits, key=lambda h: float(h.get("score", 0.0)))
        if hit.get("frame_idx") is not None:
            frame_ids.append(int(hit["frame_idx"]))
        elif hit.get("pts_time") is not None:
            frame_ids.append(frame_lookup.nearest_frame(video_id, float(hit["pts_time"])))
        else:
            raise ValueError("TR-R1 hit không có frame_idx/pts_time")
    return frame_ids


def _dense_frames(result: Mapping[str, Any], frame_lookup: _FrameLookup, event_ids: Sequence[str]) -> list[int]:
    # Ưu tiên frame_idx nếu TR-R2 phiên bản hiện tại đã export.
    for key in ("chosen_frame_idxs", "chosen_frame_ids"):
        value = result.get(key)
        if isinstance(value, Mapping):
            return [int(value[event_id]) for event_id in event_ids]
        if isinstance(value, list):
            return [int(x) for x in value]

    value = result.get("chosen_frames")
    if isinstance(value, Mapping):
        out = []
        for event_id in event_ids:
            item = value[event_id]
            if isinstance(item, Mapping):
                frame = item.get("frame_idx", item.get("frame_id"))
            else:
                frame = item
            out.append(int(frame))
        return out

    chosen_times = result.get("chosen_times")
    if not isinstance(chosen_times, Mapping):
        raise ValueError("TR-R2 result thiếu chosen_times/chosen_frame_idxs")
    video_id = str(result["video_id"])
    return [frame_lookup.nearest_frame(video_id, float(chosen_times[event_id])) for event_id in event_ids]


def run_trake_ablation(
    *,
    gt_records: Sequence[Mapping[str, Any]],
    frame_map_path: Path | str,
    step: float = 0.16,
    min_gap: int = 1,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """
    Chạy T1 sparse và T2 dense-local+DP.

    T1/T2 được cô lập lỗi:
      - T1 lỗi không làm mất T2.
      - T2 lỗi không làm mất T1.
      - TR-R1 lỗi mới làm cả hai không thể chạy.
    """
    from aic2026.trake_retrieval import TRR1Config, tim_nhieu_su_kien
    from aic2026.trake_r2_pipeline import run_trake_r2
    from aic2026.trake_r2_windows import chon_video_rrf

    trake_gt = [
        dict(x)
        for x in gt_records
        if (
            x.get("loai_truy_van") == "chuoi_su_kien"
            or x.get("task") == "trake"
        )
    ]

    if max_queries is not None:
        trake_gt = trake_gt[: max(0, int(max_queries))]

    if not trake_gt:
        raise ValueError("không có TRAKE query trong --dev")

    frame_lookup = _load_frame_lookup(frame_map_path)

    sparse_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for gt in trake_gt:
        qid = _qid(gt)

        # ---------------------------------------------------------------
        # TR-R1 — prerequisite chung
        # ---------------------------------------------------------------
        try:
            events = _trake_events(gt)
            event_ids = [e["event_id"] for e in events]

            print(
                f"[EVAL-02][{qid}] TR-R1 START",
                flush=True,
            )

            t0 = time.perf_counter()

            tr_r1_results = tim_nhieu_su_kien(
                events,
                config=TRR1Config(),
            )

            tr_r1_ms = (
                time.perf_counter() - t0
            ) * 1000.0

            print(
                f"[EVAL-02][{qid}] TR-R1 DONE — "
                f"{tr_r1_ms:.1f} ms",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[EVAL-02][{qid}] TR-R1 ERROR: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            missing.append(
                {
                    "query_id": qid,
                    "stage": "TR-R1",
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

            # Không có coarse regions thì cả T1/T2 đều không chạy được.
            continue

        # ---------------------------------------------------------------
        # T1 sparse
        # ---------------------------------------------------------------
        try:
            events_regions = _events_regions_dict(
                tr_r1_results
            )

            sparse_video = str(
                chon_video_rrf(events_regions)
            )

            sparse_frame_ids = _sparse_frames(
                tr_r1_results,
                sparse_video,
                frame_lookup,
            )

            sparse_frame_ids = _sparse_frames(
                tr_r1_results,
                sparse_video,
                frame_lookup,
            )

            sparse_score = score_trake_prediction(
                predicted_video_id=sparse_video,
                predicted_frame_ids=sparse_frame_ids,
                gt=gt,
            )

            print(
                f"[EVAL-02][{qid}] SPARSE DONE — "
                f"video={sparse_video}",
                flush=True,
            )

            sparse_rows.append(
                {
                    "query_id": qid,
                    "status": "ok",
                    "video_id": sparse_video,
                    "frame_ids": sparse_frame_ids,
                    "latency_ms": tr_r1_ms,
                    **sparse_score,
                }
            )

        except Exception as exc:
            print(
                f"[EVAL-02][{qid}] T1 ERROR: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            missing.append(
                {
                    "query_id": qid,
                    "stage": "T1_sparse",
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

        # ---------------------------------------------------------------
        # T2 dense-local + DP
        #
        # QUAN TRỌNG: độc lập với T1.
        # ---------------------------------------------------------------
        try:
            print(
                f"[EVAL-02][{qid}] TR-R2 START",
                flush=True,
            )

            t1 = time.perf_counter()

            dense_result = run_trake_r2(
                tr_r1_results,
                step=step,
                min_gap=min_gap,
            )

            tr_r2_ms = (
                time.perf_counter() - t1
            ) * 1000.0

            print(
                f"[EVAL-02][{qid}] TR-R2 DONE — "
                f"{tr_r2_ms:.1f} ms",
                flush=True,
            )

            dense_video = str(
                dense_result["video_id"]
            )

            dense_frame_ids = _dense_frames(
                dense_result,
                frame_lookup,
                event_ids,
            )

            dense_score = score_trake_prediction(
                predicted_video_id=dense_video,
                predicted_frame_ids=dense_frame_ids,
                gt=gt,
            )

            dense_rows.append(
                {
                    "query_id": qid,
                    "status": "ok",
                    "video_id": dense_video,
                    "frame_ids": dense_frame_ids,
                    "latency_ms": (
                        tr_r1_ms + tr_r2_ms
                    ),
                    "tr_r1_ms": tr_r1_ms,
                    "tr_r2_ms": tr_r2_ms,
                    **dense_score,
                }
            )

        except Exception as exc:
            print(
                f"[EVAL-02][{qid}] T2 ERROR: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            missing.append(
                {
                    "query_id": qid,
                    "stage": "T2_dense_dp",
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

    return {
        "requested_queries": len(trake_gt),

        "completed_queries": {
            "T1_sparse": len(sparse_rows),
            "T2_dense_dp": len(dense_rows),
        },

        "missing_or_error": missing,

        "T1_sparse": {
            "summary": aggregate_trake(
                sparse_rows
            ),
            "per_query": sparse_rows,
        },

        "T2_dense_dp": {
            "summary": aggregate_trake(
                dense_rows
            ),
            "per_query": dense_rows,
            "step_seconds": step,
            "min_gap": min_gap,
        },
    }


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


def pareto_front(
    points: Sequence[Mapping[str, Any]],
    *,
    accuracy_key: str = "accuracy",
    latency_key: str = "latency_ms",
) -> list[dict[str, Any]]:
    """Maximize accuracy, minimize latency. Null/NaN points are ignored."""
    valid: list[dict[str, Any]] = []
    for point in points:
        accuracy = point.get(accuracy_key)
        latency = point.get(latency_key)
        if accuracy is None or latency is None:
            continue
        accuracy = float(accuracy)
        latency = float(latency)
        if not math.isfinite(accuracy) or not math.isfinite(latency):
            continue
        p = dict(point)
        p[accuracy_key] = accuracy
        p[latency_key] = latency
        valid.append(p)

    front: list[dict[str, Any]] = []
    for i, a in enumerate(valid):
        dominated = False
        for j, b in enumerate(valid):
            if i == j:
                continue
            no_worse = b[accuracy_key] >= a[accuracy_key] and b[latency_key] <= a[latency_key]
            strictly_better = b[accuracy_key] > a[accuracy_key] or b[latency_key] < a[latency_key]
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(a)

    front.sort(key=lambda x: (x[latency_key], -x[accuracy_key], str(x.get("config_id", ""))))
    return front


def _qa_pareto_points(qa: Mapping[str, Any], qa_scores: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    qa_scores = qa_scores or {}
    points = []
    for config_id, result in qa.get("configs", {}).items():
        total_p95 = result.get("latency_ms", {}).get("total", {}).get("p95")
        score = qa_scores.get(config_id)
        accuracy = float(score) if score is not None else result.get("reader_recall")
        points.append(
            {
                "config_id": config_id,
                "accuracy": accuracy,
                "accuracy_metric": "qa_score" if score is not None else "reader_recall",
                "latency_ms": total_p95,
                "reader_recall": result.get("reader_recall"),
                "qa_score": score,
                "mean_reader_k": result.get("reader_k", {}).get("mean"),
            }
        )
    return points


def _trake_pareto_points(trake: Mapping[str, Any]) -> list[dict[str, Any]]:
    points = []
    for config_id in ("T1_sparse", "T2_dense_dp"):
        summary = trake.get(config_id, {}).get("summary", {})
        points.append(
            {
                "config_id": config_id,
                "accuracy": summary.get("mean_event_rscore"),
                "accuracy_metric": "mean_event_rscore",
                "latency_ms": summary.get("latency_ms", {}).get("p95"),
                "video_recall": summary.get("video_recall"),
            }
        )
    return points


# ---------------------------------------------------------------------------
# Report / CLI
# ---------------------------------------------------------------------------


def _load_qa_scores(path: Path | str | None) -> dict[str, float]:
    if path is None:
        return {}
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        raise ValueError("--qa-score-json phải là JSON object config_id -> score")
    return {str(k): float(v) for k, v in doc.items() if v is not None}


def build_manifest(*, config: Mapping[str, Any], query_ids: Sequence[str], inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "eval_02.v1",
        "task": "EVAL-02",
        "run_id": config.get("run_id"),
        "timestamp": _dt.datetime.now().astimezone().isoformat(),
        "git_commit": get_git_commit(),
        "dataset_version": Path(str(inputs.get("dev", "unknown"))).name,
        "query_ids": list(query_ids),
        "config_hash": sha256_json(config),
        "model_id": "CLIP-L/14 + existing local reader",
        "precision": "inherit-from-upstream-run",
        "prompt_version": "inherit-from-upstream-run",
        "seed": None,
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "inputs": dict(inputs),
        "missing_assets": [],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown_report(
    path: Path,
    *,
    retrieval_profile: Mapping[str, Any] | None,
    qa: Mapping[str, Any],
    trake: Mapping[str, Any] | None,
    pareto: Mapping[str, Any],
) -> None:
    lines = ["# EVAL-02 — Retrieval, Adaptive K và TRAKE", ""]

    lines += ["## Retrieval", ""]
    if retrieval_profile is None:
        lines += ["N-02 profile: **chưa cung cấp**; Recall@50/100/300/500 chưa được chốt trong run này.", ""]
    else:
        recalls = retrieval_profile.get("recall_at_k", {})
        lines += [
            "| Metric | Value |",
            "|---|---:|",
            *[f"| Recall@{k} | {_fmt(recalls.get(str(k)))} |" for k in RETRIEVAL_KS],
            f"| p50 latency (ms) | {_fmt(retrieval_profile.get('latency_ms', {}).get('p50'), 2)} |",
            f"| p95 latency (ms) | {_fmt(retrieval_profile.get('latency_ms', {}).get('p95'), 2)} |",
            "",
        ]

    lines += ["## Q&A — fixed K vs adaptive K", "", "| Config | Reader recall | MRR | mean K | FAISS p95 | rerank p95 | VLM p95 | total p95 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for config_id, result in qa.get("configs", {}).items():
        lat = result.get("latency_ms", {})
        lines.append(
            f"| {config_id} | {_fmt(result.get('reader_recall'))} | {_fmt(result.get('mrr'))} | "
            f"{_fmt(result.get('reader_k', {}).get('mean'), 2)} | "
            f"{_fmt(lat.get('faiss', {}).get('p95'), 2)} | {_fmt(lat.get('rerank', {}).get('p95'), 2)} | "
            f"{_fmt(lat.get('vlm', {}).get('p95'), 2)} | {_fmt(lat.get('total', {}).get('p95'), 2)} |"
        )
    lines.append("")

    lines += ["## QA Pareto front", ""]
    qa_front = pareto.get("qa", [])
    if qa_front:
        for row in qa_front:
            lines.append(
                f"- `{row['config_id']}` — {row['accuracy_metric']}={_fmt(row['accuracy'])}, p95={_fmt(row['latency_ms'], 2)} ms"
            )
    else:
        lines.append("- Chưa xác định được vì thiếu latency tổng (thường là chưa có VLM timing).")
    lines.append("")

    lines += ["## TRAKE", ""]
    if trake is None:
        lines.append("TRAKE: **chưa chạy trong invocation này**.")
    else:
        lines += ["| Config | video recall | mean event R-Score | mean frame error | p95 ms |", "|---|---:|---:|---:|---:|"]
        for config_id in ("T1_sparse", "T2_dense_dp"):
            s = trake.get(config_id, {}).get("summary", {})
            lines.append(
                f"| {config_id} | {_fmt(s.get('video_recall'))} | {_fmt(s.get('mean_event_rscore'))} | "
                f"{_fmt(s.get('mean_frame_error'), 2)} | {_fmt(s.get('latency_ms', {}).get('p95'), 2)} |"
            )
        lines.append("")
        if trake.get("missing_or_error"):
            lines.append(f"Missing/error TRAKE: {len(trake['missing_or_error'])} query.")
            lines.append("")

    lines += ["## Acceptance", ""]
    lines.append("- Pareto chỉ được chốt khi accuracy và p95 latency đều có số đo trên cùng split/config.")
    lines.append("- VLM timing thiếu => QA Pareto được đánh dấu chưa hoàn tất, không tự thay bằng 0.")
    lines.append("- T1/T2 dùng cùng GT; T2 chỉ thay sparse bằng dense-local + DP của TR-R2.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

def _prepare_trake_windows_runtime() -> None:
    """
    Bootstrap native runtime cho TRAKE trên Windows.

    Đồng bộ với scripts/benchmark_tr_r1.py để tránh xung đột
    OpenMP giữa open_clip/torch và các dependency native của TR-R1.
    """
    if os.name != "nt":
        return

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault(
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
        "python",
    )

    # Giữ cùng import order với benchmark TR-R1.
    import open_clip  # noqa: F401


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EVAL-02 accuracy-latency Pareto")
    parser.add_argument("--dev", type=Path, required=True, help="dev_holdout/baseline_clean JSONL")
    parser.add_argument("--s4", type=Path, required=True, help="QA-S4 QueryPlan JSONL")
    parser.add_argument("--r3", type=Path, required=True, help="QA-R3 output JSON/JSONL")
    parser.add_argument("--qa-r1-report", type=Path, default=None)
    parser.add_argument("--n02-report", type=Path, default=None)
    parser.add_argument("--vlm-latency", type=Path, default=None)
    parser.add_argument("--vlm-ms-per-frame", type=float, default=None)
    parser.add_argument("--qa-score-json", type=Path, default=None)
    parser.add_argument("--run-trake", action="store_true")
    parser.add_argument("--frame-map", type=Path, default=None)
    parser.add_argument("--trake-step", type=float, default=0.16)
    parser.add_argument("--trake-min-gap", type=int, default=1)
    parser.add_argument("--trake-limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="Đánh dấu artifact dùng tạm cho downstream; metrics chưa final",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.run_trake:
        print("[EVAL-02] Preparing TRAKE native runtime...", flush=True)
        _prepare_trake_windows_runtime()
        print("[EVAL-02] TRAKE native runtime ready", flush=True)

    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_root = Path(os.environ.get("DATA_ROOT", "D:/aic-data"))
    out_dir = args.output_dir or (data_root / "runs" / f"eval_02_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_records = load_records(args.dev)
    s4_records = load_records(args.s4)
    r3_records = load_records(args.r3)

    qa = evaluate_qa_reader_k(
        gt_records=gt_records,
        r3_records=r3_records,
        s4_records=s4_records,
        r1_latency_by_query=load_qa_r1_per_query_latency(args.qa_r1_report),
        vlm_latency_by_query_config=load_vlm_latency(args.vlm_latency),
        vlm_ms_per_frame=args.vlm_ms_per_frame,
    )
    retrieval_profile = extract_n02_retrieval_profile(args.n02_report)
    qa_scores = _load_qa_scores(args.qa_score_json)

    trake = None
    if args.run_trake:
        frame_map = args.frame_map or (data_root / "index" / "frame_map.parquet")
        trake = run_trake_ablation(
            gt_records=gt_records,
            frame_map_path=frame_map,
            step=args.trake_step,
            min_gap=args.trake_min_gap,
            max_queries=args.trake_limit,
        )

    qa_points = _qa_pareto_points(qa, qa_scores)
    trake_points = _trake_pareto_points(trake) if trake is not None else []
    pareto = {
        "qa_points": qa_points,
        "qa": pareto_front(qa_points),
        "trake_points": trake_points,
        "trake": pareto_front(trake_points),
    }

    config = {
        "run_id": run_id,
        "provisional": args.provisional,
        "fixed_reader_k": list(FIXED_READER_K),
        "adaptive_policy": DEFAULT_ADAPTIVE_POLICY,
        "run_trake": args.run_trake,
        "trake_step": args.trake_step,
        "trake_min_gap": args.trake_min_gap,
        "windows_native_runtime": {
            "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": os.environ.get(
                "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"
            ),
        },
    }

    all_query_ids = list(qa["query_ids"])

    if trake is not None:
        trake_query_ids = []

        for config_id in ("T1_sparse", "T2_dense_dp"):
            for row in trake.get(config_id, {}).get("per_query", []):
                trake_query_ids.append(str(row["query_id"]))

        for item in trake.get("missing_or_error", []):
            if item.get("query_id") is not None:
                trake_query_ids.append(str(item["query_id"]))

        all_query_ids = list(
            dict.fromkeys(all_query_ids + trake_query_ids)
        )

    manifest = build_manifest(
        config=config,
        query_ids=all_query_ids,
        inputs={
            "dev": str(args.dev),
            "s4": str(args.s4),
            "r3": str(args.r3),
            "qa_r1_report": str(args.qa_r1_report) if args.qa_r1_report else None,
            "n02_report": str(args.n02_report) if args.n02_report else None,
            "vlm_latency": str(args.vlm_latency) if args.vlm_latency else None,
            "qa_score_json": str(args.qa_score_json) if args.qa_score_json else None,
        },
    )

    manifest["status"] = (
        "provisional_for_downstream"
        if args.provisional
        else "final"
    )
    manifest["metrics_final"] = not args.provisional

    if trake is not None:
        manifest["missing_assets"] = [
            {
                "query_id": item.get("query_id"),
                "stage": item.get("stage"),
                "reason": "no_dense_frame_in_local_window",
                "error": item.get("error"),
            }
            for item in trake.get("missing_or_error", [])
            if (
                item.get("stage") == "T2_dense_dp"
                and "Không có dense frame trong local window"
                in str(item.get("error", ""))
            )
        ]

    _write_json(out_dir / "manifest.json", manifest)
    _write_json(
        out_dir / "qa_ablation.json",
        {"retrieval_profile": retrieval_profile, "reader_k_ablation": qa, "qa_scores": qa_scores},
    )
    _write_json(out_dir / "trake_ablation.json", trake or {"status": "not_run"})
    _write_json(out_dir / "pareto.json", pareto)
    write_markdown_report(
        out_dir / "REPORT_EVAL_02.md",
        retrieval_profile=retrieval_profile,
        qa=qa,
        trake=trake,
        pareto=pareto,
    )

    print("=" * 72)
    print("EVAL-02 — RETRIEVAL / ADAPTIVE K / TRAKE")
    print("=" * 72)
    print(f"QA queries : {qa['n_queries']}")
    for config_id, result in qa["configs"].items():
        total_p95 = result["latency_ms"]["total"]["p95"]
        print(
            f"{config_id:<10} recall={result['reader_recall']:.4f} "
            f"MRR={result['mrr']:.4f} meanK={result['reader_k']['mean']:.2f} "
            f"p95={_fmt(total_p95, 2)} ms"
        )
    if retrieval_profile:
        print("Recall@K  :", retrieval_profile["recall_at_k"])
    if trake:
        for config_id in ("T1_sparse", "T2_dense_dp"):
            s = trake[config_id]["summary"]
            print(
                f"{config_id:<12} videoR={_fmt(s['video_recall'])} "
                f"eventR={_fmt(s['mean_event_rscore'])} "
                f"p95={_fmt(s['latency_ms']['p95'], 2)} ms"
            )
    print(f"Output     : {out_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
