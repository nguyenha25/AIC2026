"""
QA-R1 — Multi-stage Retrieval Funnel
SAGE-QA v1.1

Pipeline
--------
177,321 frames
    │
    ├── CLIP-B/32 top-500 frames
    └── CLIP-L/14 top-500 frames
              │
              ▼
    Video-level Weighted RRF
              │
              ▼
       TOP-300 VIDEO
              │
              ▼
    Frame provenance merge
    + temporal diversity
              │
              ▼
       TOP-50 FRAMES
              │
              ▼
       TOP-12 READER
              │
              ▼
     Ground Truth / TRAKE

Official QA gate
----------------
- Recall@300 videos
- Recall@50 frames
- Recall@12 frames
- Funnel recall monotonic
- Baseline regression
- Latency p50 / p95
- p95 <= 150 ms

Performance modes
-----------------
single_query:
    Official latency benchmark.
    One query -> one FAISS search pair.

micro_batch:
    Optional throughput profiling.
    NOT used for the official single-query latency gate.

Important hardening
-------------------
1. dataclass(slots=True) instead of free-form dicts internally.
2. mmap_mode="r" for query embeddings.
3. Query ID <-> embedding alignment is explicit.
4. No global retrieval_results containing all candidates.
5. Metrics are accumulated online.
6. Only compact query records are retained for manifest.
7. Deterministic tie-breaking.
8. FAISS -1 IDs are ignored.
9. Mapping lengths / dimensions / IDs are validated.
10. Frame provenance is preserved.
11. TRAKE temporal diversity is preserved.
12. Optional micro-batch profiling is separated from QA latency.

Expected files
--------------
D:/aic-data/index/faiss/clip_b32.index
D:/aic-data/index/faiss/clip_b32_ids.parquet
D:/aic-data/index/faiss/clip_l.index
D:/aic-data/index/faiss/clip_l_ids.parquet
D:/aic-data/index/frame_map.parquet
D:/aic-data/dev/dev_questions.jsonl
D:/aic-data/dev/dev_query_embeddings_clip_b.npy
D:/aic-data/dev/dev_query_embeddings_clip_l.npy

NEW REQUIRED ALIGNMENT FILES
----------------------------
D:/aic-data/dev/dev_query_embeddings_clip_b_ids.json
D:/aic-data/dev/dev_query_embeddings_clip_l_ids.json

Each alignment file:
[
    "Q001",
    "Q002",
    ...
]

The order in the alignment file corresponds to the rows in the
corresponding .npy file.

Output
------
D:/aic-data/runs/dev_qa_r1_profile.json
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


import faiss
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path("D:/aic-data")

INDEX_DIR = DATA_ROOT / "index" / "faiss"
DEV_DIR = DATA_ROOT / "dev"
RUNS_DIR = DATA_ROOT / "runs"

INDEX_B_PATH = INDEX_DIR / "clip_b32.index"
INDEX_B_IDS_PATH = INDEX_DIR / "clip_b32_ids.parquet"

INDEX_L_PATH = INDEX_DIR / "clip_l.index"
INDEX_L_IDS_PATH = INDEX_DIR / "clip_l_ids.parquet"

FRAME_MAP_PATH = DATA_ROOT / "index" / "frame_map.parquet"

DEV_PATH = DEV_DIR / "dev_questions.jsonl"

QUERY_B_PATH = DEV_DIR / "dev_query_embeddings_clip_b.npy"
QUERY_L_PATH = DEV_DIR / "dev_query_embeddings_clip_l.npy"

# Explicit query <-> embedding alignment.
QUERY_B_IDS_PATH = DEV_DIR / "dev_query_embeddings_clip_b_ids.json"
QUERY_L_IDS_PATH = DEV_DIR / "dev_query_embeddings_clip_l_ids.json"

OUTPUT_PATH = RUNS_DIR / "dev_qa_r1_profile.json"


# ============================================================
# LOCKED CONFIG
# ============================================================

SCHEMA_VERSION = "1.1"

K_SOURCE = 500
K_FUSED = 300
K_COVERAGE = 50
K_READER = 12

RRF_K = 60

W_B = 1.0
W_L = 0.2

TRAKE_FRAMES_PER_VIDEO = 3
TRAKE_MIN_GAP_SECONDS = 1.0

FRAME_TOLERANCE = 5

BUDGET_P95_MS = 150.0

EXPECTED_VECTORS = 177_321
CLIP_B_DIMENSION = 512
CLIP_L_DIMENSION = 768

# Baseline regression floor.
MIN_RECALL_300 = 56 / 76
MIN_RECALL_50 = 13 / 76
MIN_RECALL_12 = 9 / 76

# Official QA benchmark mode.
BENCHMARK_MODE = "single_query"

# Optional throughput profile.
ENABLE_MICRO_BATCH_PROFILE = True
MICRO_BATCH_SIZE = 8


# ============================================================
# TYPE-SAFE DATA STRUCTURES
# ============================================================

@dataclass(slots=True, frozen=True)
class GroundTruth:
    query_id: str
    video_id: str
    query_type: str
    frame_tolerance: int
    raw_item: dict


@dataclass(slots=True, frozen=True)
class SourceFrame:
    n: int
    source: str
    source_rank: int
    source_score: float


@dataclass(slots=True)
class SourceEvidence:
    rank: int
    score: float


@dataclass(slots=True)
class MergedFrame:
    n: int
    sources: dict[str, SourceEvidence] = field(default_factory=dict)


@dataclass(slots=True)
class VideoCandidate:
    video_id: str
    rrf_score: float
    best_b_rank: int | None
    best_l_rank: int | None
    frames: list[SourceFrame]


@dataclass(slots=True)
class FrameCandidate:
    query_id: str
    video_id: str
    n: int
    frame_idx: int
    pts_time: float

    stage: str

    source_hits: tuple[str, ...]
    source_ranks: dict[str, int]
    source_scores: dict[str, float]

    evidence_score: float
    best_source_rank: int

    score_fused: float
    rank_video: int
    rank_final: int

    status: str = "ok"
    error: str | None = None


@dataclass(slots=True, frozen=True)
class StageLatency:
    source_retrieval: float
    video_fusion: float
    top300: float
    top50: float
    top12: float
    total: float


# ============================================================
# GENERAL UTILS
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file bắt buộc:\n{path}"
        )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0

    return float(
        np.percentile(
            np.asarray(values, dtype=np.float64),
            q,
        )
    )


def summarize_latency(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "p50": 0.0,
            "p95": 0.0,
            "mean": 0.0,
        }

    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "mean": float(np.mean(values)),
    }


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(
    path: Path,
) -> tuple[list[GroundTruth], dict[str, GroundTruth]]:

    data: list[GroundTruth] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:

        for line_no, line in enumerate(f, start=1):

            if not line.strip():
                continue

            item = json.loads(line)

            query_id = str(item["id"])
            video_id = str(item["video_id"])

            if query_id in seen_ids:
                raise ValueError(
                    f"Query ID bị trùng tại line {line_no}: "
                    f"{query_id}"
                )

            seen_ids.add(query_id)

            query_type = str(
                item.get(
                    "loai_truy_van",
                    "unknown",
                )
            )

            if query_type == "chuoi_su_kien":

                events = item.get(
                    "cac_giai_doan",
                    [],
                )

                if not events:
                    raise ValueError(
                        f"Query {query_id}: "
                        "chuỗi sự kiện nhưng thiếu "
                        "cac_giai_doan."
                    )

                gt_frame_idx = int(
                    events[0]["frame_start"]
                )

            else:

                if "frame_start" not in item:
                    raise ValueError(
                        f"Query {query_id}: "
                        "thiếu frame_start."
                    )

                gt_frame_idx = int(
                    item["frame_start"]
                )

            gt = GroundTruth(
                query_id=query_id,
                video_id=video_id,
                query_type=query_type,
                frame_tolerance=FRAME_TOLERANCE,
                raw_item=item,
            )

            # Force validation.
            _ = gt_frame_idx

            data.append(gt)

    mapping = {
        item.query_id: item
        for item in data
    }

    return data, mapping


# ============================================================
# QUERY EMBEDDING ALIGNMENT
# ============================================================

def load_query_id_mapping(
    path: Path,
) -> list[str]:

    require_file(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"{path} phải chứa JSON list query_id."
        )

    query_ids = [str(x) for x in data]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError(
            f"{path} chứa query_id bị trùng."
        )

    return query_ids


def validate_embedding_alignment(
    embedding_path: Path,
    mapping_path: Path,
    ground_truth_ids: set[str],
    expected_dimension: int,
) -> np.ndarray:

    require_file(embedding_path)

    query_ids = load_query_id_mapping(
        mapping_path
    )

    embeddings = np.load(
        embedding_path,
        mmap_mode="r",
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"{embedding_path}: embedding phải "
            "là ma trận 2 chiều."
        )

    if embeddings.shape[1] != expected_dimension:
        raise ValueError(
            f"{embedding_path}: dimension = "
            f"{embeddings.shape[1]}, "
            f"mong đợi {expected_dimension}."
        )

    if len(embeddings) != len(query_ids):
        raise ValueError(
            f"{embedding_path}: số embedding "
            f"{len(embeddings):,} != số query ID "
            f"{len(query_ids):,}."
        )

    embedding_ids = set(query_ids)

    missing = ground_truth_ids - embedding_ids
    extra = embedding_ids - ground_truth_ids

    if missing:
        raise ValueError(
            f"{embedding_path}: thiếu embedding cho "
            f"{len(missing)} query ID. "
            f"Ví dụ: {sorted(missing)[:5]}"
        )

    if extra:
        raise ValueError(
            f"{embedding_path}: có embedding không "
            f"tồn tại trong dev_questions. "
            f"Ví dụ: {sorted(extra)[:5]}"
        )

    return embeddings


def build_embedding_row_map(
    query_ids: list[str],
) -> dict[str, int]:

    return {
        query_id: idx
        for idx, query_id in enumerate(query_ids)
    }


# ============================================================
# FRAME MAP
# ============================================================

def load_frame_lookup(
    path: Path,
) -> dict[tuple[str, int], tuple[int, float]]:

    frame_map = pd.read_parquet(path)

    required = {
        "video_id",
        "n",
        "pts_time",
        "fps",
        "frame_idx",
    }

    missing = required - set(frame_map.columns)

    if missing:
        raise ValueError(
            f"frame_map thiếu cột: "
            f"{sorted(missing)}"
        )

    lookup: dict[
        tuple[str, int],
        tuple[int, float],
    ] = {}

    for row in frame_map.itertuples(
        index=False
    ):

        key = (
            str(row.video_id),
            int(row.n),
        )

        if key in lookup:
            raise ValueError(
                f"frame_map duplicate key: {key}"
            )

        lookup[key] = (
            int(row.frame_idx),
            float(row.pts_time),
        )

    return lookup


# ============================================================
# FAISS ID MAPPING
# ============================================================

def load_index_ids(
    path: Path,
    expected_ntotal: int,
) -> list[tuple[str, int]]:

    df = pd.read_parquet(path)

    required = {
        "video_id",
        "n",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} thiếu cột: "
            f"{sorted(missing)}"
        )

    if len(df) != expected_ntotal:
        raise ValueError(
            f"{path}: {len(df):,} rows != "
            f"FAISS ntotal {expected_ntotal:,}"
        )

    result = [
        (
            str(video_id),
            int(n),
        )
        for video_id, n in zip(
            df["video_id"],
            df["n"],
        )
    ]

    return result


# ============================================================
# STAGE 1 — VIDEO LEVEL WEIGHTED RRF
# ============================================================

def fuse_video_rrf(
    b_indices: np.ndarray,
    b_scores: np.ndarray,
    b_ids: list[tuple[str, int]],
    l_indices: np.ndarray,
    l_scores: np.ndarray,
    l_ids: list[tuple[str, int]],
) -> list[VideoCandidate]:

    # Internal temporary state.
    scores: dict[str, float] = {}

    best_b: dict[str, int] = {}
    best_l: dict[str, int] = {}

    frames: dict[
        str,
        list[SourceFrame],
    ] = {}

    def add_source(
        indices: np.ndarray,
        scores_array: np.ndarray,
        ids: list[tuple[str, int]],
        source: str,
        weight: float,
    ) -> None:

        seen_videos: set[str] = set()
        video_rank = 0

        for frame_rank, vector_id_raw in enumerate(
            indices,
            start=1,
        ):

            vector_id = int(vector_id_raw)

            if vector_id < 0:
                continue

            if vector_id >= len(ids):
                raise IndexError(
                    f"FAISS ID {vector_id} vượt mapping "
                    f"size {len(ids)}."
                )

            video_id, n = ids[vector_id]

            source_frame = SourceFrame(
                n=int(n),
                source=source,
                source_rank=frame_rank,
                source_score=float(
                    scores_array[frame_rank - 1]
                ),
            )

            frames.setdefault(
                video_id,
                [],
            ).append(source_frame)

            if video_id in seen_videos:
                continue

            seen_videos.add(video_id)

            video_rank += 1

            scores[video_id] = (
                scores.get(video_id, 0.0)
                + weight / (RRF_K + video_rank)
            )

            if source == "clip_b":
                best_b[video_id] = video_rank
            else:
                best_l[video_id] = video_rank

    add_source(
        b_indices,
        b_scores,
        b_ids,
        "clip_b",
        W_B,
    )

    add_source(
        l_indices,
        l_scores,
        l_ids,
        "clip_l",
        W_L,
    )

    video_ids = sorted(
        scores.keys(),
        key=lambda video_id: (
            -scores[video_id],
            video_id,
        ),
    )

    result: list[VideoCandidate] = []

    for video_id in video_ids:

        result.append(
            VideoCandidate(
                video_id=video_id,
                rrf_score=float(
                    scores[video_id]
                ),
                best_b_rank=best_b.get(
                    video_id
                ),
                best_l_rank=best_l.get(
                    video_id
                ),
                frames=frames[video_id],
            )
        )

    return result


# ============================================================
# STAGE 2 — FRAME PROVENANCE
# ============================================================

def merge_frame_provenance(
    frames: list[SourceFrame],
) -> list[MergedFrame]:

    by_n: dict[int, MergedFrame] = {}

    for frame in frames:

        merged = by_n.get(frame.n)

        if merged is None:

            merged = MergedFrame(
                n=frame.n
            )

            by_n[frame.n] = merged

        current = merged.sources.get(
            frame.source
        )

        if (
            current is None
            or frame.source_rank < current.rank
            or (
                frame.source_rank == current.rank
                and frame.source_score > current.score
            )
        ):

            merged.sources[frame.source] = (
                SourceEvidence(
                    rank=frame.source_rank,
                    score=frame.source_score,
                )
            )

    return list(by_n.values())


def calculate_frame_evidence(
    sources: dict[str, SourceEvidence],
) -> float:

    total = 0.0

    for source, evidence in sources.items():

        weight = (
            W_B
            if source == "clip_b"
            else W_L
        )

        total += (
            weight
            / (RRF_K + evidence.rank)
        )

    return total


def build_frame_candidate(
    query_id: str,
    video: VideoCandidate,
    frame: MergedFrame,
    frame_lookup: dict[
        tuple[str, int],
        tuple[int, float],
    ],
) -> FrameCandidate | None:

    key = (
        video.video_id,
        frame.n,
    )

    meta = frame_lookup.get(key)

    if meta is None:
        return None

    frame_idx, pts_time = meta

    evidence_score = (
        calculate_frame_evidence(
            frame.sources
        )
    )

    best_source_rank = min(
        evidence.rank
        for evidence in frame.sources.values()
    )

    source_ranks = {
        source: int(evidence.rank)
        for source, evidence
        in frame.sources.items()
    }

    source_scores = {
        source: float(evidence.score)
        for source, evidence
        in frame.sources.items()
    }

    return FrameCandidate(
        query_id=query_id,
        video_id=video.video_id,
        n=frame.n,
        frame_idx=frame_idx,
        pts_time=pts_time,
        stage="coverage",
        source_hits=tuple(
            sorted(frame.sources.keys())
        ),
        source_ranks=source_ranks,
        source_scores=source_scores,
        evidence_score=evidence_score,
        best_source_rank=best_source_rank,
        score_fused=video.rrf_score,
        rank_video=0,
        rank_final=0,
    )


# ============================================================
# STAGE 2 — TEMPORAL DIVERSITY
# ============================================================

def select_diverse_frames(
    candidates: list[FrameCandidate],
    videos_by_id: dict[str, VideoCandidate],
    query_type: str,
    limit: int,
) -> list[FrameCandidate]:

    if not candidates or limit <= 0:
        return []

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.evidence_score,
            candidate.rank_video,
            candidate.best_source_rank,
            candidate.video_id,
            candidate.n,
        ),
    )

    quota = (
        TRAKE_FRAMES_PER_VIDEO
        if query_type == "chuoi_su_kien"
        else 1
    )

    selected: list[FrameCandidate] = []

    selected_by_video: dict[
        str,
        list[FrameCandidate],
    ] = {}

    # --------------------------------------------------------
    # PASS 1
    # --------------------------------------------------------

    for candidate in ranked:

        if len(selected) >= limit:
            break

        video_id = candidate.video_id

        chosen = selected_by_video.setdefault(
            video_id,
            [],
        )

        if len(chosen) >= quota:
            continue

        too_close = any(
            abs(
                candidate.pts_time
                - other.pts_time
            )
            < TRAKE_MIN_GAP_SECONDS
            for other in chosen
        )

        if too_close:
            continue

        selected.append(candidate)
        chosen.append(candidate)

    # --------------------------------------------------------
    # PASS 2
    #
    # Fallback để bảo đảm K_COVERAGE nếu diversity
    # quá mạnh.
    # --------------------------------------------------------

    if len(selected) < limit:

        selected_keys = {
            (
                candidate.video_id,
                candidate.n,
            )
            for candidate in selected
        }

        for candidate in ranked:

            if len(selected) >= limit:
                break

            key = (
                candidate.video_id,
                candidate.n,
            )

            if key in selected_keys:
                continue

            selected.append(candidate)
            selected_keys.add(key)

    return selected[:limit]


def extract_top50_coverage(
    query_id: str,
    query_type: str,
    top300: list[VideoCandidate],
    frame_lookup: dict[
        tuple[str, int],
        tuple[int, float],
    ],
) -> list[FrameCandidate]:

    all_candidates: list[FrameCandidate] = []

    for video_rank, video in enumerate(
        top300,
        start=1,
    ):

        video.frames = video.frames

        merged_frames = merge_frame_provenance(
            video.frames
        )

        for merged_frame in merged_frames:

            candidate = build_frame_candidate(
                query_id=query_id,
                video=video,
                frame=merged_frame,
                frame_lookup=frame_lookup,
            )

            if candidate is None:
                continue

            candidate.rank_video = video_rank

            all_candidates.append(
                candidate
            )

    videos_by_id = {
        video.video_id: video
        for video in top300
    }

    selected = select_diverse_frames(
        candidates=all_candidates,
        videos_by_id=videos_by_id,
        query_type=query_type,
        limit=K_COVERAGE,
    )

    for rank, candidate in enumerate(
        selected,
        start=1,
    ):
        candidate.rank_final = rank

    return selected


# ============================================================
# STAGE 3 — TOP 12 READER
# ============================================================

def select_top12_reader(
    candidates_50: list[FrameCandidate],
) -> list[FrameCandidate]:

    selected: list[FrameCandidate] = []

    for rank, candidate in enumerate(
        candidates_50[:K_READER],
        start=1,
    ):

        candidate.stage = "reader"
        candidate.rank_final = rank

        selected.append(candidate)

    return selected


# ============================================================
# GROUND TRUTH EVALUATION
# ============================================================

def frame_hits_event(
    candidate: FrameCandidate,
    target_video: str,
    frame_start: int,
    frame_end: int,
    tolerance: int,
) -> bool:

    return (
        candidate.video_id == target_video
        and
        frame_start - tolerance
        <= candidate.frame_idx
        <= frame_end + tolerance
    )


def frame_candidates_hit(
    candidates: list[FrameCandidate],
    gt: GroundTruth,
) -> bool:

    target_video = gt.video_id
    tolerance = gt.frame_tolerance
    raw = gt.raw_item

    # --------------------------------------------------------
    # TRAKE
    # --------------------------------------------------------

    if gt.query_type == "chuoi_su_kien":

        events = raw.get(
            "cac_giai_doan",
            [],
        )

        if not events:
            return False

        for event in events:

            frame_start = int(
                event["frame_start"]
            )

            frame_end = int(
                event["frame_end"]
            )

            found = any(
                frame_hits_event(
                    candidate,
                    target_video,
                    frame_start,
                    frame_end,
                    tolerance,
                )
                for candidate in candidates
            )

            if not found:
                return False

        return True

    # --------------------------------------------------------
    # QUERY ĐƠN
    # --------------------------------------------------------

    frame_start = int(
        raw["frame_start"]
    )

    frame_end = int(
        raw.get(
            "frame_end",
            frame_start + 5,
        )
    )

    return any(
        frame_hits_event(
            candidate,
            target_video,
            frame_start,
            frame_end,
            tolerance,
        )
        for candidate in candidates
    )


def video_pool_hit(
    videos: list[VideoCandidate],
    gt: GroundTruth,
) -> bool:

    return any(
        video.video_id == gt.video_id
        for video in videos
    )


# ============================================================
# QUALITY GATE
# ============================================================

def evaluate_quality_gate(
    recall_300: float,
    recall_50: float,
    recall_12: float,
    num_queries: int,
) -> dict:

    thresholds = {
        "recall_at_300_videos": MIN_RECALL_300,
        "recall_at_50_frames": MIN_RECALL_50,
        "recall_at_12_frames": MIN_RECALL_12,
    }

    observed = {
        "recall_at_300_videos": recall_300,
        "recall_at_50_frames": recall_50,
        "recall_at_12_frames": recall_12,
    }

    checks = {
        name: (
            observed[name]
            + 1e-12
            >= threshold
        )
        for name, threshold
        in thresholds.items()
    }

    return {
        "kind": "baseline_regression",
        "baseline_reference_queries": 76,
        "evaluated_queries": num_queries,
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ============================================================
# QUERY RECORD SERIALIZATION
# ============================================================

def frame_to_json(
    candidate: FrameCandidate,
) -> dict:

    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": candidate.query_id,
        "event_id": None,
        "video_id": candidate.video_id,
        "n": candidate.n,
        "frame_idx": candidate.frame_idx,
        "pts_time": candidate.pts_time,
        "stage": candidate.stage,
        "source_hits": list(
            candidate.source_hits
        ),
        "source_ranks": candidate.source_ranks,
        "scores": candidate.source_scores,
        "evidence_score": candidate.evidence_score,
        "best_source_rank": candidate.best_source_rank,
        "score_fused": candidate.score_fused,
        "rank_video": candidate.rank_video,
        "rank_final": candidate.rank_final,
        "status": candidate.status,
        "error": candidate.error,
    }


# ============================================================
# OFFICIAL SINGLE-QUERY BENCHMARK
# ============================================================

def run_single_query_benchmark(
    index_b: faiss.Index,
    index_l: faiss.Index,
    b_ids: list[tuple[str, int]],
    l_ids: list[tuple[str, int]],
    frame_lookup: dict[
        tuple[str, int],
        tuple[int, float],
    ],
    gt_data: list[GroundTruth],
    query_b: np.ndarray,
    query_l: np.ndarray,
    b_row_map: dict[str, int],
    l_row_map: dict[str, int],
) -> tuple[
    dict[str, list[FrameCandidate]],
    dict[str, list[FrameCandidate]],
    dict[str, list[VideoCandidate]],
    list[dict],
    dict[str, list[float]],
]:

    stage_latencies = {
        "source_retrieval_ms": [],
        "video_fusion_ms": [],
        "top300_ms": [],
        "top50_ms": [],
        "top12_ms": [],
        "total_ms": [],
    }

    records: list[dict] = []

    results_300: dict[
        str,
        list[VideoCandidate],
    ] = {}

    results_50: dict[
        str,
        list[FrameCandidate],
    ] = {}

    results_12: dict[
        str,
        list[FrameCandidate],
    ] = {}

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    first_gt = gt_data[0]

    first_b_idx = b_row_map[
        first_gt.query_id
    ]

    first_l_idx = l_row_map[
        first_gt.query_id
    ]

    index_b.search(
        np.asarray(
            query_b[first_b_idx:first_b_idx + 1],
            dtype=np.float32,
        ),
        K_SOURCE,
    )

    index_l.search(
        np.asarray(
            query_l[first_l_idx:first_l_idx + 1],
            dtype=np.float32,
        ),
        K_SOURCE,
    )

    # --------------------------------------------------------
    # Query loop
    # --------------------------------------------------------

    for gt in gt_data:

        total_start = time.perf_counter()

        b_idx = b_row_map[
            gt.query_id
        ]

        l_idx = l_row_map[
            gt.query_id
        ]

        # ----------------------------------------------------
        # Stage 0 — source retrieval
        # ----------------------------------------------------

        start = time.perf_counter()

        q_b = np.asarray(
            query_b[b_idx:b_idx + 1],
            dtype=np.float32,
        )

        q_l = np.asarray(
            query_l[l_idx:l_idx + 1],
            dtype=np.float32,
        )

        b_scores, b_indices = index_b.search(
            q_b,
            K_SOURCE,
        )

        l_scores, l_indices = index_l.search(
            q_l,
            K_SOURCE,
        )

        source_elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        # ----------------------------------------------------
        # Stage 1 — video RRF
        # ----------------------------------------------------

        start = time.perf_counter()

        ranked_videos = fuse_video_rrf(
            b_indices=b_indices[0],
            b_scores=b_scores[0],
            b_ids=b_ids,
            l_indices=l_indices[0],
            l_scores=l_scores[0],
            l_ids=l_ids,
        )

        fusion_elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        # ----------------------------------------------------
        # Stage 1 output — Top 300
        # ----------------------------------------------------

        start = time.perf_counter()

        top300 = ranked_videos[
            :K_FUSED
        ]

        top300_elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        # ----------------------------------------------------
        # Stage 2 — Top 50
        # ----------------------------------------------------

        start = time.perf_counter()

        top50 = extract_top50_coverage(
            query_id=gt.query_id,
            query_type=gt.query_type,
            top300=top300,
            frame_lookup=frame_lookup,
        )

        top50_elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        # ----------------------------------------------------
        # Stage 3 — Top 12
        # ----------------------------------------------------

        start = time.perf_counter()

        top12 = select_top12_reader(
            top50
        )

        top12_elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        total_elapsed = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        stage_latencies[
            "source_retrieval_ms"
        ].append(source_elapsed)

        stage_latencies[
            "video_fusion_ms"
        ].append(fusion_elapsed)

        stage_latencies[
            "top300_ms"
        ].append(top300_elapsed)

        stage_latencies[
            "top50_ms"
        ].append(top50_elapsed)

        stage_latencies[
            "top12_ms"
        ].append(top12_elapsed)

        stage_latencies[
            "total_ms"
        ].append(total_elapsed)

        results_300[
            gt.query_id
        ] = top300

        results_50[
            gt.query_id
        ] = top50

        results_12[
            gt.query_id
        ] = top12

        # ----------------------------------------------------
        # GT video rank
        # ----------------------------------------------------

        gt_video_rank = None

        for rank, video in enumerate(
            ranked_videos,
            start=1,
        ):

            if video.video_id == gt.video_id:

                gt_video_rank = rank
                break

        hit_300 = video_pool_hit(
            top300,
            gt,
        )

        hit_50 = frame_candidates_hit(
            top50,
            gt,
        )

        hit_12 = frame_candidates_hit(
            top12,
            gt,
        )

        # ----------------------------------------------------
        # Compact query record.
        #
        # We intentionally DO NOT store all top300
        # candidates here.
        # ----------------------------------------------------

        records.append(
            {
                "query_id": gt.query_id,
                "query_type": gt.query_type,
                "gt_video_id": gt.video_id,
                "gt_video_rank": gt_video_rank,
                "hit_video_300": hit_300,
                "hit_frame_50": hit_50,
                "hit_frame_12": hit_12,
                "num_source_frames_b": int(
                    len(b_indices[0])
                ),
                "num_source_frames_l": int(
                    len(l_indices[0])
                ),
                "num_videos_fused": len(
                    ranked_videos
                ),
                "num_videos_top300": len(
                    top300
                ),
                "num_candidates_50": len(
                    top50
                ),
                "num_candidates_12": len(
                    top12
                ),
                "reader_candidates": [
                    frame_to_json(
                        candidate
                    )
                    for candidate in top12
                ],
                "latency_ms": {
                    "source_retrieval": source_elapsed,
                    "video_fusion": fusion_elapsed,
                    "top300": top300_elapsed,
                    "top50": top50_elapsed,
                    "top12": top12_elapsed,
                    "total": total_elapsed,
                },
            }
        )

        # Explicitly remove temporary references.
        del q_b
        del q_l
        del b_scores
        del b_indices
        del l_scores
        del l_indices
        del ranked_videos
        del top300
        del top50
        del top12

    return (
        results_12,
        results_50,
        results_300,
        records,
        stage_latencies,
    )


# ============================================================
# OPTIONAL MICRO-BATCH THROUGHPUT PROFILE
# ============================================================

def run_micro_batch_profile(
    index_b: faiss.Index,
    index_l: faiss.Index,
    query_b: np.ndarray,
    query_l: np.ndarray,
    b_row_map: dict[str, int],
    l_row_map: dict[str, int],
    gt_data: list[GroundTruth],
) -> dict:

    if not gt_data:
        return {
            "enabled": True,
            "batch_size": MICRO_BATCH_SIZE,
            "queries": 0,
            "total_ms": 0.0,
            "queries_per_second": 0.0,
            "mean_ms_per_query": 0.0,
        }

    ordered_b_indices = [
        b_row_map[gt.query_id]
        for gt in gt_data
    ]

    ordered_l_indices = [
        l_row_map[gt.query_id]
        for gt in gt_data
    ]

    total_queries = len(gt_data)

    batch_times: list[float] = []

    start_total = time.perf_counter()

    for start_idx in range(
        0,
        total_queries,
        MICRO_BATCH_SIZE,
    ):

        end_idx = min(
            start_idx + MICRO_BATCH_SIZE,
            total_queries,
        )

        b_rows = ordered_b_indices[
            start_idx:end_idx
        ]

        l_rows = ordered_l_indices[
            start_idx:end_idx
        ]

        batch_b = np.asarray(
            query_b[b_rows],
            dtype=np.float32,
        )

        batch_l = np.asarray(
            query_l[l_rows],
            dtype=np.float32,
        )

        start = time.perf_counter()

        index_b.search(
            batch_b,
            K_SOURCE,
        )

        index_l.search(
            batch_l,
            K_SOURCE,
        )

        batch_elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        batch_times.append(
            batch_elapsed
        )

        del batch_b
        del batch_l

    total_elapsed = (
        time.perf_counter()
        - start_total
    ) * 1000.0

    qps = (
        total_queries
        / (total_elapsed / 1000.0)
        if total_elapsed > 0
        else 0.0
    )

    return {
        "enabled": True,
        "batch_size": MICRO_BATCH_SIZE,
        "queries": total_queries,
        "total_ms": total_elapsed,
        "queries_per_second": qps,
        "mean_ms_per_query": (
            total_elapsed
            / total_queries
        ),
        "batch_latency_ms": summarize_latency(
            batch_times
        ),
    }


# ============================================================
# MAIN
# ============================================================

def run_profiling() -> None:

    print(
        "[QA-R1] Bắt đầu multi-stage retrieval funnel...",
        flush=True,
    )

    if BENCHMARK_MODE != "single_query":
        raise ValueError(
            "QA-R1 official mode phải là "
            "'single_query'. "
            "Micro-batch chỉ dùng profiling."
        )

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    required_files = [
        INDEX_B_PATH,
        INDEX_B_IDS_PATH,
        INDEX_L_PATH,
        INDEX_L_IDS_PATH,
        FRAME_MAP_PATH,
        DEV_PATH,
        QUERY_B_PATH,
        QUERY_L_PATH,
        QUERY_B_IDS_PATH,
        QUERY_L_IDS_PATH,
    ]

    for path in required_files:
        require_file(path)

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    print(
        "[QA-R1] Đọc FAISS CLIP-B...",
        flush=True,
    )

    index_b = faiss.read_index(
        str(INDEX_B_PATH)
    )

    print(
        f"[QA-R1] CLIP-B ntotal = "
        f"{index_b.ntotal:,}",
        flush=True,
    )

    print(
        "[QA-R1] Đọc FAISS CLIP-L...",
        flush=True,
    )

    index_l = faiss.read_index(
        str(INDEX_L_PATH)
    )

    print(
        f"[QA-R1] CLIP-L ntotal = "
        f"{index_l.ntotal:,}",
        flush=True,
    )

    # --------------------------------------------------------
    # Index validation
    # --------------------------------------------------------

    if index_b.ntotal != EXPECTED_VECTORS:
        raise ValueError(
            f"CLIP-B ntotal = {index_b.ntotal:,}, "
            f"mong đợi {EXPECTED_VECTORS:,}."
        )

    if index_l.ntotal != EXPECTED_VECTORS:
        raise ValueError(
            f"CLIP-L ntotal = {index_l.ntotal:,}, "
            f"mong đợi {EXPECTED_VECTORS:,}."
        )

    if index_b.ntotal != index_l.ntotal:
        raise ValueError(
            "CLIP-B và CLIP-L ntotal khác nhau."
        )

    if index_b.d != CLIP_B_DIMENSION:
        raise ValueError(
            f"CLIP-B dimension = {index_b.d}, "
            f"mong đợi {CLIP_B_DIMENSION}."
        )

    if index_l.d != CLIP_L_DIMENSION:
        raise ValueError(
            f"CLIP-L dimension = {index_l.d}, "
            f"mong đợi {CLIP_L_DIMENSION}."
        )

    # --------------------------------------------------------
    # Load mappings
    # --------------------------------------------------------

    print(
        "[QA-R1] Đọc FAISS ID mappings...",
        flush=True,
    )

    b_ids = load_index_ids(
        INDEX_B_IDS_PATH,
        index_b.ntotal,
    )

    l_ids = load_index_ids(
        INDEX_L_IDS_PATH,
        index_l.ntotal,
    )

    # --------------------------------------------------------
    # Frame map
    # --------------------------------------------------------

    print(
        "[QA-R1] Đọc frame_map...",
        flush=True,
    )

    frame_lookup = load_frame_lookup(
        FRAME_MAP_PATH
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print(
        "[QA-R1] Đọc ground truth...",
        flush=True,
    )

    gt_data, ground_truth_map = (
        load_ground_truth(
            DEV_PATH
        )
    )

    if not gt_data:
        raise ValueError(
            "dev_questions.jsonl không có query."
        )

    ground_truth_ids = set(
        ground_truth_map.keys()
    )

    print(
        f"[QA-R1] Queries = "
        f"{len(gt_data)}",
        flush=True,
    )

    # --------------------------------------------------------
    # Query embeddings — B
    # --------------------------------------------------------

    print(
        "[QA-R1] mmap CLIP-B query embeddings...",
        flush=True,
    )

    query_b = validate_embedding_alignment(
        embedding_path=QUERY_B_PATH,
        mapping_path=QUERY_B_IDS_PATH,
        ground_truth_ids=ground_truth_ids,
        expected_dimension=CLIP_B_DIMENSION,
    )

    b_embedding_ids = load_query_id_mapping(
        QUERY_B_IDS_PATH
    )

    b_row_map = build_embedding_row_map(
        b_embedding_ids
    )

    # --------------------------------------------------------
    # Query embeddings — L
    # --------------------------------------------------------

    print(
        "[QA-R1] mmap CLIP-L query embeddings...",
        flush=True,
    )

    query_l = validate_embedding_alignment(
        embedding_path=QUERY_L_PATH,
        mapping_path=QUERY_L_IDS_PATH,
        ground_truth_ids=ground_truth_ids,
        expected_dimension=CLIP_L_DIMENSION,
    )

    l_embedding_ids = load_query_id_mapping(
        QUERY_L_IDS_PATH
    )

    l_row_map = build_embedding_row_map(
        l_embedding_ids
    )

    # --------------------------------------------------------
    # Single-query official benchmark
    # --------------------------------------------------------

    (
        results_12,
        results_50,
        results_300,
        query_records,
        stage_latencies,
    ) = run_single_query_benchmark(
        index_b=index_b,
        index_l=index_l,
        b_ids=b_ids,
        l_ids=l_ids,
        frame_lookup=frame_lookup,
        gt_data=gt_data,
        query_b=query_b,
        query_l=query_l,
        b_row_map=b_row_map,
        l_row_map=l_row_map,
    )

    # --------------------------------------------------------
    # Recall
    #
    # Metrics are accumulated from compact per-query results.
    # --------------------------------------------------------

    hits_300 = sum(
        1
        for record in query_records
        if record["hit_video_300"]
    )

    hits_50 = sum(
        1
        for record in query_records
        if record["hit_frame_50"]
    )

    hits_12 = sum(
        1
        for record in query_records
        if record["hit_frame_12"]
    )

    total_queries = len(gt_data)

    recall_300 = (
        hits_300 / total_queries
    )

    recall_50 = (
        hits_50 / total_queries
    )

    recall_12 = (
        hits_12 / total_queries
    )

    # --------------------------------------------------------
    # Funnel monotonic
    # --------------------------------------------------------

    funnel_recall_monotonic = (
        recall_300
        >= recall_50
        >= recall_12
    )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latency_report = {
        stage: summarize_latency(
            values
        )
        for stage, values
        in stage_latencies.items()
    }

    latency_pass = (
        latency_report["total_ms"]["p95"]
        <= BUDGET_P95_MS
    )

    # --------------------------------------------------------
    # Regression gate
    # --------------------------------------------------------

    quality_gate = evaluate_quality_gate(
        recall_300=recall_300,
        recall_50=recall_50,
        recall_12=recall_12,
        num_queries=total_queries,
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    status = (
        "PASSED"
        if (
            funnel_recall_monotonic
            and latency_pass
            and quality_gate["passed"]
        )
        else "FAILED"
    )

    # --------------------------------------------------------
    # Optional throughput profile
    # --------------------------------------------------------

    micro_batch_report = {
        "enabled": False
    }

    if ENABLE_MICRO_BATCH_PROFILE:

        print(
            "\n[QA-R1] Chạy optional micro-batch "
            "throughput profile...",
            flush=True,
        )

        micro_batch_report = (
            run_micro_batch_profile(
                index_b=index_b,
                index_l=index_l,
                query_b=query_b,
                query_l=query_l,
                b_row_map=b_row_map,
                l_row_map=l_row_map,
                gt_data=gt_data,
            )
        )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("\n" + "=" * 76)

    print(
        "BÁO CÁO KẾT QUẢ TASK QA-R1"
    )

    print("=" * 76)

    print(
        f"Queries               : "
        f"{total_queries}"
    )

    print(
        f"FAISS vectors         : "
        f"{index_b.ntotal:,}"
    )

    print(
        f"CLIP-B dimension      : "
        f"{index_b.d}"
    )

    print(
        f"CLIP-L dimension      : "
        f"{index_l.d}"
    )

    print("-" * 76)

    print(
        f"K_SOURCE              : "
        f"{K_SOURCE}"
    )

    print(
        f"K_FUSED videos        : "
        f"{K_FUSED}"
    )

    print(
        f"K_COVERAGE frames     : "
        f"{K_COVERAGE}"
    )

    print(
        f"K_READER frames       : "
        f"{K_READER}"
    )

    print(
        f"RRF_K                 : "
        f"{RRF_K}"
    )

    print(
        f"W_B                   : "
        f"{W_B}"
    )

    print(
        f"W_L                   : "
        f"{W_L}"
    )

    print("-" * 76)

    print(
        f"Recall@300 videos     : "
        f"{recall_300:.4f} "
        f"({hits_300}/{total_queries})"
    )

    print(
        f"Recall@50 frames      : "
        f"{recall_50:.4f} "
        f"({hits_50}/{total_queries})"
    )

    print(
        f"Recall@12 frames      : "
        f"{recall_12:.4f} "
        f"({hits_12}/{total_queries})"
    )

    print(
        "Funnel monotonic      : "
        f"{'PASS' if funnel_recall_monotonic else 'FAIL'}"
    )

    print(
        "Baseline regression   : "
        f"{'PASS' if quality_gate['passed'] else 'FAIL'}"
    )

    print("-" * 76)

    for stage, values in latency_report.items():

        print(
            f"{stage:<25}"
            f"p50={values['p50']:>8.2f} ms | "
            f"p95={values['p95']:>8.2f} ms"
        )

    print("-" * 76)

    print(
        f"P95 budget            : "
        f"{BUDGET_P95_MS:.2f} ms"
    )

    print(
        f"Latency gate          : "
        f"{'PASS' if latency_pass else 'FAIL'}"
    )

    if micro_batch_report.get(
        "enabled",
        False,
    ):

        print("-" * 76)

        print(
            f"Micro-batch size      : "
            f"{micro_batch_report['batch_size']}"
        )

        print(
            f"Micro-batch QPS       : "
            f"{micro_batch_report['queries_per_second']:.2f}"
        )

        print(
            f"Micro-batch ms/query  : "
            f"{micro_batch_report['mean_ms_per_query']:.2f}"
        )

    print("-" * 76)

    print(
        f"QA-R1 Status          : "
        f"{status}"
    )

    print("=" * 76)

    # --------------------------------------------------------
    # Output manifest
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "schema_version": SCHEMA_VERSION,
        "task": "QA-R1",
        "retrieval_method": (
            "n02_video_level_weighted_rrf_"
            "multi_stage_funnel"
        ),
        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),

        "benchmark": {
            "official_mode": BENCHMARK_MODE,
            "micro_batch_profile": (
                micro_batch_report
            ),
        },

        "dataset_info": {
            "num_queries": total_queries,
            "index_ntotal": int(
                index_b.ntotal
            ),
            "clip_b_dimension": int(
                index_b.d
            ),
            "clip_l_dimension": int(
                index_l.d
            ),
            "expected_vectors": (
                EXPECTED_VECTORS
            ),
        },

        "input_sha256": {
            "dev_questions": sha256_file(
                DEV_PATH
            ),
            "query_embeddings_clip_b": (
                sha256_file(
                    QUERY_B_PATH
                )
            ),
            "query_embeddings_clip_l": (
                sha256_file(
                    QUERY_L_PATH
                )
            ),
            "query_embeddings_clip_b_ids": (
                sha256_file(
                    QUERY_B_IDS_PATH
                )
            ),
            "query_embeddings_clip_l_ids": (
                sha256_file(
                    QUERY_L_IDS_PATH
                )
            ),
        },

        "funnel": {
            "K_source": K_SOURCE,
            "K_fused_videos": K_FUSED,
            "K_coverage_frames": K_COVERAGE,
            "K_reader_frames": K_READER,
            "trake_frames_per_video": (
                TRAKE_FRAMES_PER_VIDEO
            ),
            "trake_min_gap_seconds": (
                TRAKE_MIN_GAP_SECONDS
            ),
        },

        "rrf": {
            "rrf_k": RRF_K,
            "weight_clip_b": W_B,
            "weight_clip_l": W_L,
            "level": "video",
        },

        "metrics": {
            "recall_at_300_videos": recall_300,
            "recall_at_50_frames": recall_50,
            "recall_at_12_frames": recall_12,
            "hits": {
                "at_300_videos": hits_300,
                "at_50_frames": hits_50,
                "at_12_frames": hits_12,
            },
            "funnel_recall_monotonic": (
                funnel_recall_monotonic
            ),
            "latency_ms": latency_report,
        },

        "quality_gate": quality_gate,

        "status": status.lower(),

        "query_records": query_records,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[QA-R1] Đã ghi manifest:\n"
        f"{OUTPUT_PATH}",
        flush=True,
    )

    # --------------------------------------------------------
    # Important mmap cleanup.
    # --------------------------------------------------------

    del query_b
    del query_l

    # FAISS Python objects.
    del index_b
    del index_l

    print(
        "[QA-R1] Hoàn tất.",
        flush=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_profiling()