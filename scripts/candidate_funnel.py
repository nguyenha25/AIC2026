"""
QA-R1 — Multi-stage Retrieval Funnel (SAGE-QA v1.1)

Pipeline:
    Full corpus: 177,321 frames
            ↓
    CLIP-B/32 top-500 frames
            +
    CLIP-L/14 top-500 frames
            ↓
    Video-level Weighted RRF
            ↓
    TOP-300 VIDEOS
            ↓
    Video diversity / frame deduplication
            ↓
    TOP-50 FRAME CANDIDATES
            ↓
    TOP-12 READER CANDIDATES

QA-R1 constraints:
    - Reuse N-02 video-level weighted RRF.
    - No VLM at full-corpus retrieval.
    - Top-300 is explicitly a VIDEO candidate pool.
    - Recall@300 is evaluated at VIDEO level.
    - Top-50 and Top-12 are FRAME candidate pools.
    - Preserve video_id + frame_idx mapping.
    - Preserve CandidateRecord Schema 1.1.
    - Preserve source provenance.
    - Measure latency at every funnel stage.

Locked configuration:
    K_SOURCE   = 500
    K_FUSED    = 300 videos
    K_COVERAGE = 50 frames
    K_READER   = 12 frames

Output:
    D:/aic-data/runs/dev_qa_r1_profile.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

K_SOURCE = 500
K_FUSED = 300
K_COVERAGE = 50
K_READER = 12

BUDGET_P95_MS = 150.0

W_B = 1.0
W_L = 0.2
RRF_K = 60

EXPECTED_VECTORS = 177_321

CLIP_B_DIMENSION = 512
CLIP_L_DIMENSION = 768

SCHEMA_VERSION = "1.1"


# ============================================================
# PATHS
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

OUTPUT_PATH = RUNS_DIR / "dev_qa_r1_profile.json"


# ============================================================
# DATA LOADING
# ============================================================

def load_ground_truth(path: Path) -> list[dict]:
    data = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            query_id = str(item["id"])
            video_id = str(item["video_id"])

            if item.get("loai_truy_van") == "chuoi_su_kien":
                events = item.get("cac_giai_doan", [])

                if not events:
                    raise ValueError(
                        f"Query {query_id} là chuỗi_sự_kiện "
                        "nhưng không có cac_giai_doan."
                    )

                gt_frame_idx = int(
                    events[0]["frame_start"]
                )

            else:
                gt_frame_idx = int(
                    item["frame_start"]
                )

            data.append(
                {
                    "query_id": query_id,
                    "gt_video_id": video_id,
                    "gt_frame_idx": gt_frame_idx,
                    "frame_tolerance": 5,
                    "raw_item": item,
                }
            )

    return data


def load_frame_map(path: Path) -> pd.DataFrame:
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
            f"frame_map thiếu cột: {sorted(missing)}"
        )

    return frame_map


def build_frame_lookup(
    frame_map: pd.DataFrame,
) -> dict[tuple[str, int], dict]:
    lookup = {}

    for row in frame_map.itertuples(index=False):
        lookup[
            (
                str(row.video_id),
                int(row.n),
            )
        ] = {
            "frame_idx": int(row.frame_idx),
            "pts_time": float(row.pts_time),
            "fps": float(row.fps),
        }

    return lookup


# ============================================================
# STAGE 1 — VIDEO-LEVEL WEIGHTED RRF
# ============================================================

def fuse_video_rrf(
    b_indices: np.ndarray,
    b_scores: np.ndarray,
    b_ids: list[tuple[str, int]],
    l_indices: np.ndarray,
    l_scores: np.ndarray,
    l_ids: list[tuple[str, int]],
) -> list[tuple[str, dict]]:
    """
    Video-level Weighted RRF.

    Mỗi source bắt đầu bằng frame ranking:

        frame rank
            ↓
        deduplicate theo video
            ↓
        video rank
            ↓
        Weighted RRF

    Mỗi video nhận tối đa:
        - 1 contribution từ CLIP-B
        - 1 contribution từ CLIP-L

    Tuy nhiên toàn bộ frame xuất hiện trong source top-500
    vẫn được giữ lại làm provenance.
    """

    video_scores = {}

    def add_source(
        indices,
        scores,
        ids,
        source,
        weight,
        rank_field,
    ):
        seen_videos = set()
        video_rank = 0

        for frame_rank, vector_id in enumerate(
            indices,
            start=1,
        ):
            vector_id = int(vector_id)

            if vector_id < 0:
                continue

            video_id, n = ids[vector_id]
            video_id = str(video_id)

            if video_id not in video_scores:
                video_scores[video_id] = {
                    "rrf_score": 0.0,
                    "best_b_rank": None,
                    "best_l_rank": None,
                    "frames": [],
                }

            info = video_scores[video_id]

            info["frames"].append(
                {
                    "n": int(n),
                    "source": source,
                    "source_rank": frame_rank,
                    "source_score": float(
                        scores[frame_rank - 1]
                    ),
                }
            )

            if video_id in seen_videos:
                continue

            seen_videos.add(video_id)
            video_rank += 1

            info[rank_field] = video_rank

            info["rrf_score"] += (
                weight
                / (RRF_K + video_rank)
            )

    add_source(
        b_indices,
        b_scores,
        b_ids,
        "clip_b",
        W_B,
        "best_b_rank",
    )

    add_source(
        l_indices,
        l_scores,
        l_ids,
        "clip_l",
        W_L,
        "best_l_rank",
    )

    return sorted(
        video_scores.items(),
        key=lambda item: item[1]["rrf_score"],
        reverse=True,
    )


# ============================================================
# STAGE 1 OUTPUT — TOP 300 VIDEOS
# ============================================================

def select_top300_videos(
    ranked_videos: list[tuple[str, dict]],
) -> list[dict]:
    """
    Chọn đúng Top-300 VIDEO candidates.

    Đây là candidate pool của Stage 1.
    Không biến 300 video thành 300 frame.
    """

    results = []

    for rank, (
        video_id,
        info,
    ) in enumerate(
        ranked_videos[:K_FUSED],
        start=1,
    ):
        results.append(
            {
                "video_id": str(video_id),
                "rank_video": rank,
                "score_fused": float(
                    info["rrf_score"]
                ),
                "best_b_rank": info["best_b_rank"],
                "best_l_rank": info["best_l_rank"],
                "frames": info["frames"],
            }
        )

    return results


# ============================================================
# STAGE 2 — TOP 50 FRAME COVERAGE
# ============================================================

def build_frame_candidate(
    video_id: str,
    video_rank: int,
    info: dict,
    frame: dict,
    frame_lookup: dict[tuple[str, int], dict],
    stage: str,
    rank_final: int | None = None,
) -> dict | None:
    """
    Tạo CandidateRecord Schema 1.1 từ một frame.
    """

    key = (
        str(video_id),
        int(frame["n"]),
    )

    meta = frame_lookup.get(key)

    if meta is None:
        return None

    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": None,
        "event_id": None,

        "video_id": str(video_id),

        # Internal keyframe number.
        "n": int(frame["n"]),

        # Official frame identity.
        "frame_idx": int(meta["frame_idx"]),
        "pts_time": float(meta["pts_time"]),

        "stage": stage,

        "source_hits": [
            frame["source"],
        ],

        "source_ranks": {
            frame["source"]: int(
                frame["source_rank"]
            ),
        },

        "scores": {
            frame["source"]: float(
                frame["source_score"]
            ),
        },

        "score_fused": float(
            info["rrf_score"]
        ),

        "rank_video": int(video_rank),

        "rank_final": rank_final,

        "status": "ok",
        "error": None,
    }


def extract_top50_coverage_candidates(
    top300_videos: list[dict],
    frame_lookup: dict[tuple[str, int], dict],
) -> list[dict]:
    """
    Stage 2:

        Top-300 VIDEO
              ↓
        mỗi video chọn 1 frame đại diện
              ↓
        dedup frame/video
              ↓
        Top-50 FRAME candidates

    Frame đại diện được chọn theo source_rank tốt nhất.

    Nếu một video có nhiều frame từ B/L:
        source_rank nhỏ nhất thắng.

    Như vậy mỗi video đóng góp tối đa 1 frame
    vào coverage pool.
    """

    candidates = []

    for video in top300_videos:
        video_id = str(
            video["video_id"]
        )

        video_rank = int(
            video["rank_video"]
        )

        info = {
            "rrf_score": video["score_fused"],
        }

        frames = video["frames"]

        if not frames:
            continue

        # Chọn frame có source rank tốt nhất.
        best_frame = min(
            frames,
            key=lambda frame: (
                int(frame["source_rank"]),
                -float(frame["source_score"]),
            ),
        )

        candidate = build_frame_candidate(
            video_id=video_id,
            video_rank=video_rank,
            info=info,
            frame=best_frame,
            frame_lookup=frame_lookup,
            stage="coverage",
        )

        if candidate is None:
            continue

        candidates.append(candidate)

        if len(candidates) >= K_COVERAGE:
            break

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        candidate["rank_final"] = rank

    return candidates


# ============================================================
# STAGE 3 — TOP 12 READER
# ============================================================

def select_top12_reader_candidates(
    candidates_50: list[dict],
) -> list[dict]:
    """
    Stage 3:

        Top-50 frame candidates
                ↓
        Top-12 reader candidates
    """

    candidates_12 = []

    for rank, candidate in enumerate(
        candidates_50[:K_READER],
        start=1,
    ):
        reader_candidate = dict(candidate)

        reader_candidate["stage"] = "reader"
        reader_candidate["rank_final"] = rank

        candidates_12.append(
            reader_candidate
        )

    return candidates_12


# ============================================================
# RECALL
# ============================================================

def frame_hits_event(
    candidate: dict,
    target_video: str,
    frame_start: int,
    frame_end: int,
    tolerance: int,
) -> bool:
    return (
        str(candidate["video_id"])
        == target_video
        and
        frame_start - tolerance
        <= int(candidate["frame_idx"])
        <= frame_end + tolerance
    )


def video_pool_hits_query(
    videos: list[dict],
    gt: dict,
) -> bool:
    """
    Recall@300.

    Stage 1 chỉ hỏi:

        GT video có nằm trong Top-300 video không?

    Không yêu cầu frame representative phải chạm GT.
    """

    target_video = str(
        gt["gt_video_id"]
    )

    return any(
        str(video["video_id"])
        == target_video
        for video in videos
    )


def frame_candidates_hit_query(
    candidates: list[dict],
    gt: dict,
) -> bool:
    """
    Recall cho Stage 2 / Stage 3.

    Query đơn:
        đúng video + frame chạm GT range.

    Chuỗi sự kiện:
        tất cả event phải được tìm thấy.
    """

    target_video = str(
        gt["gt_video_id"]
    )

    tolerance = int(
        gt.get("frame_tolerance", 5)
    )

    raw = gt["raw_item"]

    if raw.get(
        "loai_truy_van"
    ) == "chuoi_su_kien":

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

    frame_start = int(
        raw.get(
            "frame_start",
            gt["gt_frame_idx"],
        )
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


def calculate_video_recall(
    results: dict[str, list[dict]],
    ground_truths: dict[str, dict],
) -> float:
    if not ground_truths:
        return 0.0

    hits = 0

    for query_id, gt in ground_truths.items():
        if video_pool_hits_query(
            results.get(query_id, []),
            gt,
        ):
            hits += 1

    return hits / len(ground_truths)


def calculate_frame_recall(
    results: dict[str, list[dict]],
    ground_truths: dict[str, dict],
) -> float:
    if not ground_truths:
        return 0.0

    hits = 0

    for query_id, gt in ground_truths.items():
        if frame_candidates_hit_query(
            results.get(query_id, []),
            gt,
        ):
            hits += 1

    return hits / len(ground_truths)


# ============================================================
# LATENCY
# ============================================================

def percentile(
    values: list[float],
    p: float,
) -> float:
    if not values:
        return 0.0

    return float(
        np.percentile(
            np.asarray(values),
            p,
        )
    )


# ============================================================
# MAIN
# ============================================================

def run_profiling() -> None:

    print(
        "[QA-R1] Bắt đầu multi-stage retrieval funnel...",
        flush=True,
    )

    # ========================================================
    # 1. LOAD FAISS
    # ========================================================

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

    if (
        index_b.ntotal
        != EXPECTED_VECTORS
    ):
        raise ValueError(
            "CLIP-B ntotal không khớp "
            "177,321."
        )

    if (
        index_l.ntotal
        != EXPECTED_VECTORS
    ):
        raise ValueError(
            "CLIP-L ntotal không khớp "
            "177,321."
        )

    if index_b.d != CLIP_B_DIMENSION:
        raise ValueError(
            "CLIP-B dimension không đúng."
        )

    if index_l.d != CLIP_L_DIMENSION:
        raise ValueError(
            "CLIP-L dimension không đúng."
        )

    # ========================================================
    # 2. LOAD ID MAPPING
    # ========================================================

    print(
        "[QA-R1] Đọc mapping IDs...",
        flush=True,
    )

    b_df = pd.read_parquet(
        INDEX_B_IDS_PATH
    )

    l_df = pd.read_parquet(
        INDEX_L_IDS_PATH
    )

    if len(b_df) != index_b.ntotal:
        raise ValueError(
            "CLIP-B mapping không khớp index."
        )

    if len(l_df) != index_l.ntotal:
        raise ValueError(
            "CLIP-L mapping không khớp index."
        )

    b_ids = [
        (
            str(video_id),
            int(n),
        )
        for video_id, n in zip(
            b_df["video_id"],
            b_df["n"],
        )
    ]

    l_ids = [
        (
            str(video_id),
            int(n),
        )
        for video_id, n in zip(
            l_df["video_id"],
            l_df["n"],
        )
    ]

    # ========================================================
    # 3. FRAME MAP
    # ========================================================

    print(
        "[QA-R1] Đọc frame_map...",
        flush=True,
    )

    frame_map = load_frame_map(
        FRAME_MAP_PATH
    )

    frame_lookup = build_frame_lookup(
        frame_map
    )

    # ========================================================
    # 4. GROUND TRUTH
    # ========================================================

    print(
        "[QA-R1] Đọc dev...",
        flush=True,
    )

    gt_data = load_ground_truth(
        DEV_PATH
    )

    ground_truths = {
        item["query_id"]: item
        for item in gt_data
    }

    # ========================================================
    # 5. QUERY EMBEDDINGS
    # ========================================================

    print(
        "[QA-R1] Đọc query embeddings...",
        flush=True,
    )

    query_b = np.load(
        QUERY_B_PATH
    ).astype(np.float32)

    query_l = np.load(
        QUERY_L_PATH
    ).astype(np.float32)

    if query_b.ndim != 2:
        raise ValueError(
            "CLIP-B query embeddings phải 2D."
        )

    if query_l.ndim != 2:
        raise ValueError(
            "CLIP-L query embeddings phải 2D."
        )

    if query_b.shape[1] != CLIP_B_DIMENSION:
        raise ValueError(
            "CLIP-B query dimension sai."
        )

    if query_l.shape[1] != CLIP_L_DIMENSION:
        raise ValueError(
            "CLIP-L query dimension sai."
        )

    if len(query_b) != len(gt_data):
        raise ValueError(
            "Số CLIP-B query không khớp dev."
        )

    if len(query_l) != len(gt_data):
        raise ValueError(
            "Số CLIP-L query không khớp dev."
        )

    # ========================================================
    # 6. METRIC STORAGE
    # ========================================================

    stage_latencies = {
        "source_retrieval_ms": [],
        "video_fusion_ms": [],
        "top300_ms": [],
        "top50_ms": [],
        "top12_ms": [],
        "total_ms": [],
    }

    recall_300_results = {}
    recall_50_results = {}
    recall_12_results = {}

    all_query_records = []

    # ========================================================
    # 7. QUERY LOOP
    # ========================================================

    for idx, item in enumerate(
        gt_data
    ):

        query_id = item["query_id"]

        total_start = time.perf_counter()

        # ----------------------------------------------------
        # STAGE 0 — SOURCE RETRIEVAL
        # ----------------------------------------------------

        start = time.perf_counter()

        b_scores, b_indices = index_b.search(
            query_b[idx : idx + 1],
            K_SOURCE,
        )

        l_scores, l_indices = index_l.search(
            query_l[idx : idx + 1],
            K_SOURCE,
        )

        source_elapsed = (
            time.perf_counter()
            - start
        ) * 1000.0

        stage_latencies[
            "source_retrieval_ms"
        ].append(
            source_elapsed
        )

        # ----------------------------------------------------
        # STAGE 1 — VIDEO FUSION
        # ----------------------------------------------------

        start = time.perf_counter()

        ranked_videos = fuse_video_rrf(
            b_indices[0],
            b_scores[0],
            b_ids,
            l_indices[0],
            l_scores[0],
            l_ids,
        )

        fusion_elapsed = (
            time.perf_counter()
            - start
        ) * 1000.0

        stage_latencies[
            "video_fusion_ms"
        ].append(
            fusion_elapsed
        )

        # ----------------------------------------------------
        # STAGE 1 OUTPUT — TOP 300 VIDEOS
        # ----------------------------------------------------

        start = time.perf_counter()

        top300_videos = select_top300_videos(
            ranked_videos
        )

        top300_elapsed = (
            time.perf_counter()
            - start
        ) * 1000.0

        stage_latencies[
            "top300_ms"
        ].append(
            top300_elapsed
        )

        # ----------------------------------------------------
        # STAGE 2 — TOP 50 FRAME COVERAGE
        # ----------------------------------------------------

        start = time.perf_counter()

        candidates_50 = (
            extract_top50_coverage_candidates(
                top300_videos,
                frame_lookup,
            )
        )

        top50_elapsed = (
            time.perf_counter()
            - start
        ) * 1000.0

        stage_latencies[
            "top50_ms"
        ].append(
            top50_elapsed
        )

        # ----------------------------------------------------
        # STAGE 3 — TOP 12 READER
        # ----------------------------------------------------

        start = time.perf_counter()

        candidates_12 = (
            select_top12_reader_candidates(
                candidates_50
            )
        )

        top12_elapsed = (
            time.perf_counter()
            - start
        ) * 1000.0

        stage_latencies[
            "top12_ms"
        ].append(
            top12_elapsed
        )

        # ----------------------------------------------------
        # RECALL DATA
        # ----------------------------------------------------

        recall_300_results[
            query_id
        ] = top300_videos

        recall_50_results[
            query_id
        ] = candidates_50

        recall_12_results[
            query_id
        ] = candidates_12

        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        total_elapsed = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        stage_latencies[
            "total_ms"
        ].append(
            total_elapsed
        )

        # ----------------------------------------------------
        # QUERY RECORD
        # ----------------------------------------------------

        all_query_records.append(
            {
                "query_id": query_id,

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
                    top300_videos
                ),

                "num_candidates_50": len(
                    candidates_50
                ),

                "num_candidates_12": len(
                    candidates_12
                ),

                "latency_ms": {
                    "source_retrieval":
                        source_elapsed,

                    "video_fusion":
                        fusion_elapsed,

                    "top300":
                        top300_elapsed,

                    "top50":
                        top50_elapsed,

                    "top12":
                        top12_elapsed,

                    "total":
                        total_elapsed,
                },
            }
        )

    # ========================================================
    # 8. RECALL
    # ========================================================

    recall_300 = calculate_video_recall(
        recall_300_results,
        ground_truths,
    )

    recall_50 = calculate_frame_recall(
        recall_50_results,
        ground_truths,
    )

    recall_12 = calculate_frame_recall(
        recall_12_results,
        ground_truths,
    )

    # ========================================================
    # 9. LATENCY REPORT
    # ========================================================

    latency_report = {}

    for stage, values in stage_latencies.items():
        latency_report[stage] = {
            "p50": percentile(
                values,
                50,
            ),
            "p95": percentile(
                values,
                95,
            ),
            "mean": float(
                np.mean(values)
            ) if values else 0.0,
        }

    # ========================================================
    # 10. VALIDATION
    # ========================================================

    funnel_recall_monotonic = (
        recall_300
        >= recall_50
        >= recall_12
    )

    latency_pass = (
        latency_report[
            "total_ms"
        ]["p95"]
        <= BUDGET_P95_MS
    )

    status = (
        "PASSED"
        if (
            funnel_recall_monotonic
            and latency_pass
        )
        else "FAILED"
    )

    # ========================================================
    # 11. REPORT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "BÁO CÁO KẾT QUẢ TASK QA-R1"
    )

    print(
        "=" * 70
    )

    print(
        f"Tổng số query : {len(gt_data)}"
    )

    print(
        f"FAISS vectors : {index_b.ntotal:,}"
    )

    print("-" * 70)

    print(
        f"K_SOURCE   : {K_SOURCE} frames/source"
    )

    print(
        f"K_FUSED    : {K_FUSED} videos"
    )

    print(
        f"K_COVERAGE : {K_COVERAGE} frames"
    )

    print(
        f"K_READER   : {K_READER} frames"
    )

    print("-" * 70)

    print(
        f"Recall@300 videos : "
        f"{recall_300:.4f}"
    )

    print(
        f"Recall@50 frames  : "
        f"{recall_50:.4f}"
    )

    print(
        f"Recall@12 frames  : "
        f"{recall_12:.4f}"
    )

    print(
        f"Funnel recall monotonic : "
        f"{'PASS' if funnel_recall_monotonic else 'FAIL'}"
    )

    print("-" * 70)

    for stage, values in latency_report.items():
        print(
            f"{stage:<24}"
            f"p50={values['p50']:.2f} ms | "
            f"p95={values['p95']:.2f} ms"
        )

    print("-" * 70)

    print(
        f"Total p95 budget : "
        f"{BUDGET_P95_MS:.2f} ms -> "
        f"{'PASSED' if latency_pass else 'FAILED'}"
    )

    print(
        f"QA-R1 Status : {status}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 12. MANIFEST
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "schema_version": SCHEMA_VERSION,

        "task": "QA-R1",

        "retrieval_method": (
            "n02_video_level_weighted_rrf"
            "_multi_stage_funnel"
        ),

        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),

        "dataset_info": {
            "num_queries": len(gt_data),
            "index_ntotal": int(
                index_b.ntotal
            ),
            "clip_b_dim": CLIP_B_DIMENSION,
            "clip_l_dim": CLIP_L_DIMENSION,
        },

        "funnel": {
            "K_source": K_SOURCE,
            "K_fused_videos": K_FUSED,
            "K_coverage_frames": K_COVERAGE,
            "K_reader_frames": K_READER,
        },

        "rrf": {
            "rrf_k": RRF_K,
            "weight_clip_b": W_B,
            "weight_clip_l": W_L,
        },

        "metrics": {
            "recall_at_300_videos":
                recall_300,

            "recall_at_50_frames":
                recall_50,

            "recall_at_12_frames":
                recall_12,

            "funnel_recall_monotonic":
                funnel_recall_monotonic,

            "latency_ms":
                latency_report,
        },

        "budget": {
            "p95_ms":
                BUDGET_P95_MS,

            "passed":
                latency_pass,
        },

        "status":
            status.lower(),

        "query_records":
            all_query_records,
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
        f"[QA-R1] Đã ghi manifest: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    run_profiling()