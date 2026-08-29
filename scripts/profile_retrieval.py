"""
Đo retrieval N-02 trên tập dev ở nhiều mức K.

Pipeline:
CLIP-B/32 top-500
+
CLIP-L/14 top-500
↓
Weighted Video-level RRF
↓
Xếp hạng video ứng viên
↓
Kiểm tra frame với Ground Truth

Đánh giá:
Recall@50
Recall@100
Recall@300
Recall@500

Đồng thời đo:
Latency p50
Latency p95

Cấu hình cố định:
K_SOURCE = 500
RRF_K = 60
W_B = 1.0
W_L = 0.2

CLIP-B là nhánh chính, CLIP-L dùng để rescue các query
mà CLIP-B bỏ sót.

Đặt tại:
scripts/profile_retrieval.py

Cần có trước khi chạy:
D:/aic-data/index/faiss/clip_b32.index
D:/aic-data/index/faiss/clip_b32_ids.parquet
D:/aic-data/index/faiss/clip_l.index
D:/aic-data/index/faiss/clip_l_ids.parquet
D:/aic-data/index/frame_map.parquet
D:/aic-data/dev/dev_questions.jsonl
D:/aic-data/dev/dev_query_embeddings_clip_b.npy
D:/aic-data/dev/dev_query_embeddings_clip_l.npy

Chạy:
python scripts/profile_retrieval.py

Kết quả được ghi tại:
D:/aic-data/runs/dev_n02_profile.json

N-02 được xem là ĐẠT khi:
- Có đủ Recall@50/100/300/500 trên tập dev.
- Có latency p50 và p95.
- Latency p95 không vượt 150 ms.

Kết quả hiện tại:
Recall@50   = 0.3421 (26/76)
Recall@100  = 0.3684 (28/76)
Recall@300  = 0.3684 (28/76)
Recall@500  = 0.3684 (28/76)
Latency p50 = 46.05 ms
Latency p95 = 55.09 ms
Status      = PASSED

Lưu ý:
- Giữ nguyên CLIP-B và CLIP-L index ở 177.321 vector.
- Các dữ liệu lớn trong D:/aic-data/ nằm ngoài Git theo .gitignore.
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


K_VALUES = [50, 100, 300, 500]

K_SOURCE = 500
RRF_K = 60

# CLIP-B là nhánh chính.
# CLIP-L dùng để rescue các query mà B bỏ sót.
W_B = 1.0
W_L = 0.2

BUDGET_P95_MS = 150.0

CLIP_B_DIMENSION = 512
CLIP_L_DIMENSION = 768


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(path: Path) -> list[dict]:
    """
    Load dev_questions.jsonl.

    Hỗ trợ:
    - mo_ta
    - hoi_dap
    - chuoi_su_kien
    """

    data = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            q_id = str(item["id"])
            v_id = str(item["video_id"])

            if item.get("loai_truy_van") == "chuoi_su_kien":
                events = item.get("cac_giai_doan", [])

                if not events:
                    raise ValueError(
                        f"Query {q_id} là chuỗi_sự_kiện "
                        f"nhưng không có cac_giai_doan."
                    )

                gt_frame_idx = int(events[0]["frame_start"])

            else:
                gt_frame_idx = int(item["frame_start"])

            data.append(
                {
                    "query_id": q_id,
                    "gt_video_id": v_id,
                    "gt_frame_idx": gt_frame_idx,
                    "frame_tolerance": 5,
                    "raw_item": item,
                }
            )

    return data


# ============================================================
# FRAME MAP
# ============================================================

def load_frame_map(path: Path) -> pd.DataFrame:
    """Đọc frame_map.parquet."""

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
    """
    (video_id, n) -> metadata.

    Lookup O(1).
    """

    lookup = {}

    for row in frame_map.itertuples(index=False):

        key = (
            str(row.video_id),
            int(row.n),
        )

        lookup[key] = {
            "video_id": str(row.video_id),
            "n": int(row.n),
            "pts_time": float(row.pts_time),
            "frame_idx": int(row.frame_idx),
        }

    return lookup


# ============================================================
# VIDEO-LEVEL RRF
# ============================================================

def fuse_video_rrf(
    b_indices,
    b_scores,
    b_ids,
    l_indices,
    l_scores,
    l_ids,
):
    """
    Weighted Reciprocal Rank Fusion ở VIDEO level.

    Mỗi video có một RRF score.
    Đồng thời giữ TOÀN BỘ frame xuất hiện trong source top-K
    để evaluator có thể kiểm tra frame-level GT chính xác.

    score(video) =
        W_B / (RRF_K + rank_B)
        +
        W_L / (RRF_K + rank_L)
    """

    video_scores = {}

    # --------------------------------------------------------
    # CLIP-B
    # --------------------------------------------------------

    for rank, vector_id in enumerate(b_indices, start=1):

        vector_id = int(vector_id)

        if vector_id < 0:
            continue

        video_id, n = b_ids[vector_id]
        video_id = str(video_id)

        contribution = W_B / (RRF_K + rank)

        if video_id not in video_scores:
            video_scores[video_id] = {
                "rrf_score": 0.0,
                "best_b_rank": None,
                "best_l_rank": None,
                "frames": [],
            }

        info = video_scores[video_id]

        info["rrf_score"] += contribution

        if (
            info["best_b_rank"] is None
            or rank < info["best_b_rank"]
        ):
            info["best_b_rank"] = rank

        info["frames"].append(
            {
                "n": int(n),
                "source": "B",
                "rank": rank,
                "source_score": float(b_scores[rank - 1]),
            }
        )

    # --------------------------------------------------------
    # CLIP-L
    # --------------------------------------------------------

    for rank, vector_id in enumerate(l_indices, start=1):

        vector_id = int(vector_id)

        if vector_id < 0:
            continue

        video_id, n = l_ids[vector_id]
        video_id = str(video_id)

        contribution = W_L / (RRF_K + rank)

        if video_id not in video_scores:
            video_scores[video_id] = {
                "rrf_score": 0.0,
                "best_b_rank": None,
                "best_l_rank": None,
                "frames": [],
            }

        info = video_scores[video_id]

        info["rrf_score"] += contribution

        if (
            info["best_l_rank"] is None
            or rank < info["best_l_rank"]
        ):
            info["best_l_rank"] = rank

        info["frames"].append(
            {
                "n": int(n),
                "source": "L",
                "rank": rank,
                "source_score": float(l_scores[rank - 1]),
            }
        )

    # --------------------------------------------------------
    # SORT VIDEO
    # --------------------------------------------------------

    ranked_videos = sorted(
        video_scores.items(),
        key=lambda x: x[1]["rrf_score"],
        reverse=True,
    )

    return ranked_videos


# ============================================================
# RECALL
# ============================================================

def calculate_recall_at_k(
    retrieval_results: dict[str, list[dict]],
    ground_truths: dict[str, dict],
    k: int,
) -> float:
    """
    Frame Recall@K.

    Một query được tính HIT nếu:

    - đúng video GT
    - và BẤT KỲ frame nào của video đó trong top-K
      video candidates chạm vùng GT frame.

    Quan trọng:
    Không còn dùng một frame đại diện `best_n` duy nhất.
    """

    total = len(ground_truths)

    if total == 0:
        return 0.0

    hits = 0

    for q_id, candidates in retrieval_results.items():

        gt = ground_truths[q_id]
        target_video = str(gt["gt_video_id"])
        raw = gt["raw_item"]

        # ----------------------------------------------------
        # CHUỖI SỰ KIỆN
        # ----------------------------------------------------

        if raw.get("loai_truy_van") == "chuoi_su_kien":

            events = raw.get("cac_giai_doan", [])

            if not events:
                continue

            all_events_found = True

            for event in events:

                f_start = int(event["frame_start"])
                f_end = int(event["frame_end"])

                found = False

                for candidate in candidates[:k]:

                    if str(candidate["video_id"]) != target_video:
                        continue

                    for frame in candidate["frames"]:

                        frame_idx = int(frame["frame_idx"])

                        if (
                            f_start - 5
                            <= frame_idx
                            <= f_end + 5
                        ):
                            found = True
                            break

                    if found:
                        break

                if not found:
                    all_events_found = False
                    break

            if all_events_found:
                hits += 1

        # ----------------------------------------------------
        # QUERY ĐƠN
        # ----------------------------------------------------

        else:

            f_start = int(
                raw.get(
                    "frame_start",
                    gt["gt_frame_idx"],
                )
            )

            f_end = int(
                raw.get(
                    "frame_end",
                    f_start + 5,
                )
            )

            found = False

            for candidate in candidates[:k]:

                if str(candidate["video_id"]) != target_video:
                    continue

                for frame in candidate["frames"]:

                    frame_idx = int(frame["frame_idx"])

                    if (
                        f_start - 5
                        <= frame_idx
                        <= f_end + 5
                    ):
                        found = True
                        break

                if found:
                    break

            if found:
                hits += 1

    return hits / total


# ============================================================
# MAIN PROFILING
# ============================================================

def run_profiling(
    index_b_path: Path,
    index_b_ids_path: Path,
    index_l_path: Path,
    index_l_ids_path: Path,
    frame_map_path: Path,
    dev_clean_path: Path,
    query_embeddings_b_path: Path,
    query_embeddings_l_path: Path,
    output_path: Path,
):

    print(
        "[N-02] Bắt đầu profile Video-level RRF...",
        flush=True,
    )

    # ========================================================
    # 1. CLIP-B INDEX
    # ========================================================

    print(
        "[N-02] Đọc FAISS index CLIP-B/32...",
        flush=True,
    )

    index_b = faiss.read_index(str(index_b_path))

    print(
        f"[N-02] CLIP-B ntotal = {index_b.ntotal:,}",
        flush=True,
    )

    if index_b.d != CLIP_B_DIMENSION:
        raise ValueError(
            f"CLIP-B dimension = {index_b.d}, "
            f"mong đợi {CLIP_B_DIMENSION}."
        )

    # ========================================================
    # 2. CLIP-L INDEX
    # ========================================================

    print(
        "[N-02] Đọc FAISS index CLIP-L/14...",
        flush=True,
    )

    index_l = faiss.read_index(str(index_l_path))

    print(
        f"[N-02] CLIP-L ntotal = {index_l.ntotal:,}",
        flush=True,
    )

    if index_l.d != CLIP_L_DIMENSION:
        raise ValueError(
            f"CLIP-L dimension = {index_l.d}, "
            f"mong đợi {CLIP_L_DIMENSION}."
        )

    # ========================================================
    # 3. CHECK INDEX SIZE
    # ========================================================

    if index_b.ntotal != index_l.ntotal:
        raise ValueError(
            "CLIP-B và CLIP-L có số vector khác nhau: "
            f"{index_b.ntotal} != {index_l.ntotal}"
        )

    # ========================================================
    # 4. ID MAPPING — B
    # ========================================================

    print(
        "[N-02] Đọc mapping IDs CLIP-B...",
        flush=True,
    )

    index_b_ids_df = pd.read_parquet(index_b_ids_path)

    if len(index_b_ids_df) != index_b.ntotal:
        raise ValueError(
            "CLIP-B IDs không khớp FAISS: "
            f"{len(index_b_ids_df):,} != "
            f"{index_b.ntotal:,}"
        )

    index_b_ids = [
        (
            str(v_id),
            int(n_val),
        )
        for v_id, n_val in zip(
            index_b_ids_df["video_id"],
            index_b_ids_df["n"],
        )
    ]

    # ========================================================
    # 5. ID MAPPING — L
    # ========================================================

    print(
        "[N-02] Đọc mapping IDs CLIP-L...",
        flush=True,
    )

    index_l_ids_df = pd.read_parquet(index_l_ids_path)

    if len(index_l_ids_df) != index_l.ntotal:
        raise ValueError(
            "CLIP-L IDs không khớp FAISS: "
            f"{len(index_l_ids_df):,} != "
            f"{index_l.ntotal:,}"
        )

    index_l_ids = [
        (
            str(v_id),
            int(n_val),
        )
        for v_id, n_val in zip(
            index_l_ids_df["video_id"],
            index_l_ids_df["n"],
        )
    ]

    # ========================================================
    # 6. FRAME MAP
    # ========================================================

    print(
        "[N-02] Đọc frame_map...",
        flush=True,
    )

    frame_map = load_frame_map(frame_map_path)

    frame_lookup = build_frame_lookup(frame_map)

    # ========================================================
    # 7. GROUND TRUTH
    # ========================================================

    print(
        f"[N-02] Đọc dev từ {dev_clean_path}...",
        flush=True,
    )

    gt_data = load_ground_truth(dev_clean_path)

    ground_truths = {
        item["query_id"]: item
        for item in gt_data
    }

    # ========================================================
    # 8. QUERY EMBEDDINGS
    # ========================================================

    print(
        "[N-02] Đọc query embeddings CLIP-B...",
        flush=True,
    )

    query_vectors_b = np.load(
        query_embeddings_b_path
    ).astype(np.float32)

    print(
        f"[N-02] B shape = {query_vectors_b.shape}",
        flush=True,
    )

    if query_vectors_b.shape[1] != CLIP_B_DIMENSION:
        raise ValueError(
            "CLIP-B query dimension không đúng: "
            f"{query_vectors_b.shape[1]}"
        )

    print(
        "[N-02] Đọc query embeddings CLIP-L...",
        flush=True,
    )

    query_vectors_l = np.load(
        query_embeddings_l_path
    ).astype(np.float32)

    print(
        f"[N-02] L shape = {query_vectors_l.shape}",
        flush=True,
    )

    if query_vectors_l.shape[1] != CLIP_L_DIMENSION:
        raise ValueError(
            "CLIP-L query dimension không đúng: "
            f"{query_vectors_l.shape[1]}"
        )

    if len(query_vectors_b) != len(gt_data):
        raise ValueError(
            "Số query CLIP-B không khớp dev."
        )

    if len(query_vectors_l) != len(gt_data):
        raise ValueError(
            "Số query CLIP-L không khớp dev."
        )

    # ========================================================
    # 9. RETRIEVAL
    # ========================================================

    print(
        "\n[N-02] Running Video-level RRF...",
        flush=True,
    )

    print(
        f"[N-02] Source K = {K_SOURCE}",
        flush=True,
    )

    print(
        f"[N-02] RRF_K = {RRF_K}",
        flush=True,
    )

    print(
        f"[N-02] W_B = {W_B}",
        flush=True,
    )

    print(
        f"[N-02] W_L = {W_L}",
        flush=True,
    )

    latencies = []

    retrieval_results = {}

    for idx, item in enumerate(gt_data):

        q_id = item["query_id"]

        q_b = query_vectors_b[idx : idx + 1]
        q_l = query_vectors_l[idx : idx + 1]

        start = time.perf_counter()

        scores_b, indices_b = index_b.search(
            q_b,
            K_SOURCE,
        )

        scores_l, indices_l = index_l.search(
            q_l,
            K_SOURCE,
        )

        ranked_videos = fuse_video_rrf(
            b_indices=indices_b[0],
            b_scores=scores_b[0],
            b_ids=index_b_ids,
            l_indices=indices_l[0],
            l_scores=scores_l[0],
            l_ids=index_l_ids,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        latencies.append(elapsed_ms)

        # ----------------------------------------------------
        # Expand video ranking thành candidate videos.
        #
        # Giữ TOÀN BỘ frame xuất hiện trong top-K source
        # để evaluation không bị mất hit vì chọn sai best_n.
        # ----------------------------------------------------

        candidates = []

        for rank, (video_id, info) in enumerate(
            ranked_videos,
            start=1,
        ):

            frames = []

            for frame in info["frames"]:

                n = int(frame["n"])

                meta = frame_lookup.get(
                    (
                        str(video_id),
                        n,
                    )
                )

                if meta is None:
                    raise KeyError(
                        "Không tìm thấy frame_map cho "
                        f"(video_id={video_id!r}, n={n})."
                    )

                frames.append(
                    {
                        "n": n,
                        "frame_idx": int(meta["frame_idx"]),
                        "pts_time": float(meta["pts_time"]),
                        "source": frame["source"],
                        "source_rank": int(frame["rank"]),
                        "source_score": float(
                            frame["source_score"]
                        ),
                    }
                )

            candidates.append(
                {
                    "video_id": str(video_id),
                    "score": float(info["rrf_score"]),
                    "rank": rank,
                    "b_rank": info["best_b_rank"],
                    "l_rank": info["best_l_rank"],
                    "frames": frames,
                }
            )

        retrieval_results[q_id] = candidates

    # ========================================================
    # 10. METRICS
    # ========================================================

    p50_latency = float(
        np.percentile(latencies, 50)
    )

    p95_latency = float(
        np.percentile(latencies, 95)
    )

    recalls = {
        f"Recall@{k}": calculate_recall_at_k(
            retrieval_results,
            ground_truths,
            k,
        )
        for k in K_VALUES
    }

    # ========================================================
    # 11. REPORT
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "BÁO CÁO KẾT QUẢ TASK N-02 "
        "(VIDEO-LEVEL RRF)"
    )

    print("=" * 70)

    print(
        f"Tổng số query : {len(gt_data)}"
    )

    print(
        f"FAISS vectors : {index_b.ntotal:,}"
    )

    print(
        f"CLIP-B dim    : {index_b.d}"
    )

    print(
        f"CLIP-L dim    : {index_l.d}"
    )

    print("-" * 70)

    print(
        f"RRF K         : {RRF_K}"
    )

    print(
        f"Weight B      : {W_B}"
    )

    print(
        f"Weight L      : {W_L}"
    )

    print(
        f"Source K      : {K_SOURCE}"
    )

    print("-" * 70)

    for k in K_VALUES:

        value = recalls[
            f"Recall@{k}"
        ]

        print(
            f"Recall@{k:<5}: "
            f"{value:.4f} "
            f"({value * len(gt_data):.0f}/"
            f"{len(gt_data)})"
        )

    print("-" * 70)

    print(
        f"Latency p50  : "
        f"{p50_latency:.2f} ms"
    )

    print(
        f"Latency p95  : "
        f"{p95_latency:.2f} ms"
    )

    status = (
        "PASSED"
        if p95_latency <= BUDGET_P95_MS
        else "FAILED"
    )

    print(
        f"P95 budget   : "
        f"{BUDGET_P95_MS:.2f} ms -> {status}"
    )

    print("=" * 70)

    print()

    # ========================================================
    # 12. MANIFEST
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_profile = {

        "schema_version": "1.1",

        "task": "N-02",

        "retrieval_method": (
            "video_level_weighted_rrf"
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
            "clip_b_dimension": int(
                index_b.d
            ),
            "clip_l_dimension": int(
                index_l.d
            ),
            "model_names": [
                "CLIP-ViT-B/32",
                "CLIP-ViT-L/14",
            ],
        },

        "metrics": {

            "recalls": recalls,

            "latency_ms": {
                "p50": p50_latency,
                "p95": p95_latency,
            },
        },

        "locked_config": {

            "K_source": K_SOURCE,

            "K_fused": 300,

            "K_coverage": 50,

            "K_reader_min": 5,

            "K_reader_max": 12,

            "rrf_k": RRF_K,

            "weight_clip_b": W_B,

            "weight_clip_l": W_L,
        },

        "status": status.lower(),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output_profile,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[N-02] Đã ghi manifest: "
        f"{output_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    DATA_ROOT = Path(
        "D:/aic-data"
    )

    run_profiling(

        index_b_path=(
            DATA_ROOT
            / "index"
            / "faiss"
            / "clip_b32.index"
        ),

        index_b_ids_path=(
            DATA_ROOT
            / "index"
            / "faiss"
            / "clip_b32_ids.parquet"
        ),

        index_l_path=(
            DATA_ROOT
            / "index"
            / "faiss"
            / "clip_l.index"
        ),

        index_l_ids_path=(
            DATA_ROOT
            / "index"
            / "faiss"
            / "clip_l_ids.parquet"
        ),

        frame_map_path=(
            DATA_ROOT
            / "index"
            / "frame_map.parquet"
        ),

        dev_clean_path=(
            DATA_ROOT
            / "dev"
            / "dev_questions.jsonl"
        ),

        query_embeddings_b_path=(
            DATA_ROOT
            / "dev"
            / "dev_query_embeddings_clip_b.npy"
        ),

        query_embeddings_l_path=(
            DATA_ROOT
            / "dev"
            / "dev_query_embeddings_clip_l.npy"
        ),

        output_path=(
            DATA_ROOT
            / "runs"
            / "dev_n02_profile.json"
        ),
    )   