"""
Kho FAISS cho đặc trưng CLIP của toàn bộ keyframe.

Đọc các vector CLIP do BTC cung cấp trong raw/clip-features-32/,
chuyển float16 sang float32, chuẩn hoá độ dài vector và xây dựng
kho tra cứu nhanh, ghi ra index/faiss/clip_b32.index.

THỨ TỰ VECTOR — ĐỌC KỸ, ĐÂY LÀ CHỖ SAI MÀ KHÔNG CÓ TRIỆU CHỨNG.

FAISS chỉ trả về SỐ THỨ TỰ: 0, 1, 2… Nó không biết gì về video_id hay n.
Muốn dịch số 12345 thành "ảnh thứ 57 của L21_V001" thì phải có một bảng
thứ tự, và bảng đó phải khớp TUYỆT ĐỐI với thứ tự lúc nạp vào kho.

Cách sai: quét raw/clip-features-32/ rồi ghép theo thứ tự thư mục trả về,
và ngầm tin rằng nó trùng với thứ tự dòng của frame_map.parquet. Hai vòng
quét độc lập trên hai thư mục khác nhau — không có gì bảo đảm điều đó.
Lệch một dòng thì mọi kết quả tìm kiếm đều trỏ sai ảnh, mà màn hình vẫn
hiện đủ 100 kết quả trông rất bình thường.

Cách làm ở đây:
    1. Thứ tự lấy TỪ frame_map.parquet, không lấy từ thư mục.
    2. Mỗi video kiểm số vector khớp số dòng của video đó trước khi nạp.
    3. Ghi kèm index/faiss/clip_b32_ids.parquet — bảng (video_id, n) theo
       đúng thứ tự đã nạp. Kho tự mô tả được chính nó, không phải đoán.

CÁCH DÙNG (Việc 2 — Thi):
    Dựng một lần:   python scripts/build_faiss.py
    Dùng hằng ngày: search_by_text("...")   (cần Việc 3 xong trước)
"""

from __future__ import annotations

import os
import time

import faiss
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..frame_map import load_frame_map, lookup
from ..paths import CLIP_FEATURES_DIR, FAISS_DIR, clip_features_file

@dataclass(frozen=True)
class Hit:
    video_id: str
    n: int
    score: float
    frame_idx: int
    pts_time: float
    source: str

CLIP_DIMENSION = 512

EXPECTED_ROWS = 177_321
EXPECTED_VIDEOS = 873

INDEX_PATH = FAISS_DIR / "clip_b32.index"
# Bảng thứ tự đi kèm kho. Thiếu tệp này thì không dịch được số thứ tự FAISS
# thành ảnh, và cũng không kiểm được kho có khớp frame_map hay không.
INDEX_IDS_PATH = FAISS_DIR / "clip_b32_ids.parquet"

# Cosine của vector tự nạp so với vector BTC phát. Dưới ngưỡng này nghĩa là
# đã lỡ chuẩn hoá hai lần hoặc đọc nhầm kiểu số.
MIN_SELF_COSINE = 0.999

# Bộ nhớ đệm: kho nặng khoảng 363 MB, đọc lại mỗi truy vấn thì không bao giờ
# đạt mốc 200 mili giây của Việc 2.
_CACHED_INDEX: faiss.Index | None = None
_CACHED_IDS: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Dựng kho
# ---------------------------------------------------------------------------


def _video_row_counts(frame_map: pd.DataFrame) -> list[tuple[str, int]]:
    """
    Danh sách (video_id, số ảnh) theo ĐÚNG thứ tự dòng của frame_map.

    Dùng sort=False: giữ nguyên thứ tự xuất hiện trong bảng, không để pandas
    tự sắp lại theo cách riêng của nó.
    """
    counts = frame_map.groupby("video_id", sort=False).size()
    return [(str(video_id), int(count)) for video_id, count in counts.items()]


def load_clip_features(video_id: str, expected_rows: int) -> np.ndarray:
    """
    Đọc raw/clip-features-32/<video_id>.npy, đổi sang float32, chuẩn hoá.

    Tham số:
        expected_rows: số ảnh của video này theo frame_map. Không khớp là
                       dừng ngay — đây là phép kiểm quan trọng nhất của Việc 2.
    """
    feature_path = clip_features_file(video_id)
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Không thấy đặc trưng CLIP của {video_id} tại {feature_path}. "
            "Đã giải nén clip-features-32 chưa?"
        )

    features = np.load(feature_path)

    if features.ndim != 2 or features.shape[1] != CLIP_DIMENSION:
        raise ValueError(
            f"{feature_path.name} có shape {features.shape}, "
            f"mong đợi (n, {CLIP_DIMENSION})."
        )

    if len(features) != expected_rows:
        raise ValueError(
            f"{video_id}: tệp .npy có {len(features)} vector nhưng frame_map "
            f"ghi {expected_rows} ảnh. Lệch số lượng nghĩa là thứ tự vector "
            "không còn khớp bảng đối chiếu — mọi kết quả tìm kiếm sẽ trỏ sai "
            "ảnh mà không có triệu chứng. Kiểm lại xem đã giải nén đủ chưa."
        )

    features = features.astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError(
            f"{video_id}: đặc trưng CLIP có giá trị NaN hoặc vô hạn. "
            "Tệp .npy hỏng hoặc giải nén dở."
        )

    norms = np.linalg.norm(features, axis=1)
    if (norms == 0).any():
        bad = int(np.argmin(norms))
        raise ValueError(
            f"{video_id}: vector thứ {bad} có độ dài bằng 0, chuẩn hoá sẽ ra "
            "NaN và ảnh đó sẽ không bao giờ khớp câu nào."
        )

    faiss.normalize_L2(features)
    return features


def build_index(strict: bool = True) -> tuple[faiss.Index, pd.DataFrame]:
    """
    Nạp toàn bộ đặc trưng CLIP vào FAISS theo đúng thứ tự của frame_map.

    Trả về:
        (kho FAISS, bảng thứ tự gồm hai cột video_id và n)

    Tham số:
        strict: True thì DỪNG nếu chưa đủ 177.321 vector hoặc 873 video.
    """
    frame_map = load_frame_map()
    video_counts = _video_row_counts(frame_map)
    total_videos = len(video_counts)

    if not total_videos:
        raise FileNotFoundError(
            "frame_map.parquet rỗng. Chạy scripts/build_frame_map.py trước."
        )

    print(f"Nạp đặc trưng CLIP của {total_videos} video từ {CLIP_FEATURES_DIR}")
    print("Thứ tự lấy từ frame_map.parquet, không lấy từ thứ tự thư mục.")

    total_rows = len(frame_map)

    all_vectors = np.empty(
        (total_rows, CLIP_DIMENSION),
        dtype=np.float32,
    )

    loaded_vectors = 0

    for i, (video_id, expected_rows) in enumerate(
        video_counts,
        start=1,
    ):
        features = load_clip_features(
            video_id,
            expected_rows,
        )

        end = loaded_vectors + len(features)
        all_vectors[loaded_vectors:end] = features
        loaded_vectors = end

        if i % 50 == 0 or i == total_videos:
            print(
                f"  {i}/{total_videos} video — "
                f"{loaded_vectors:,} vector"
            )

    # Bảng thứ tự: dòng thứ i của bảng này ứng với vector thứ i trong kho.
    # Vì thứ tự nạp lấy từ frame_map nên đây chính là frame_map cắt hai cột.
    index_ids = frame_map[["video_id", "n"]].reset_index(drop=True)

    problems = check_index(all_vectors, index_ids, total_videos)
    if problems:
        message = "CHƯA ĐẠT: " + "; ".join(problems)
        if strict:
            raise ValueError(message)
        print(message)

    print(f"Dựng IndexFlatIP {CLIP_DIMENSION} chiều…")
    index = faiss.IndexFlatIP(CLIP_DIMENSION)
    index.add(all_vectors)

    verify_self_cosine(index, all_vectors)

    return index, index_ids


def check_index(
    all_vectors: np.ndarray,
    index_ids: pd.DataFrame,
    video_count: int,
) -> list[str]:
    """Soát điều kiện 'Xong khi' của Việc 2. Danh sách rỗng nghĩa là ĐẠT."""
    problems: list[str] = []

    if len(all_vectors) != EXPECTED_ROWS:
        problems.append(f"nạp được {len(all_vectors):,} vector, cần {EXPECTED_ROWS:,}")

    if video_count != EXPECTED_VIDEOS:
        problems.append(f"đếm được {video_count} video, cần {EXPECTED_VIDEOS}")

    if all_vectors.shape[1] != CLIP_DIMENSION:
        problems.append(f"vector {all_vectors.shape[1]} chiều, cần {CLIP_DIMENSION}")

    if all_vectors.dtype != np.float32:
        problems.append(f"kiểu số {all_vectors.dtype}, cần float32")

    if len(index_ids) != len(all_vectors):
        problems.append(
            f"bảng thứ tự {len(index_ids):,} dòng nhưng có {len(all_vectors):,} vector"
        )

    norms = np.linalg.norm(all_vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        problems.append(
            f"độ dài vector không phải 1 (nhỏ nhất {norms.min():.4f}, "
            f"lớn nhất {norms.max():.4f}) — chưa chuẩn hoá hoặc chuẩn hoá hỏng"
        )

    return problems


def verify_self_cosine(index: faiss.Index, all_vectors: np.ndarray) -> None:
    """
    Lấy vài vector vừa nạp đem tra chính kho vừa dựng.

    Vì dữ liệu đã chuẩn hoá float32, sai số số học có thể khiến một
    vector khác có inner product bằng hoặc nhỉnh hơn 1 rất nhỏ.
    Do đó không bắt buộc chính nó phải đứng đúng vị trí #1;
    chỉ cần chính nó nằm trong nhóm kết quả có cosine gần 1.
    """
    rng = np.random.default_rng(2026)

    sample_ids = rng.choice(
        len(all_vectors),
        size=min(30, len(all_vectors)),
        replace=False,
    )

    queries = all_vectors[sample_ids]

    scores, found = index.search(queries, 10)

    tolerance = 1e-4

    missing = []

    for row, position in enumerate(sample_ids):
        own_score = float(
            np.dot(
                all_vectors[position],
                all_vectors[position],
            )
        )

        top_score = float(scores[row, 0])

        # Chính vector phải có cosine gần 1.
        if own_score < MIN_SELF_COSINE:
            missing.append(
                (int(position), own_score, top_score, "self cosine thấp")
            )
            continue

        # Nếu chính nó không nằm trong top 10, chỉ coi là lỗi
        # khi điểm top cao hơn chính nó quá sai số cho phép.
        if int(position) not in found[row]:
            if top_score - own_score > tolerance:
                missing.append(
                    (int(position), own_score, top_score, "self không nằm top 10")
                )

    if missing:
        print("\nDEBUG self-search:")
        for position, own_score, top_score, reason in missing[:3]:
            print(f"  query position: {position}")
            print(f"  reason        : {reason}")
            print(f"  own score     : {own_score:.9f}")
            print(f"  top score     : {top_score:.9f}")

        raise ValueError(
            f"Vector tự tra không nhất quán, ví dụ {missing[0][0]}. "
            "Có thể dữ liệu bị nạp sai hoặc vector chưa được chuẩn hoá đúng."
        )

    worst = float(
        np.min(
            [
                float(
                    np.dot(
                        all_vectors[position],
                        all_vectors[position],
                    )
                )
                for position in sample_ids
            ]
        )
    )

    if worst < MIN_SELF_COSINE:
        raise ValueError(
            f"Cosine của vector với chính nó chỉ {worst:.4f}, "
            f"cần ≥ {MIN_SELF_COSINE}."
        )

    print(
        f"Tự tra 30 vector: đạt, "
        f"cosine tự thân thấp nhất {worst:.6f}"
    )

def write_index(index: faiss.Index, index_ids: pd.DataFrame) -> None:
    """
    Ghi kho và bảng thứ tự.

    Ghi vào tệp tạm rồi đổi tên (quy tắc đặt tên số 7). Kho nặng khoảng
    363 MB, ghi mất vài giây; ngắt giữa chừng để lại tệp cụt mà lần sau
    faiss.read_index sẽ báo một lỗi rất khó hiểu.
    """
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    tmp_index = INDEX_PATH.with_suffix(".index.tmp")
    faiss.write_index(index, str(tmp_index))
    os.replace(tmp_index, INDEX_PATH)

    tmp_ids = INDEX_IDS_PATH.with_suffix(".parquet.tmp")
    index_ids.to_parquet(tmp_ids, index=False)
    os.replace(tmp_ids, INDEX_IDS_PATH)


def benchmark_search(index: faiss.Index, n_queries: int = 10, top_k: int = 100) -> float:
    """
    Đo thời gian tra trung bình bằng vector ngẫu nhiên, tính bằng mili giây.

    Đây là phần "mười câu tra thử trung bình dưới 200 mili giây" của Việc 2,
    nhưng chỉ đo phần FAISS. Con số cuối cùng phải đo lại sau khi có
    ClipEncoder của Việc 3, vì mã hoá câu chữ cũng tốn thời gian.
    """
    rng = np.random.default_rng(2026)
    queries = rng.standard_normal((n_queries, CLIP_DIMENSION)).astype(np.float32)
    faiss.normalize_L2(queries)

    started_at = time.perf_counter()
    for i in range(n_queries):
        index.search(queries[i : i + 1], top_k)
    elapsed_ms = (time.perf_counter() - started_at) * 1000 / n_queries

    return elapsed_ms


# ---------------------------------------------------------------------------
# Hợp đồng với ba nhánh kia (mục 5.3)
# ---------------------------------------------------------------------------


def load_index() -> tuple[faiss.Index, pd.DataFrame]:
    """
    Đọc kho FAISS và bảng thứ tự, giữ trong bộ nhớ.

    Lần gọi thứ hai trở đi không chạm đĩa. Kho 363 MB đọc lại mỗi truy vấn
    thì không bao giờ đạt mốc 200 mili giây.
    """
    global _CACHED_INDEX, _CACHED_IDS

    if _CACHED_INDEX is not None and _CACHED_IDS is not None:
        return _CACHED_INDEX, _CACHED_IDS

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Chưa có {INDEX_PATH}. Chạy scripts/build_faiss.py trước."
        )
    if not INDEX_IDS_PATH.exists():
        raise FileNotFoundError(
            f"Có kho nhưng thiếu bảng thứ tự {INDEX_IDS_PATH}. "
            "Kho dựng bằng bản cũ — phải dựng lại bằng scripts/build_faiss.py, "
            "vì không có bảng này thì không biết vector nào là ảnh nào."
        )

    index = faiss.read_index(str(INDEX_PATH))
    index_ids = pd.read_parquet(INDEX_IDS_PATH)

    if index.ntotal != len(index_ids):
        raise ValueError(
            f"Kho có {index.ntotal:,} vector nhưng bảng thứ tự có "
            f"{len(index_ids):,} dòng. Hai tệp lệch nhau, dựng lại cả hai."
        )

    _CACHED_INDEX, _CACHED_IDS = index, index_ids
    return index, index_ids


def search_by_vector(
    vector: np.ndarray,
    top_k: int = 100,
) -> list[Hit]:
    """
    Tra kho bằng một vector CLIP.

    Vector được chuyển sang float32, kiểm tra đúng 512 chiều
    và chuẩn hoá L2 trước khi search để inner product tương đương cosine.
    """
    index, index_ids = load_index()

    query = np.ascontiguousarray(
        vector,
        dtype=np.float32,
    ).reshape(1, -1)

    if query.shape[1] != CLIP_DIMENSION:
        raise ValueError(
            f"Vector truy vấn có {query.shape[1]} chiều, "
            f"cần {CLIP_DIMENSION}."
        )

    if not np.isfinite(query).all():
        raise ValueError(
            "Vector truy vấn chứa NaN hoặc vô hạn."
        )

    norm = np.linalg.norm(query)

    if norm == 0:
        raise ValueError(
            "Vector truy vấn có độ dài bằng 0, không thể chuẩn hoá."
        )

    faiss.normalize_L2(query)

    scores, positions = index.search(
        query,
        top_k,
    )

    hits: list[Hit] = []

    for score, position in zip(
        scores[0],
        positions[0],
    ):
        if position < 0:
            continue

        row = index_ids.iloc[int(position)]

        video_id = str(row["video_id"])
        n = int(row["n"])

        frame_idx, pts_time = lookup(
            video_id,
            n,
        )

        hits.append(
            Hit(
                video_id=video_id,
                n=n,
                score=float(score),
                frame_idx=frame_idx,
                pts_time=pts_time,
                source="clip",
            )
        )

    return hits


def search_by_text(query: str, top_k: int = 100) -> list[Hit]:
    """
    Tìm kiếm bằng câu chữ.

    Cần ClipEncoder của Việc 3. Nối vào bằng cách bỏ chú thích hai dòng dưới
    khi encode_text() đã qua phép thử cùng hệ (cosine ≥ 0,99).
    """
    # from .encode.clip_encoder import ClipEncoder
    # return search_by_vector(ClipEncoder().encode_text(query), top_k)
    raise NotImplementedError(
        "search_by_text() cần ClipEncoder của Việc 3. "
        "Trong lúc chờ, dùng search_by_vector() để kiểm kho."
    )


def clear_cache() -> None:
    """Xoá bộ nhớ đệm. Gọi sau khi dựng lại kho trong cùng một tiến trình."""
    global _CACHED_INDEX, _CACHED_IDS
    _CACHED_INDEX = None
    _CACHED_IDS = None
