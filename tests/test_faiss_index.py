"""
Kiểm tra Việc 2.

Các phép kiểm:
- đúng 177.321 vector
- 512 chiều
- bảng thứ tự khớp frame_map
- mỗi video đủ vector
- vector đã chuẩn hoá
- vector trong kho khớp .npy gốc
- tra thử dưới 200 ms
- search_by_vector trả đúng cấu trúc Hit

Chạy:
    pytest tests/test_faiss_index.py -v

Máy chưa dựng kho thì các phép kiểm tự bỏ qua.
"""

from __future__ import annotations

import numpy as np
import pytest

from aic2026.index.faiss_index import (
    CLIP_DIMENSION,
    INDEX_IDS_PATH,
    INDEX_PATH,
    benchmark_search,
    load_index,
    search_by_vector,
)
from aic2026.frame_map import EXPECTED_ROWS, EXPECTED_VIDEOS, load_frame_map
from aic2026.paths import clip_features_file


MAX_SEARCH_MS = 200
SAMPLE_VIDEOS = 20


@pytest.fixture(scope="module")
def index_and_ids():
    if not INDEX_PATH.exists():
        pytest.skip(
            f"Chưa có {INDEX_PATH}. "
            "Chạy scripts/build_faiss.py trước."
        )

    if not INDEX_IDS_PATH.exists():
        pytest.skip(
            f"Chưa có bảng thứ tự {INDEX_IDS_PATH}. "
            "Dựng lại kho."
        )

    return load_index()


def test_dung_so_vector(index_and_ids):
    index, _ = index_and_ids
    assert index.ntotal == EXPECTED_ROWS


def test_dung_so_chieu(index_and_ids):
    index, _ = index_and_ids
    assert index.d == CLIP_DIMENSION


def test_bang_thu_tu_khop_so_vector(index_and_ids):
    index, index_ids = index_and_ids

    assert len(index_ids) == index.ntotal
    assert index_ids["video_id"].nunique() == EXPECTED_VIDEOS


def test_thu_tu_vector_khop_frame_map(index_and_ids):
    """
    Phép kiểm quan trọng nhất của Việc 2.

    Dòng thứ i của bảng thứ tự phải trùng dòng thứ i
    của frame_map. Lệch một dòng thì mọi kết quả tìm kiếm
    đều trỏ sai ảnh.
    """
    _, index_ids = index_and_ids
    frame_map = load_frame_map()

    assert index_ids["video_id"].tolist() == frame_map["video_id"].tolist()
    assert index_ids["n"].tolist() == frame_map["n"].tolist()


def test_moi_video_du_so_vector(index_and_ids):
    """Số vector của từng video phải khớp số ảnh trong frame_map."""
    _, index_ids = index_and_ids
    frame_map = load_frame_map()

    trong_kho = index_ids.groupby("video_id").size()
    trong_bang = frame_map.groupby("video_id").size()

    lech = (trong_kho - trong_bang).dropna()
    lech = lech[lech != 0]

    assert lech.empty, (
        f"Lệch số vector ở các video: {lech.to_dict()}"
    )


def test_vector_da_chuan_hoa(index_and_ids):
    """Vector trong kho phải có độ dài xấp xỉ 1."""
    index, _ = index_and_ids

    vi_tri = np.linspace(
        0,
        index.ntotal - 1,
        200,
        dtype=np.int64,
    )

    vectors = index.reconstruct_batch(vi_tri)
    do_dai = np.linalg.norm(vectors, axis=1)

    assert np.allclose(do_dai, 1.0, atol=1e-4)


def test_tu_tra_vector_cosine_gan_bang_1(index_and_ids):
    """
    Lấy vector trong kho đem tra lại kho.

    Do sai số float32 và các vector có thể rất giống nhau,
    không bắt buộc chính nó phải đứng đúng vị trí #1.
    Chỉ kiểm tra chính vector có cosine gần 1.
    """
    index, _ = index_and_ids

    rng = np.random.default_rng(2026)

    vi_tri = rng.choice(
        index.ntotal,
        size=30,
        replace=False,
    )

    vectors = index.reconstruct_batch(vi_tri)

    diem, _ = index.search(vectors, 10)

    self_cosine = np.sum(vectors * vectors, axis=1)

    assert self_cosine.min() >= 0.999
    assert diem[:, 0].min() >= 0.999


def test_vector_khop_tep_npy_goc(index_and_ids):
    """
    Đọc thẳng .npy gốc của vài video, so với vector trong kho.

    Bắt được trường hợp thứ tự đúng nhưng nội dung nạp nhầm video.
    """
    index, index_ids = index_and_ids

    rng = np.random.default_rng(2026)

    video_ids = index_ids["video_id"].drop_duplicates().tolist()

    mau = rng.choice(
        len(video_ids),
        size=min(SAMPLE_VIDEOS, len(video_ids)),
        replace=False,
    )

    for i in mau:
        video_id = video_ids[int(i)]

        dong_dau = int(
            index_ids.index[
                index_ids["video_id"] == video_id
            ][0]
        )

        goc = np.load(
            clip_features_file(video_id)
        ).astype(np.float32)

        goc = goc[0] / np.linalg.norm(goc[0])

        trong_kho = index.reconstruct(dong_dau)

        assert float(np.dot(goc, trong_kho)) >= 0.999, (
            f"{video_id}: vector ở vị trí {dong_dau} "
            "không khớp tệp .npy gốc"
        )


def test_tra_du_nhanh(index_and_ids):
    index, _ = index_and_ids

    ms = benchmark_search(index)

    assert ms < MAX_SEARCH_MS, (
        f"Tra mất {ms:.1f} ms, "
        f"mốc là {MAX_SEARCH_MS} ms"
    )


def test_search_by_vector_tra_ve_hit_du_truong(index_and_ids):
    index, _ = index_and_ids

    vector = index.reconstruct(0)

    hits = search_by_vector(vector, top_k=10)

    assert len(hits) == 10

    dau = hits[0]

    assert isinstance(dau.frame_idx, int)
    assert isinstance(dau.pts_time, float)
    assert dau.source == "clip"
    assert dau.score >= 0.999