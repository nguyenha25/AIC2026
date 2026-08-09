"""
Kiểm tra Việc 1.

Các điều kiện:

1. Đúng 177.321 dòng.
2. Đúng 873 video.
3. Đúng 5 cột.
4. Không có ô trống.
5. frame_idx có dtype số nguyên.
6. frame_idx ≈ pts_time × fps.
7. Không trùng (video_id, n).
8. 20 dòng ngẫu nhiên khớp CSV gốc.
9. lookup() trả đúng kiểu dữ liệu.
10. lookup() báo lỗi với key không tồn tại.

Chạy:

    pytest tests/test_frame_map.py -v
"""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from aic2026.frame_map import (
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    EXPECTED_VIDEOS,
    FRAME_MAP_DTYPES,
    MAX_ALLOWED_DRIFT,
    load_frame_map,
    lookup,
)
from aic2026.paths import (
    FRAME_MAP_PARQUET,
    map_keyframes_file,
)


RANDOM_SEED = 2026
SAMPLE_SIZE = 20


@pytest.fixture(scope="module")
def frame_map():
    if not FRAME_MAP_PARQUET.exists():
        pytest.skip(
            f"Chưa có {FRAME_MAP_PARQUET}. "
            "Chạy scripts/build_frame_map.py trước."
        )

    return load_frame_map()


def test_dung_so_dong_va_so_video(frame_map):
    assert len(frame_map) == EXPECTED_ROWS
    assert (
        frame_map["video_id"].nunique()
        == EXPECTED_VIDEOS
    )


def test_du_nam_cot_dung_ten(frame_map):
    assert list(frame_map.columns) == EXPECTED_COLUMNS


def test_khong_o_nao_trong(frame_map):
    assert (
        int(frame_map.isna().sum().sum())
        == 0
    )


def test_dtype_dung(frame_map):
    for column, expected_dtype in FRAME_MAP_DTYPES.items():
        assert str(
            frame_map[column].dtype
        ) == expected_dtype


def test_frame_idx_la_so_nguyen(frame_map):
    assert pd.api.types.is_integer_dtype(
        frame_map["frame_idx"]
    )


def test_frame_idx_khop_pts_time_nhan_fps(
    frame_map,
):
    """
    Kiểm tra toàn bộ 177.321 dòng.

    frame_idx phải lệch tối đa 1 so với:
        round(pts_time × fps)
    """

    tinh_lai = (
        frame_map["pts_time"]
        * frame_map["fps"]
    ).round()

    lech = (
        frame_map["frame_idx"]
        - tinh_lai
    ).abs()

    max_drift = int(lech.max())

    assert max_drift <= MAX_ALLOWED_DRIFT, (
        f"Lệch tối đa {max_drift} khung, "
        f"cho phép {MAX_ALLOWED_DRIFT}."
    )


def test_khong_trung_cap_video_id_va_n(
    frame_map,
):
    assert not frame_map.duplicated(
        subset=["video_id", "n"]
    ).any()


def test_tra_nguoc_20_dong_ngau_nhien_khop_tep_goc(
    frame_map,
):
    """
    Lấy 20 dòng cố định bằng random_state,
    sau đó đối chiếu trực tiếp với CSV gốc.
    """

    mau = frame_map.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED,
    )

    for dong in mau.itertuples():
        frame_idx, pts_time = lookup(
            dong.video_id,
            int(dong.n),
        )

        goc = None

        with map_keyframes_file(
            dong.video_id
        ).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            for raw in csv.DictReader(f):
                if int(float(raw["n"])) == int(
                    dong.n
                ):
                    goc = raw
                    break

        assert goc is not None, (
            f"{dong.video_id} không có "
            f"n={dong.n} trong CSV gốc"
        )

        assert frame_idx == int(
            float(goc["frame_idx"])
        ), (
            f"{dong.video_id} n={dong.n}: "
            f"Parquet={frame_idx}, "
            f"CSV={goc['frame_idx']}"
        )

        assert pts_time == pytest.approx(
            float(goc["pts_time"])
        )


def test_lookup_tra_dung_kieu(frame_map):
    dong = frame_map.iloc[0]

    frame_idx, pts_time = lookup(
        str(dong["video_id"]),
        int(dong["n"]),
    )

    assert isinstance(frame_idx, int)
    assert isinstance(pts_time, float)


def test_lookup_bao_loi_khi_khong_co(
    frame_map,
):
    with pytest.raises(KeyError):
        lookup(
            "L99_V999",
            1,
        )