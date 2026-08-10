"""
tests/test_thumbnails.py -- Viec 6.

Bon thu can khoa lai: duong dan dung hop dong, kich thuoc dung quy cach,
ghi nguyen tu (dut giua chung khong de lai anh cut), va quet de quy dung
qua thu muc long nhieu tang.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.aic2026.enrich import thumbnails as th


def _make_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 30, 200)).save(path, "JPEG")


def test_thumbnail_path_dung_hop_dong():
    path = th.thumbnail_path("L21_V001", 47)
    assert path.name == "047.jpg"
    assert path.parent.name == "L21_V001"
    assert path.parent.parent == th.THUMBNAILS_DIR


def test_frame_number():
    assert th.frame_number(Path("047.jpg")) == 47
    assert th.frame_number(Path("000.jpg")) == 0
    assert th.frame_number(Path("abc.jpg")) is None


def test_frame_number_tu_choi_thieu_so_0():
    # Khong tu them so 0: neu ten khong dung du FRAME_DIGITS chu so
    # thi coi la lech quy cach, khong doan y.
    assert th.frame_number(Path("47.jpg")) is None


@pytest.mark.parametrize("size,expected", [
    ((1280, 720), (224, 126)),   # ngang
    ((720, 1280), (126, 224)),   # doc
    ((500, 500), (224, 224)),    # vuong
])
def test_canh_dai_luon_224(tmp_path, size, expected):
    src = tmp_path / "src.jpg"
    dst = tmp_path / "out" / "0000.jpg"
    _make_image(src, size)

    th.make_thumbnail(src, dst)

    with Image.open(dst) as img:
        assert img.size == expected
        assert max(img.size) == th.THUMB_LONG_EDGE


def test_khong_de_lai_tep_tmp(tmp_path):
    src = tmp_path / "src.jpg"
    dst = tmp_path / "out" / "0000.jpg"
    _make_image(src, (640, 360))

    th.make_thumbnail(src, dst)

    assert dst.exists()
    assert list(dst.parent.glob("*.tmp")) == []


def test_anh_hong_khong_de_lai_rac(tmp_path):
    src = tmp_path / "hong.jpg"
    src.write_bytes(b"khong phai anh")
    dst = tmp_path / "out" / "0000.jpg"

    with pytest.raises(Exception):
        th.make_thumbnail(src, dst)

    assert not dst.exists()
    assert list(dst.parent.glob("*.tmp")) == []


def test_quet_de_quy_qua_thu_muc_long(tmp_path):
    # Gia lap dung cach zip cua BTC giai nen ra
    nested = tmp_path / "keyframes-aic25-b1" / "keyframes" / "L21_V001"
    _make_image(nested / "0000.jpg", (64, 64))
    (tmp_path / "keyframes-aic25-b1" / "keyframes" / "L21_V001_backup").mkdir()

    dirs = th.iter_video_dirs(tmp_path)

    assert [d.name for d in dirs] == ["L21_V001"]  # _backup phai bi loai


def test_iter_video_keyframes_tra_ve_tuple_video_id_va_danh_sach_anh(tmp_path):
    # Ham quet dung chung cho Task 6 va Task 8: phai tra ve
    # (video_id, danh sach anh) chu khong phai chi thu muc.
    nested = tmp_path / "raw" / "L22_V001"
    _make_image(nested / "0000.jpg", (64, 64))
    _make_image(nested / "0001.jpg", (64, 64))

    results = list(th.iter_video_keyframes(tmp_path))

    assert len(results) == 1
    video_id, frames = results[0]
    assert video_id == "L22_V001"
    assert [p.name for p in frames] == ["0000.jpg", "0001.jpg"]


def test_trung_video_id_phat_canh_bao_khong_bi_lang_le_bo_qua(tmp_path):
    # Cung mot video_id xuat hien o hai nhanh thu muc khac nhau (vd
    # chua don phang het) phai duoc canh bao, khong duoc am tham bo qua.
    (tmp_path / "batch1" / "L23_V001").mkdir(parents=True)
    (tmp_path / "batch2" / "L23_V001").mkdir(parents=True)

    with pytest.warns(UserWarning, match="Trung video_id"):
        th.iter_video_dirs(tmp_path)
