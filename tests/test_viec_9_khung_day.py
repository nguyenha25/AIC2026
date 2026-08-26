"""Việc 9 — lịch lấy mẫu, danh sách video và nghiệm thu khoảng cách."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import trich_khung_day as dense
from scripts import verify_task9_acceptance as verify
from aic2026.trake_align import Khung, khoang_cach_trung_binh


def test_buoc_mac_dinh_duoi_nua_giay_voi_fps_thuong_gap():
    """Bước mặc định không vượt 4 frame — độ rộng nhỏ nhất của dev TRAKE."""
    assert dense.tinh_buoc_khung(25.0) == 4
    assert dense.tinh_buoc_khung(30.0) == 4
    assert dense.tinh_buoc_khung(29.97) / 29.97 < 0.5


def test_khong_cho_cau_hinh_dung_nguong_0_5():
    """Yêu cầu là dưới 0,5 giây, không phải nhỏ hơn hoặc bằng."""
    with pytest.raises(ValueError, match="nhỏ hơn"):
        dense.tinh_buoc_khung(25.0, 0.5)


def test_lich_lay_mau_giu_hai_dau_va_keyframe_moc():
    """Mốc BTC được chen thêm nhưng danh sách vẫn tăng và không trùng."""
    assert dense.tao_chi_so_khung(3, 25, 10, [7, 23, 23, 99]) == [3, 7, 13, 23, 25]


def test_doc_danh_sach_txt_loai_trung_giu_thu_tu(tmp_path):
    """Danh sách thủ công chấp nhận chú thích, dấu phẩy và video lặp."""
    tep = tmp_path / "videos.txt"
    tep.write_text(
        "# video nghi ngờ\nL23_V025, L23_V026\nL23_V025  # lặp\n",
        encoding="utf-8",
    )
    assert dense.doc_danh_sach_video(tep) == ["L23_V025", "L23_V026"]


def test_doc_top10_jsonl_lay_video_id_va_loai_trung(tmp_path):
    """Có thể đưa thẳng top10.jsonl của run_search vào --danh-sach."""
    tep = tmp_path / "top10.jsonl"
    tep.write_text(
        "\n".join(
            [
                json.dumps({"query_id": "1", "video_id": "L21_V001"}),
                json.dumps({"query_id": "1", "video_id": "L21_V002"}),
                json.dumps({"query_id": "2", "video_id": "L21_V001"}),
            ]
        ),
        encoding="utf-8",
    )
    assert dense.doc_danh_sach_video(tep) == ["L21_V001", "L21_V002"]


def _tao_thu_muc_dense(goc: Path, video_id: str, chi_so: list[int]) -> None:
    thu_muc = goc / video_id
    thu_muc.mkdir(parents=True)
    for frame_idx in chi_so:
        (thu_muc / f"{frame_idx:06d}.jpg").write_bytes(b"x")
    (thu_muc / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video_id": video_id,
                "ranges": [{"frame_start": 0, "frame_end": 30}],
            }
        ),
        encoding="utf-8",
    )


def test_nghiem_thu_xet_lo_lon_nhat_khong_phai_lo_nho_nhat(tmp_path, monkeypatch):
    """Một cặp ảnh gần nhau không được che lỗ 0,6 giây ở phần còn lại."""
    _tao_thu_muc_dense(tmp_path, "L00_V001", [0, 12, 30])
    monkeypatch.setattr(verify, "FRAMES_DENSE_DIR", tmp_path)
    monkeypatch.setattr(verify, "_fps", lambda _video_id: 30.0)
    kq = verify.kiem_video("L00_V001", kiem_anh=False)
    assert not kq["dat"]
    assert kq["khoang_lon_nhat_giay"] == pytest.approx(0.6)


def test_nghiem_thu_dat_khi_moi_lo_deu_duoi_0_5(tmp_path, monkeypatch):
    """Lịch 0,4 giây và đoạn cuối ngắn hơn phải được nghiệm thu."""
    _tao_thu_muc_dense(tmp_path, "L00_V002", [0, 12, 24, 30])
    monkeypatch.setattr(verify, "FRAMES_DENSE_DIR", tmp_path)
    monkeypatch.setattr(verify, "_fps", lambda _video_id: 30.0)
    kq = verify.kiem_video("L00_V002", kiem_anh=False)
    assert kq["dat"]
    assert kq["khoang_lon_nhat_giay"] == pytest.approx(0.4)


def test_trake_khong_bao_du_day_khi_co_mot_lo_lon():
    """Tầng ghép TRAKE phải nhìn lỗ lớn nhất, không nhìn median."""
    khung = [
        Khung(0, 0.0, Path("0.jpg")),
        Khung(12, 0.4, Path("12.jpg")),
        Khung(60, 2.0, Path("60.jpg")),
    ]
    assert khoang_cach_trung_binh(khung) == pytest.approx(1.6)


def test_nghiem_thu_dev_dem_dung_tung_cua_so_trake(tmp_path, monkeypatch):
    """Nghiệm thu phải báo trượt đúng sự kiện không có ảnh trong khoảng."""
    root = tmp_path / "dense"
    thu_muc = root / "L00_V003"
    thu_muc.mkdir(parents=True)
    for frame_idx in [102, 202]:
        (thu_muc / f"{frame_idx:06d}.jpg").write_bytes(b"x")
    dev = tmp_path / "dev.jsonl"
    dev.write_text(
        json.dumps(
            {
                "id": "07",
                "loai_truy_van": "chuoi_su_kien",
                "video_id": "L00_V003",
                "cac_giai_doan": [
                    {"frame_start": 100, "frame_end": 105},
                    {"frame_start": 150, "frame_end": 155},
                    {"frame_start": 200, "frame_end": 205},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify, "FRAMES_DENSE_DIR", root)
    kq = verify.kiem_bao_phu_trake(dev)
    assert (kq["trung_cua_so"], kq["tong_cua_so"]) == (2, 3)
    assert not kq["dat"]
    assert kq["truot"][0]["event"] == 2
