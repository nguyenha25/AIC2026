from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from aic2026.rerank import Reranker, _thay_diem


@dataclass(frozen=True)
class HitGia:
    video_id: str
    n: int
    score: float
    frame_idx: int
    pts_time: float
    source: str = "clip"


def _tao_bo_gia(tmp_path: Path, monkeypatch):
    """Reranker không nạp torch/model, đủ kiểm logic khi chưa có dev v2."""
    for n in (1, 2, 4):
        (tmp_path / f"V_{n}.jpg").write_bytes(b"anh-gia")

    bo = Reranker(
        duong_dan_anh=lambda video_id, n: tmp_path / f"{video_id}_{n}.jpg",
        kich_thuoc_lo=2,
        thiet_bi="cpu",
    )
    monkeypatch.setattr(
        bo,
        "ma_hoa_cau",
        lambda _cau: np.asarray([1.0, 0.0], dtype=np.float32),
    )

    so_lan_goi = []

    def ma_hoa_lo(cac_duong_dan, kich_thuoc_lo=None):
        so_lan_goi.append([Path(p).name for p in cac_duong_dan])
        bang = {
            "V_1.jpg": np.asarray([0.2, 0.8], dtype=np.float32),
            "V_2.jpg": np.asarray([0.9, 0.1], dtype=np.float32),
            "V_4.jpg": np.asarray([0.4, 0.6], dtype=np.float32),
        }
        return [bang[Path(p).name] for p in cac_duong_dan]

    monkeypatch.setattr(bo, "ma_hoa_nhieu_anh", ma_hoa_lo)
    return bo, so_lan_goi


def _hits():
    return [
        HitGia("V", 1, 0.99, 101, 1.0),
        HitGia("V", 2, 0.80, 202, 2.0),
        HitGia("V", 3, 0.70, 303, 3.0),  # cố ý thiếu ảnh
        HitGia("V", 4, 0.60, 404, 4.0),
    ]


def test_xep_lai_top_n_theo_lo_giu_duoi_va_khong_nuot_anh_thieu(tmp_path, monkeypatch):
    bo, so_lan_goi = _tao_bo_gia(tmp_path, monkeypatch)

    ket_qua, bao_cao = bo.xep_lai("nguoi cam o", _hits(), so_dau=3)

    assert [h.n for h in ket_qua] == [2, 1, 3, 4]
    assert [h.n for h in ket_qua[3:]] == [4]  # ngoài top-N giữ nguyên
    assert ket_qua[0].score == pytest.approx(0.9)
    assert ket_qua[0].source == "clip+rerank"
    assert bao_cao.so_da_xep_lai == 2
    assert bao_cao.so_thieu_anh == 1
    assert bao_cao.so_ung_vien_vao == 4
    assert so_lan_goi == [["V_1.jpg", "V_2.jpg"]]  # một lượt batch, không 2 lượt model


def test_cache_khong_ma_hoa_lai_anh_da_gap(tmp_path, monkeypatch):
    bo, so_lan_goi = _tao_bo_gia(tmp_path, monkeypatch)

    bo.xep_lai("cau 1", _hits(), so_dau=2)
    bo.xep_lai("cau 2", _hits(), so_dau=2)

    assert len(so_lan_goi) == 1


def test_kiem_tat_dinh_ep_ma_hoa_lai_va_so_ca_diem(tmp_path, monkeypatch):
    bo, so_lan_goi = _tao_bo_gia(tmp_path, monkeypatch)

    assert bo.kiem_tat_dinh("nguoi cam o", _hits(), so_dau=3)
    assert len(so_lan_goi) == 2


def test_rrf_giu_hang_cu_khi_hai_phieu_can_bang(tmp_path, monkeypatch):
    bo, _ = _tao_bo_gia(tmp_path, monkeypatch)

    ket_qua, _ = bo.xep_lai("nguoi cam o", _hits(), so_dau=2, cach_gop="rrf")

    assert [h.n for h in ket_qua[:2]] == [1, 2]


def test_thay_diem_ho_tro_mapping_va_khong_noi_nhan_hai_lan():
    hit = {"video_id": "V", "n": 1, "score": 0.1, "source": "clip"}

    mot = _thay_diem(hit, 0.8)
    hai = _thay_diem(mot, 0.9)

    assert hit["score"] == 0.1  # không sửa đầu vào
    assert hai["score"] == pytest.approx(0.9)
    assert hai["source"] == "clip+rerank"


@pytest.mark.parametrize("so_dau", [0, -1])
def test_tu_choi_so_dau_khong_hop_le(tmp_path, monkeypatch, so_dau):
    bo, _ = _tao_bo_gia(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="so_dau"):
        bo.xep_lai("cau", _hits(), so_dau=so_dau)
