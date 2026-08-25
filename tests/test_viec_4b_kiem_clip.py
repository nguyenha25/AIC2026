"""Test phần quyết định đạt/chưa đạt của Việc 4b, không nạp torch/open_clip."""

from __future__ import annotations

import numpy as np
import pytest

from aic2026.index.clip_audit import (
    chuan_hoa_vector,
    cosine_vector,
    tong_hop_ket_qua,
)


def _chi_tiet(cosine: list[float]) -> list[dict]:
    return [{"cosine": gia_tri} for gia_tri in cosine]


def test_du_200_mau_va_cosine_gan_mot_thi_dat():
    """Đúng 200/200 mẫu, trung bình > 0,999 và từng mẫu > 0,99 thì đạt."""
    ket_qua = tong_hop_ket_qua(
        _chi_tiet([0.9999] * 200),
        so_mau_yeu_cau=200,
    )

    assert ket_qua["dat"] is True
    assert ket_qua["so_mau_do"] == 200
    assert ket_qua["cosine_trung_binh"] == pytest.approx(0.9999)


def test_199_mau_khong_duoc_bao_dat():
    """Cosine hoàn hảo vẫn chưa đạt nếu phép nghiệm thu thiếu một mẫu."""
    ket_qua = tong_hop_ket_qua(
        _chi_tiet([1.0] * 199),
        so_mau_yeu_cau=200,
        so_mau_thieu=1,
    )

    assert ket_qua["dat"] is False
    assert ket_qua["du_mau"] is False


def test_mot_hang_lech_bi_bat_du_trung_binh_van_cao():
    """Một vector ghép nhầm không được 199 vector đúng che mất."""
    ket_qua = tong_hop_ket_qua(
        _chi_tiet([1.0] * 199 + [0.95]),
        so_mau_yeu_cau=200,
    )

    assert ket_qua["cosine_trung_binh"] > 0.999
    assert ket_qua["dat_trung_binh"] is True
    assert ket_qua["dat_tung_mau"] is False
    assert ket_qua["dat"] is False


def test_dao_thu_tu_vector_lam_cosine_tut():
    """Phép đo phát hiện rõ khi hàng .npy bị ghép với nhầm ảnh."""
    rng = np.random.default_rng(20260822)
    vector = rng.standard_normal((200, 512))
    vector /= np.linalg.norm(vector, axis=1, keepdims=True)

    dung = [cosine_vector(a, b) for a, b in zip(vector, vector)]
    sai = [cosine_vector(a, b) for a, b in zip(vector, np.roll(vector, 1, axis=0))]

    assert np.mean(dung) == pytest.approx(1.0)
    assert np.mean(sai) < 0.2


@pytest.mark.parametrize(
    "vector, noi_dung_loi",
    [
        (np.zeros(512), "độ dài 0"),
        (np.full(512, np.nan), "NaN"),
        (np.ones(511), "shape"),
    ],
)
def test_vector_hong_bi_tu_choi(vector, noi_dung_loi):
    """Vector 0, NaN hoặc sai 512 chiều phải làm phép kiểm dừng rõ ràng."""
    with pytest.raises(ValueError, match=noi_dung_loi):
        chuan_hoa_vector(vector)
