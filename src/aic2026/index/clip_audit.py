"""Phần tính toán thuần NumPy cho phép nghiệm thu CLIP của Việc 4b.

Module này cố ý không import torch, open_clip, pandas hay faiss để có thể kiểm
thử nhanh trong pytest. Việc đọc ảnh và mã hoá vẫn nằm ở scripts/kiem_clip.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


CLIP_DIMENSION = 512
NGUONG_TRUNG_BINH = 0.999
NGUONG_TUNG_MAU = 0.99


def chuan_hoa_vector(
    vector: np.ndarray,
    *,
    so_chieu: int = CLIP_DIMENSION,
) -> np.ndarray:
    """Đổi một vector CLIP thành float64 và chuẩn hoá L2.

    Dùng float64 ở đúng phép đối chiếu để sai số float16/float32 của vector BTC
    không che mất chênh lệch nhỏ gần cosine 1.
    """
    ket_qua = np.asarray(vector, dtype=np.float64).reshape(-1)

    if ket_qua.shape != (so_chieu,):
        raise ValueError(
            f"Vector có shape {ket_qua.shape}, cần đúng ({so_chieu},)."
        )
    if not np.isfinite(ket_qua).all():
        raise ValueError("Vector chứa NaN hoặc giá trị vô hạn.")

    do_dai = float(np.linalg.norm(ket_qua))
    if do_dai == 0:
        raise ValueError("Vector có độ dài 0.")

    return ket_qua / do_dai


def cosine_vector(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine giữa hai vector CLIP, kèm đầy đủ kiểm tra dữ liệu đầu vào."""
    return float(np.dot(chuan_hoa_vector(a), chuan_hoa_vector(b)))


def tong_hop_ket_qua(
    chi_tiet: Sequence[dict],
    *,
    so_mau_yeu_cau: int,
    so_mau_thieu: int = 0,
    nguong_trung_binh: float = NGUONG_TRUNG_BINH,
    nguong_tung_mau: float = NGUONG_TUNG_MAU,
) -> dict:
    """Tổng hợp báo cáo và quyết định đạt/chưa đạt cho Việc 4b.

    Không được báo đạt khi đo thiếu mẫu. Kiểm ngưỡng từng mẫu giúp bắt một hàng
    bị ghép nhầm ngay cả khi 199 hàng còn lại kéo trung bình lên trên 0,999.
    """
    if so_mau_yeu_cau <= 0:
        raise ValueError("Số mẫu yêu cầu phải lớn hơn 0.")
    if so_mau_thieu < 0:
        raise ValueError("Số mẫu thiếu không được âm.")

    gia_tri = np.asarray(
        [float(mau["cosine"]) for mau in chi_tiet],
        dtype=np.float64,
    )
    if gia_tri.size and not np.isfinite(gia_tri).all():
        raise ValueError("Kết quả cosine chứa NaN hoặc giá trị vô hạn.")

    du_mau = len(chi_tiet) == so_mau_yeu_cau and so_mau_thieu == 0
    if gia_tri.size:
        trung_binh = float(gia_tri.mean())
        thap_nhat = float(gia_tri.min())
        cao_nhat = float(gia_tri.max())
        trung_vi = float(np.median(gia_tri))
    else:
        trung_binh = thap_nhat = cao_nhat = trung_vi = None

    dat_trung_binh = trung_binh is not None and trung_binh > nguong_trung_binh
    dat_tung_mau = thap_nhat is not None and thap_nhat > nguong_tung_mau

    ly_do_chua_dat: list[str] = []
    if not du_mau:
        ly_do_chua_dat.append(
            f"chỉ đo được {len(chi_tiet)}/{so_mau_yeu_cau} mẫu"
        )
    if not dat_trung_binh:
        hien_thi = "không có" if trung_binh is None else f"{trung_binh:.6f}"
        ly_do_chua_dat.append(
            f"cosine trung bình {hien_thi} không lớn hơn {nguong_trung_binh}"
        )
    if not dat_tung_mau:
        hien_thi = "không có" if thap_nhat is None else f"{thap_nhat:.6f}"
        ly_do_chua_dat.append(
            f"cosine thấp nhất {hien_thi} không lớn hơn {nguong_tung_mau}"
        )

    return {
        "so_mau_yeu_cau": int(so_mau_yeu_cau),
        "so_mau_do": len(chi_tiet),
        "so_mau_thieu": int(so_mau_thieu),
        "du_mau": bool(du_mau),
        "cosine_trung_binh": trung_binh,
        "cosine_trung_vi": trung_vi,
        "cosine_thap_nhat": thap_nhat,
        "cosine_cao_nhat": cao_nhat,
        "nguong_trung_binh": float(nguong_trung_binh),
        "nguong_tung_mau": float(nguong_tung_mau),
        "dat_trung_binh": bool(dat_trung_binh),
        "dat_tung_mau": bool(dat_tung_mau),
        "dat": bool(du_mau and dat_trung_binh and dat_tung_mau),
        "ly_do_chua_dat": ly_do_chua_dat,
    }
