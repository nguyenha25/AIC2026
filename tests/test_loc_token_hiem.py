"""
C4 — nhánh OCR chỉ tra TOKEN HIẾM, bỏ token phổ biến.

Bài học vòng p1 đã ghi trong sổ: truy vấn tiếng Việt dài làm phồng điểm BM25
cho tài liệu không liên quan. Đây là chỗ nó chưa được áp dụng.

Đo trên câu dev 11 ("Tượng màu đỏ ... bảng ghi Mạc Cửu 1655-1735 ..."):
    cả câu 99 ký tự -> khung đúng (n=122) KHÔNG có trong 1000 kết quả đầu
    "Mac Cuu"       -> khung đúng ở HẠNG 6
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from aic2026.index.fts_index import (  # noqa: E402
    BANG_OCR,
    TextSearchIndex,
    chuan_token,
    loc_token_hiem,
)


def test_chuan_token_giong_fts5():
    """Bảng ocr_fts dùng unicode61 remove_diacritics 2 — hạ chữ thường VÀ bỏ dấu.

    Bảng tần suất không chuẩn hoá y hệt thì tra "online" ra 0 trong khi kho đầy
    "Online", và mọi phép lọc dựa trên tần suất đều sai.
    """
    assert chuan_token("Online") == "online"
    assert chuan_token("Mạc") == "mac"
    assert chuan_token("CỬU") == "cuu"
    assert chuan_token("Đèo") == "deo"


def test_giu_token_hiem_bo_token_pho_bien():
    tan_suat = {"cua": 5000, "mot": 5000, "tren": 4000, "mac": 4, "cuu": 4}
    ra = loc_token_hiem(["cua", "mot", "tren", "Mac", "Cuu"], tan_suat, 10000)
    assert set(ra) == {"Mac", "Cuu"}


def test_token_tan_suat_0_BI_LOAI_khong_phai_giu():
    """Bản đầu coi token không có trong bảng là "hiếm nhất" và giữ lại.

    Sai: token chưa từng xuất hiện thì KHÔNG BAO GIỜ khớp được gì, nên giữ
    đúng chúng là bảo đảm 0 kết quả. Đã đo: câu dev 11 giữ lại
    ['tuong','trung','quan','dung','bang'] — toàn từ không có trong kho — và
    trả về 0 kết quả, tệ hơn hẳn cả câu.
    """
    tan_suat = {"mac": 4, "cua": 5000}
    ra = loc_token_hiem(["tuong", "quan", "Mac", "cua"], tan_suat, 10000)

    assert "Mac" in ra
    assert "tuong" not in ra and "quan" not in ra


def test_moi_token_deu_pho_bien_thi_lay_it_pho_bien_nhat():
    tan_suat = {"a": 9000, "b": 8000, "c": 7000}
    ra = loc_token_hiem(["a", "b", "c"], tan_suat, 10000, so_giu=2)
    assert ra == ["c", "b"]


def test_khong_token_nao_co_trong_kho_thi_giu_nguyen_cau():
    """Lọc kiểu gì cũng 0 kết quả — trả nguyên câu để nhánh AND/OR tự xoay."""
    goc = ["xyz", "abc"]
    assert loc_token_hiem(goc, {"mac": 4}, 10000) == goc


def test_bang_rong_khong_lam_do():
    assert loc_token_hiem(["a", "b"], {}, 0) == ["a", "b"]
    assert loc_token_hiem([], {"a": 1}, 100) == []


# ---------------------------------------------------------------------------
# Đầu-cuối trên kho nhỏ
# ---------------------------------------------------------------------------

@pytest.fixture
def kho(tmp_path):
    import random

    k = TextSearchIndex(tmp_path / "text.sqlite")
    rng = random.Random(3)
    pho_bien = ["Online", "VTV", "tin", "tuc", "cua", "mot", "tren", "mau"]
    lo = [
        (f"L99_V{i // 100:03d}", i * 30, i % 100 + 1, " ".join(rng.sample(pho_bien, 5)))
        for i in range(2000)
    ]
    lo.append(("L27_V007", 3661, 122, "Online Tham TOng Mac CUu Sanh"))
    with sqlite3.connect(k.db_path) as conn:
        conn.executemany(
            f"INSERT INTO {BANG_OCR} (video_id, frame_idx, n, text) VALUES (?,?,?,?)",
            lo,
        )
        conn.commit()
    return k


def test_tan_suat_dem_theo_KHUNG_khong_theo_lan_xuat_hien(tmp_path):
    k = TextSearchIndex(tmp_path / "t.sqlite")
    with sqlite3.connect(k.db_path) as conn:
        conn.executemany(
            f"INSERT INTO {BANG_OCR} (video_id, frame_idx, n, text) VALUES (?,?,?,?)",
            [("V", 1, 1, "Online Online Online"), ("V", 2, 2, "Online")],
        )
        conn.commit()
    assert k.tan_suat_tu()["online"] == 2      # hai KHUNG, không phải bốn lần


def test_loc_hiem_tim_duoc_khung_ma_ca_cau_bo_sot(kho):
    cau = "Tuong mau do cua mot vi quan dung tren be co bang ghi Mac Cuu 1655 1735"

    tho = kho.search_text(cau, top_k=1000)
    loc = kho.search_text(cau, top_k=1000, loc_hiem=True)

    def hang(r):
        return next(
            (i for i, x in enumerate(r, 1)
             if x["video_id"] == "L27_V007" and int(x.get("n", -1)) == 122),
            None,
        )

    assert hang(loc) is not None, "lọc token hiếm phải tìm ra khung đúng"
    assert len(loc) < len(tho), "lọc phải thu hẹp tập kết quả"


def test_co_loc_hiem_la_TAT_theo_mac_dinh():
    """Câu dev 09 lên hạng 2 bằng CẢ CÂU hỏi — lọc token hiếm có thể làm hỏng
    những câu đang chạy tốt. Phải là cờ bật/tắt để đo A/B, không đổi mặc định.
    """
    import inspect

    tham_so = inspect.signature(TextSearchIndex.search_text).parameters
    assert tham_so["loc_hiem"].default is False
