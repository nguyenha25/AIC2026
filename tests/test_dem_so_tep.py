"""
Tra tệp đánh số phải chịu được MỌI kiểu đệm — lỗi nền, ảnh hưởng Việc 4, 4b,
10, 11, 12.

BTC không đệm số thống nhất. Đo thật trên máy nhóm:
    raw/keyframes/Keyframes_L30/keyframes/L30_V023/001.jpg   <- BA chữ số

Bản cũ của paths.keyframe_image() gõ cứng f"{n:04d}.jpg" nên trả 0001.jpg và
mọi phép tra ảnh trượt. Triệu chứng rất dễ đọc nhầm: năm việc trên chỉ báo
"thiếu ảnh", trông y như chưa tải shard, chứ không báo lỗi gì.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from aic2026.paths import bang_tep_theo_so, _tep_theo_so  # noqa: E402


@pytest.mark.parametrize("mau", ["{n:03d}", "{n:04d}", "{n:06d}", "{n}"])
def test_moi_kieu_dem_deu_tra_duoc(tmp_path, mau):
    for n in (1, 7, 83):
        (tmp_path / (mau.format(n=n) + ".jpg")).write_bytes(b"x")

    for n in (1, 7, 83):
        assert _tep_theo_so(tmp_path, n, ".jpg").exists(), f"trượt ở đệm {mau}"


def test_khong_lan_duoi_tep_khac(tmp_path):
    (tmp_path / "001.jpg").write_bytes(b"x")
    (tmp_path / "001.json").write_bytes(b"x")

    assert _tep_theo_so(tmp_path, 1, ".jpg").suffix == ".jpg"
    assert _tep_theo_so(tmp_path, 1, ".json").suffix == ".json"


def test_thu_muc_khong_ton_tai_van_tra_duong_dan_de_bao_loi(tmp_path):
    thieu = tmp_path / "chua_tai"
    duong_dan = _tep_theo_so(thieu, 5, ".jpg")
    assert not duong_dan.exists()
    assert "0005" in duong_dan.name      # tên đoán được, để thông báo lỗi đọc được


def test_cache_khong_giu_ket_qua_cu_sau_khi_them_tep(tmp_path):
    """Giải nén thêm dữ liệu giữa chừng thì phải gọi refresh_scan_cache()."""
    from aic2026.paths import refresh_scan_cache

    (tmp_path / "001.jpg").write_bytes(b"x")
    assert 1 in bang_tep_theo_so(tmp_path, ".jpg")

    (tmp_path / "002.jpg").write_bytes(b"x")
    assert 2 not in bang_tep_theo_so(tmp_path, ".jpg")   # còn dùng cache cũ

    refresh_scan_cache()
    assert 2 in bang_tep_theo_so(tmp_path, ".jpg")


# ---------------------------------------------------------------------------
# .env dính dòng — lỗi đã cắn thật trên máy nhóm
# ---------------------------------------------------------------------------

def test_data_root_dinh_dong_bao_loi_ro_rang(tmp_path, monkeypatch):
    """.env không kết thúc bằng xuống dòng thì Add-Content nối vào dòng cũ:
        DATA_ROOT=D:/aic-dataANTHROPIC_API_KEY=sk-ant-...

    Bản cũ nhận nguyên chuỗi đó làm đường dẫn, và triệu chứng duy nhất là một
    FileNotFoundError với đường dẫn lạ ở tận đáy ngăn xếp của tệp khác.
    """
    import importlib

    monkeypatch.setenv("DATA_ROOT", "D:/aic-dataANTHROPIC_API_KEY=sk-ant-abc")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

    import aic2026.paths as paths_mod

    with pytest.raises(ValueError, match="dính hai dòng"):
        importlib.reload(paths_mod)

    # trả lại trạng thái cũ cho các test sau
    monkeypatch.undo()
    importlib.reload(paths_mod)


def test_data_root_binh_thuong_van_chay(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

    import aic2026.paths as paths_mod

    importlib.reload(paths_mod)
    assert paths_mod.DATA_ROOT == tmp_path

    monkeypatch.undo()
    importlib.reload(paths_mod)
