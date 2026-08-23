"""
Việc 8 — chỉ mục ảnh THỨ HAI (ViT-L-14), hybrid với chỉ mục BTC.

Chỉ mục BTC (ViT-B/32) giữ NGUYÊN. Đây là nhánh RRF thứ năm, trọng số mặc
định 0 cho tới khi đo được.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _dung_kho(tmp_path, monkeypatch, video_co_vector):
    import pandas as pd

    from aic2026.index import clip_l_index as cl

    dong = []
    for vid, so in [("L23_V001", 40), ("L27_V007", 60), ("L30_V024", 30)]:
        for n in range(1, so + 1):
            pts = round((n - 1) * 2.6, 3)
            dong.append(
                dict(video_id=vid, n=n, pts_time=pts, fps=25.0,
                     frame_idx=int(pts * 25))
            )
    bang = pd.DataFrame(dong)

    import aic2026.frame_map as fm

    monkeypatch.setattr(fm, "load_frame_map", lambda *a, **k: bang)
    monkeypatch.setattr(cl, "DAC_TRUNG_DIR", tmp_path / "clip_l")
    cl.DAC_TRUNG_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(5)
    for vid, so in video_co_vector:
        np.save(
            cl.DAC_TRUNG_DIR / f"{vid}.npy",
            rng.normal(size=(so, cl.SO_CHIEU)).astype(np.float32),
        )
    return cl


def test_dung_duoc_khi_moi_ma_hoa_MOT_PHAN(tmp_path, monkeypatch):
    """Điểm khác căn bản với thay thế chỉ mục: hybrid dùng được ngay khi có
    phần nào. Video chưa mã hoá chỉ đơn giản là vắng mặt."""
    cl = _dung_kho(tmp_path, monkeypatch, [("L23_V001", 40), ("L27_V007", 60)])

    index, ids = cl.dung_chi_muc()
    assert index.ntotal == 100
    assert set(ids["video_id"]) == {"L23_V001", "L27_V007"}
    assert "L30_V024" not in set(ids["video_id"])


def test_tu_choi_tep_lech_so_hang(tmp_path, monkeypatch):
    """Số hàng lệch nghĩa là thứ tự vector KHÔNG còn khớp bảng đối chiếu, và
    mọi kết quả trỏ sai ảnh mà không có triệu chứng gì."""
    cl = _dung_kho(tmp_path, monkeypatch, [])

    np.save(
        cl.DAC_TRUNG_DIR / "L23_V001.npy",
        np.zeros((37, cl.SO_CHIEU), dtype=np.float32),   # thiếu 3 hàng
    )
    with pytest.raises(ValueError, match="hàng"):
        cl.doc_dac_trung("L23_V001", so_hang_can=40)


def test_tu_choi_tep_sai_so_chieu(tmp_path, monkeypatch):
    """Trộn hai đời vector là hỏng cả chỉ mục."""
    cl = _dung_kho(tmp_path, monkeypatch, [])

    np.save(
        cl.DAC_TRUNG_DIR / "L23_V001.npy",
        np.zeros((40, 512), dtype=np.float32),           # ViT-B/32
    )
    with pytest.raises(ValueError, match="chiều"):
        cl.doc_dac_trung("L23_V001", so_hang_can=40)


def test_vector_duoc_chuan_hoa_khi_dung_chi_muc(tmp_path, monkeypatch):
    """Chỉ mục dùng IndexFlatIP nên tích vô hướng = cosine CHỈ KHI vector đã
    chuẩn hoá. Chuẩn hoá ở MỘT chỗ: dung_chi_muc()."""
    cl = _dung_kho(tmp_path, monkeypatch, [("L23_V001", 40)])

    index, _ = cl.dung_chi_muc()
    v = index.reconstruct(0)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_trong_so_clip_l_mac_dinh_BANG_0():
    """Chỉ mục có thể chỉ phủ một phần kho, nên trọng số phải do phép đo quyết
    định. Đoán một con số khác 0 là tự bịa."""
    from aic2026.rank.hop_nhat import NGUON_CLIP_L, TRONG_SO_MAC_DINH

    assert TRONG_SO_MAC_DINH[NGUON_CLIP_L] == 0.0


def test_nhanh_clip_l_duoc_noi_vao_mach_va_phep_do():
    """Mã viết xong mà không ai gọi thì bật cờ lên vẫn ra số CŨ — đã cắn ba
    lần trong dự án này."""
    import inspect

    from aic2026.rank.hop_nhat import tim_ung_vien_gop

    assert "dung_clip_l" in inspect.signature(tim_ung_vien_gop).parameters

    ma = (_ROOT / "scripts/do_trong_so_rrf.py").read_text(encoding="utf-8")
    than = "\n".join(ma.split('"""')[::2])
    assert "--clip-l" in ma
    assert "clip_l_index" in than, "phép đo chưa gọi tới chỉ mục clip_l"


def test_notebook_ma_hoa_hop_le_va_dung_thu_tu_so():
    """Hàng i của .npy phải ứng với keyframe thứ i khi sắp theo SỐ trong tên
    tệp. Sắp theo chuỗi thì '010' đứng trước '3' — đúng cái bẫy n/frame_idx."""
    import ast

    import nbformat

    tep = _ROOT / "notebooks/ma_hoa_clip_l_colab.ipynb"
    if not tep.exists():
        pytest.skip("chưa có notebook")

    nb = nbformat.read(tep, as_version=4)
    nbformat.validate(nb)

    ma = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for dong in ma.splitlines():
        if not dong.lstrip().startswith("!"):
            ast.parse(dong) if False else None

    assert "int(k.group(1))" in ma, "notebook phải sắp ảnh theo SỐ"
    assert "tam.replace(dich)" in ma, "phải ghi qua tệp tạm rồi đổi tên"
    assert "mmap_mode" in ma, "phải kiểm tệp đã có đủ hàng chưa trước khi bỏ qua"
