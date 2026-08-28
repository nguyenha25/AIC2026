"""
Việc 3b — xuất gói soi cho FiftyOne.

Test tập trung vào PHÉP ĐỔI TOẠ ĐỘ HỘP. Đó là chỗ sai không có triệu chứng:
hộp vẽ lệch sang chỗ khác mà nhìn vẫn "có vẻ hợp lý", và người soi sẽ tin
vào một cái hộp trỏ nhầm vật.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.xuat_goi_soi import doc_vat_the, hop_sang_fiftyone  # noqa: E402


def test_doi_toa_do_hop():
    """BTC ghi [ymin, xmin, ymax, xmax]; FiftyOne đòi [x, y, rộng, cao].

    Cả hai đều chuẩn hoá 0-1, nên đổi sai KHÔNG ném lỗi và KHÔNG nhìn ra
    được — hộp chỉ đơn giản nằm sai chỗ.
    """
    assert hop_sang_fiftyone([0.10, 0.20, 0.60, 0.50]) == [0.20, 0.10, 0.30, 0.50]

    # ô vuông trọn khung
    assert hop_sang_fiftyone([0.0, 0.0, 1.0, 1.0]) == [0.0, 0.0, 1.0, 1.0]

    # hộp cao và hẹp ở mép phải
    assert hop_sang_fiftyone([0.0, 0.70, 1.0, 0.95]) == pytest.approx(
        [0.70, 0.0, 0.25, 1.0]
    )


def test_hop_nguoc_khong_ra_kich_thuoc_am():
    """Toạ độ đảo ngược thì trả 0, không trả số âm — FiftyOne từ chối số âm."""
    x, y, rong, cao = hop_sang_fiftyone([0.8, 0.9, 0.2, 0.3])
    assert rong >= 0 and cao >= 0


def test_doc_vat_the_loc_theo_nguong(tmp_path, monkeypatch):
    from scripts import xuat_goi_soi

    tep = tmp_path / "047.json"
    tep.write_text(
        json.dumps(
            {
                "detection_class_entities": ["Drum", "Person", "Hat"],
                "detection_scores": [0.9, 0.8, 0.3],
                "detection_boxes": [
                    [0.1, 0.2, 0.6, 0.5],
                    [0.0, 0.7, 1.0, 0.95],
                    [0.0, 0.0, 0.2, 0.2],
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(xuat_goi_soi, "objects_file", lambda v, n: tep)

    ra = doc_vat_the("L23_V025", 47)
    assert [v["nhan"] for v in ra] == ["Drum", "Person"]   # Hat 0.3 bị loại
    assert ra[0]["hop"] == [0.2, 0.1, 0.3, 0.5]


def test_thieu_tep_khong_lam_do(tmp_path, monkeypatch):
    from scripts import xuat_goi_soi

    monkeypatch.setattr(
        xuat_goi_soi, "objects_file", lambda v, n: tmp_path / "khong_co.json"
    )
    assert doc_vat_the("L23_V025", 47) == []


def test_tep_hong_khong_lam_do(tmp_path, monkeypatch):
    from scripts import xuat_goi_soi

    tep = tmp_path / "hong.json"
    tep.write_text("{ khong phai json", encoding="utf-8")
    monkeypatch.setattr(xuat_goi_soi, "objects_file", lambda v, n: tep)
    assert doc_vat_the("L23_V025", 47) == []


def test_notebook_colab_hop_le_va_doi_toa_do_giong_script():
    """Notebook chép lại mã nạp để chạy độc lập trên Colab — nhưng phép đổi
    toạ độ thì KHÔNG được chép lại, nó phải dùng `hop` đã tính sẵn trong gói.

    Chép lại phép đổi là mở đường cho hai bản lệch nhau.
    """
    import nbformat

    tep = _ROOT / "notebooks/soi_ung_vien_colab.ipynb"
    if not tep.exists():
        pytest.skip("chưa có notebook")

    nb = nbformat.read(tep, as_version=4)
    nbformat.validate(nb)

    ma = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "bounding_box=v['hop']" in ma, "notebook phải dùng hộp đã tính sẵn"
    assert "ymin" not in ma, "notebook KHÔNG được tự đổi toạ độ lần nữa"


# ---------------------------------------------------------------------------
# Bảng soi HTML — đường dự phòng, không phụ thuộc FiftyOne
# ---------------------------------------------------------------------------

def _goi_gia(tmp_path, ten="goi"):
    goc = tmp_path / ten
    (goc / "anh").mkdir(parents=True)
    (goc / "anh" / "0001_L23_V025_0001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (goc / "du_lieu.json").write_text(
        json.dumps(
            {
                "ten": "thu",
                "cau_hoi": "bộ trống đỏ & cây đàn piano",
                "khoang_dap_an": {"video_id": "L23_V025",
                                  "frame_start": 0, "frame_end": 10},
                "khung": [
                    {
                        "video_id": "L23_V025", "n": 1, "frame_idx": 5,
                        "pts_time": 0.2, "hang": 1, "diem": 0.9, "nguon": "clip",
                        "anh": "0001_L23_V025_0001.jpg",
                        "vat_the": [{"nhan": "Drum", "diem": 0.9,
                                     "hop": [0.20, 0.10, 0.30, 0.50]}],
                        "ocr": [{"chu": "Đèo Tà Pứa", "conf": 0.8}],
                        "la_dap_an": True,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return goc


def test_bang_soi_ve_hop_dung_vi_tri(tmp_path):
    """Hộp vẽ bằng CSS phần trăm — phải khớp [x, y, rộng, cao] trong gói.

    Sai ở đây thì hộp trỏ nhầm vật mà nhìn vẫn hợp lý, và người soi tin theo.
    """
    import re

    from scripts.dung_bang_soi import dung_html

    h = dung_html(_goi_gia(tmp_path))
    khop = re.search(
        r'class="hop" style="left:([\d.]+)%;top:([\d.]+)%;'
        r'width:([\d.]+)%;height:([\d.]+)%"',
        h,
    )
    assert khop, "không vẽ hộp nào"
    assert khop.groups() == ("20.00", "10.00", "30.00", "50.00")


def test_bang_soi_thoat_ky_tu_dac_biet(tmp_path):
    """Câu hỏi và chữ OCR đi thẳng vào HTML — phải thoát, không thì vỡ trang."""
    from scripts.dung_bang_soi import dung_html

    h = dung_html(_goi_gia(tmp_path))
    assert "trống đỏ &amp; cây đàn piano" in h
    assert "<script>alert" not in h


def test_bang_soi_danh_dau_khung_dap_an(tmp_path):
    from scripts.dung_bang_soi import dung_html

    h = dung_html(_goi_gia(tmp_path))
    assert 'class="dung"' in h and "ĐÁP ÁN" in h
    assert 'data-fid="5"' in h          # frame_idx — số phải nộp


def test_bang_soi_nhung_anh_tu_chua(tmp_path):
    from scripts.dung_bang_soi import dung_html

    goc = _goi_gia(tmp_path)
    assert "data:image/jpeg;base64," in dung_html(goc, nhung_anh=True)
    assert "data:image/jpeg;base64," not in dung_html(goc)


def test_notebook_co_buoc_sua_xung_dot_pillow():
    """Colab dựng sẵn một bản Pillow, FiftyOne kéo bản khác, trộn vào nhau thì
    `import fiftyone` chết ở dòng đầu với ImportError _Ink."""
    import nbformat

    tep = _ROOT / "notebooks/soi_ung_vien_colab.ipynb"
    if not tep.exists():
        pytest.skip("chưa có notebook")

    nb = nbformat.read(tep, as_version=4)
    nbformat.validate(nb)
    ma = "\n".join(c.source for c in nb.cells if c.cell_type == "code")

    assert "force-reinstall" in ma and "pillow" in ma
    assert "do_shutdown" in ma, "phải khởi động lại máy ảo sau khi cài lại"


def test_bao_ro_khi_video_dap_an_khong_co_anh(tmp_path):
    """Đã gặp thật: câu dev 25 nằm ở L21_V031, máy chỉ có L23/L27/L30.

    Gói ra 29/300 khung và notebook in "KHÔNG có khung nào trúng" — người đọc
    tưởng mạch hỏng, trong khi tran_du_lieu đã đo câu đó trần 1,00. Đây là
    chuyện THIẾU SHARD, phải nói đúng tên.
    """
    import inspect

    from scripts import nap_fiftyone, xuat_goi_soi

    ma_xuat = inspect.getsource(xuat_goi_soi.main)
    assert "video_dap_an_co_anh" in ma_xuat
    assert "so_thieu_anh" in ma_xuat

    ma_nap = inspect.getsource(nap_fiftyone.nap)
    assert "THIẾU SHARD" in ma_nap or "thiếu shard" in ma_nap.lower()


def test_notebook_phan_biet_thieu_shard_voi_mach_truot():
    import nbformat

    tep = _ROOT / "notebooks/soi_ung_vien_colab.ipynb"
    if not tep.exists():
        pytest.skip("chưa có notebook")

    nb = nbformat.read(tep, as_version=4)
    nbformat.validate(nb)
    ma = "\n".join(c.source for c in nb.cells if c.cell_type == "code")

    assert "video_dap_an_co_anh" in ma
    assert "THIẾU SHARD" in ma


def test_duong_dan_anh_tinh_theo_cho_dat_tep_html(tmp_path):
    """Bản đầu luôn ghi src="anh/..." trong khi tệp HTML nằm ở thư mục CHA của
    gói, nên trình duyệt tìm ảnh ở goi_soi/anh/ và không thấy gì.

    Triệu chứng rất dễ đọc nhầm: trang hiện đủ chữ, đủ hộp vật thể, đủ mọi thứ
    — chỉ trống ảnh. Nhìn như lỗi trình duyệt chứ không như lỗi đường dẫn.
    """
    import re

    from scripts.dung_bang_soi import dung_html

    goc = _goi_gia(tmp_path)

    # a) HTML đặt NGAY TRONG gói
    h = dung_html(goc, dat_tai=goc)
    src = re.search(r'<img loading="lazy" src="([^"]+)"', h).group(1)
    assert (goc / src).resolve().exists(), f"{src} không trỏ tới ảnh nào"

    # b) HTML đặt ở thư mục CHA
    h = dung_html(goc, dat_tai=goc.parent)
    src = re.search(r'<img loading="lazy" src="([^"]+)"', h).group(1)
    assert (goc.parent / src).resolve().exists(), f"{src} không trỏ tới ảnh nào"
    assert src.startswith(goc.name + "/"), "phải lùi vào thư mục gói"


def test_nhung_anh_khong_dung_duong_dan(tmp_path):
    import re

    from scripts.dung_bang_soi import dung_html

    h = dung_html(_goi_gia(tmp_path), nhung_anh=True, dat_tai=tmp_path / "bat_ky")
    src = re.search(r'<img loading="lazy" src="([^"]+)"', h).group(1)
    assert src.startswith("data:image/jpeg;base64,")
