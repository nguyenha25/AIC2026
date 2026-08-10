"""
Cấu hình chung cho pytest.

Việc import gói `aic2026` do pyproject.toml + `pip install -e .` lo.
KHÔNG vá sys.path ở đây nữa — bản cũ thêm nhầm D:/AIC2026 vào sys.path
trong khi gói nằm ở D:/AIC2026/src, nên nó chưa từng có tác dụng.

Dòng `pythonpath = .` trong pytest.ini VẪN CÒN, vì tests/test_phase0.py
đang import kiểu `from src.aic2026 import ...`. Khi nào đổi hết sang
`from aic2026 import ...` thì mới được xoá dòng đó. Đảo thứ tự là vỡ.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Bộ test kiểm BỘ KHUNG, không kiểm dữ liệu — cố ý chạy được ngay sau khi
# clone, kể cả khi máy chưa có .env (xem docstring tests/test_phase0.py).
#
# Bản cũ dùng tempfile.mkdtemp(), đẻ thêm một thư mục rác sau MỖI lần
# chạy pytest và không bao giờ dọn. Đổi sang một thư mục cố định.
#
# Điều nguy hiểm ở đây không phải bản thân thư mục giả, mà là nó IM LẶNG:
# ai đặt sai tên .env sẽ thấy test xanh và tưởng máy mình đạt. Nên có cờ
# bên dưới, và pytest_report_header in cảnh báo lên đầu mỗi lần chạy.
DANG_DUNG_DATA_ROOT_GIA = False

if not os.getenv("DATA_ROOT") and not (ROOT / ".env").exists():
    thu_muc_gia = Path(tempfile.gettempdir()) / "aic2026-test-data-root"
    thu_muc_gia.mkdir(parents=True, exist_ok=True)
    os.environ["DATA_ROOT"] = str(thu_muc_gia)
    DANG_DUNG_DATA_ROOT_GIA = True


def _doc_data_root() -> str:
    """
    Lấy DATA_ROOT để hiển thị, KHÔNG import aic2026.

    Lý do không import: chừng nào test_phase0.py còn dùng `src.aic2026`
    mà script dùng `aic2026`, import ở đây sẽ nạp gói lần thứ hai dưới
    một tên khác — hai bộ biến toàn cục, hai bộ nhớ đệm.
    """
    if os.environ.get("DATA_ROOT"):
        return os.environ["DATA_ROOT"]

    # Chưa có biến môi trường thì .env cũng chưa được nạp (python-dotenv
    # chỉ chạy lúc import aic2026.paths). Đọc thẳng tệp để khỏi báo nhầm
    # là "chưa đặt" trong khi .env vẫn ổn.
    try:
        from dotenv import dotenv_values

        gia_tri = dotenv_values(ROOT / ".env").get("DATA_ROOT")
    except Exception:
        gia_tri = None

    return gia_tri or "(chưa đặt)"


def pytest_report_header(config):
    """In DATA_ROOT lên đầu mỗi lần chạy — thấy ngay nếu trỏ sai chỗ."""
    duong_dan = _doc_data_root()

    if DANG_DUNG_DATA_ROOT_GIA:
        return (
            f"DATA_ROOT: {duong_dan}\n"
            "           ^^ THƯ MỤC GIẢ — máy này chưa có .env. "
            "Test bộ khung vẫn chạy, nhưng KHÔNG đọc dữ liệu thật."
        )

    return f"DATA_ROOT: {duong_dan}"


def pytest_itemcollected(item):
    """
    Hiện dòng mô tả tiếng Việt của mỗi mục thay vì tên hàm không dấu.

    Lưu ý: hàm này sửa item._nodeid (API nội bộ của pytest), nên
    `pytest --lf` và chạy lại một test theo nodeid sẽ không khớp.
    Chưa vướng thì cứ để; vướng thì xoá cả hàm này.
    """
    doc = (item.function.__doc__ or "").strip().splitlines()
    if doc:
        item._nodeid = f"{item.nodeid}  ->  {doc[0]}"