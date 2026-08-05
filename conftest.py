"""
Cho phép pytest import được `src.aic2026` dù chạy lệnh từ thư mục nào.
Đồng thời dựng sẵn DATA_ROOT giả nếu máy chưa có .env — để bộ test chạy
được ngay sau khi clone, không phụ thuộc dữ liệu đã tải hay chưa.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not os.getenv("DATA_ROOT") and not (ROOT / ".env").exists():
    os.environ["DATA_ROOT"] = tempfile.mkdtemp(prefix="aic2026-test-")
    
def pytest_itemcollected(item):
    """Hiện dòng mô tả tiếng Việt của mỗi mục thay vì tên hàm không dấu."""
    doc = (item.function.__doc__ or "").strip().splitlines()
    if doc:
        item._nodeid = f"{item.nodeid}  ->  {doc[0]}"