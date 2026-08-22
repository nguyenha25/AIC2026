"""
torch phải được nạp TRƯỚC pandas/faiss/onnxruntime trong mọi script chạm cả hai.

Trên Windows, sai thứ tự thì tiến trình chết với 0xC0000005: không traceback,
không thông báo, màn hình dừng giữa chừng rồi về dấu nhắc — nhìn y như chạy
xong. Đã cắn thật khi chạy scripts/chay_trake.py: nó gọi load_frame_map()
(pandas) rồi mới nạp ClipEncoder (torch).

Test này canh chừng chuyện đó quay lại. Nó đọc mã nguồn chứ không chạy, nên
chạy được trên mọi máy kể cả máy không có torch.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

# Script vừa chạm dữ liệu (pandas) vừa chạm mô hình (torch).
SCRIPT_CAN_GHIM = [
    "scripts/chay_trake.py",
    "scripts/kiem_clip.py",
    "scripts/run_caption_batch.py",
    "scripts/do_trong_so_rrf.py",
]

# Thư viện KHÔNG được nạp trước torch.
NAP_SAU = re.compile(
    r"^\s*(?:import|from)\s+(pandas|faiss|onnxruntime|aic2026)\b", re.MULTILINE
)
NAP_TORCH = re.compile(r"^\s*import torch\b", re.MULTILINE)


@pytest.mark.parametrize("ten", SCRIPT_CAN_GHIM)
def test_torch_nap_truoc_pandas_va_aic2026(ten):
    duong_dan = _ROOT / ten
    if not duong_dan.exists():
        pytest.skip(f"{ten} chưa có trên nhánh này")

    ma = duong_dan.read_text(encoding="utf-8")

    torch_o = NAP_TORCH.search(ma)
    assert torch_o, (
        f"{ten} thiếu khối ghim thứ tự nạp torch. Xem đầu scripts/chay_trake.py."
    )

    sau_o = NAP_SAU.search(ma)
    if sau_o is None:
        return

    assert torch_o.start() < sau_o.start(), (
        f"{ten}: `import torch` ở vị trí {torch_o.start()} nhưng "
        f"`{sau_o.group(1)}` nạp trước ở {sau_o.start()}. "
        "Trên Windows đây là 0xC0000005 — chết câm, không traceback."
    )


@pytest.mark.parametrize("ten", SCRIPT_CAN_GHIM)
def test_thieu_torch_khong_lam_do_script(ten):
    """Máy chưa cài torch vẫn phải chạy tới phần kiểm phụ thuộc."""
    duong_dan = _ROOT / ten
    if not duong_dan.exists():
        pytest.skip(f"{ten} chưa có trên nhánh này")

    ma = duong_dan.read_text(encoding="utf-8")
    khoi = ma[ma.index("import torch"): ma.index("import torch") + 200]
    assert "except ImportError" in khoi, (
        f"{ten}: import torch phải nằm trong try/except ImportError, "
        "để máy chưa cài torch nhận được thông báo tử tế thay vì traceback."
    )
