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
SCRIPT_REQUIRES_TORCH_FIRST = [
    "scripts/chay_trake.py",
    "scripts/kiem_clip.py",
    "scripts/run_caption_batch.py",
]

# Thư viện KHÔNG được nạp trước torch.
NAP_SAU = re.compile(
    r"^\s*(?:import|from)\s+(pandas|faiss|onnxruntime|aic2026)\b", re.MULTILINE
)
NAP_TORCH = re.compile(r"^\s*import torch\b", re.MULTILINE)


@pytest.mark.parametrize("ten", SCRIPT_REQUIRES_TORCH_FIRST)
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


@pytest.mark.parametrize("ten", SCRIPT_REQUIRES_TORCH_FIRST)
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


def test_test_khong_import_thu_vien_nang_giua_me():
    """Test KHÔNG được nạp open_clip/torch/faiss giữa một mẻ pytest.

    Mẻ test nạp pandas từ đầu (frame_map). Một test nạp open_clip sau đó là
    0xC0000005 — sập cả mẻ, không traceback Python nào cho biết vì sao.

    Đã cắn thật: pytest.importorskip("aic2026.index.encode.clip_encoder") làm
    sập mẻ 202 test ngay ở test cuối. Cần kiểm mã của module nặng thì ĐỌC
    NGUỒN bằng ast, đừng import.
    """
    import re
    from pathlib import Path

    thu_muc = Path(__file__).resolve().parent
    cam = re.compile(
        r"^\s*(?:import|from)\s+open_clip\b"
        r"|importorskip\(\s*[\"'](?:aic2026\.index\.encode|open_clip)",
        re.MULTILINE,
    )

    vi_pham = []
    for tep in sorted(thu_muc.glob("test_*.py")):
        if tep.name == Path(__file__).name:
            continue
        ma = "\n".join(
            d for d in tep.read_text(encoding="utf-8").split('"""')[::2]
        )
        if cam.search(ma):
            vi_pham.append(tep.name)

    assert not vi_pham, (
        f"nạp thư viện nặng giữa mẻ test: {vi_pham}. "
        "Dùng ast đọc mã nguồn thay vì import."
    )
