"""
Việc 15 — Dò vì sao faster-whisper báo "requires the onnxruntime package"
trong khi `python -c "import onnxruntime"` chạy bình thường.

VÌ SAO CẦN SCRIPT NÀY: faster-whisper 1.0.2 bắt ImportError rồi ném ra
RuntimeError với câu chữ của riêng nó. Lỗi gốc vẫn còn trong `__cause__`
nhưng không ai in nó ra, nên người ngồi máy chỉ thấy một câu chung chung
không sửa được.

CÁCH LÀM: nạp DLL trên Windows là việc KHÔNG HOÀN TÁC ĐƯỢC trong một tiến
trình. Muốn biết module nào phá thì phải thử mỗi module trong MỘT TIẾN TRÌNH
RIÊNG, không thể thử tuần tự trong cùng một tiến trình rồi gỡ ra.

Cách chạy (đứng ở D:\\AIC2026):
    python -m scripts.chuan_doan_vad

Không sửa gì, không ghi gì. Chỉ đọc và in.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent

# Mọi thứ nặng mà tiến trình thật có nạp trước khi chạm tới VAD.
# pandas/pyarrow vào qua frame_map; ctranslate2/av/tokenizers vào qua
# faster_whisper. Bất kỳ cái nào cũng có thể nạp DLL đè lên DLL của onnxruntime.
UNG_VIEN = [
    "numpy",
    "pandas",
    "pyarrow",
    "ctranslate2",
    "av",
    "tokenizers",
    "huggingface_hub",
    "faster_whisper",
]


def chay(ma_lenh: str) -> tuple[bool, str]:
    """Chạy một đoạn mã trong TIẾN TRÌNH RIÊNG. Trả về (chạy được, chữ in ra)."""
    kq = subprocess.run(
        [sys.executable, "-c", ma_lenh],
        capture_output=True, text=True, cwd=str(GOC),
        encoding="utf-8", errors="replace",
    )
    ra = (kq.stdout or "") + (kq.stderr or "")
    return kq.returncode == 0, ra.strip()


def in_muc(tieu_de: str) -> None:
    print("\n" + "=" * 70)
    print(tieu_de)
    print("=" * 70)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    print(f"Trình thông dịch : {sys.executable}")
    print(f"Gốc dự án        : {GOC}")

    # --- 1. Bản thân gói -----------------------------------------------------
    in_muc("1. onnxruntime nạp một mình")
    ok, ra = chay("import onnxruntime; print(onnxruntime.__version__)")
    print(("CHẠY ĐƯỢC, bản " + ra) if ok else f"HỎNG:\n{ra}")
    if not ok:
        print("\n-> Gói hỏng ngay từ gốc. Không cần chạy tiếp các mục dưới.")
        print('   Sửa: pip install "onnxruntime==1.18.1"  (bản cùng thời với')
        print("   faster-whisper 1.0.2), hoặc cài Visual C++ Redistributable.")
        return

    # --- 2. Nạp từng ứng viên TRƯỚC rồi mới nạp onnxruntime ------------------
    in_muc("2. Nạp <module> trước, rồi mới nạp onnxruntime")
    print("Mỗi dòng một tiến trình riêng.\n")
    thu_pham: list[str] = []
    for ten in UNG_VIEN:
        ok, ra = chay(
            f"import {ten}\n"
            "import onnxruntime\n"
            "print('ok')"
        )
        if ok:
            print(f"  {ten:<18} ok")
        elif f"No module named '{ten}'" in ra:
            # Chưa cài KHÁC phá hỏng. Gộp hai thứ này lại là chẩn đoán bậy.
            print(f"  {ten:<18} (chưa cài — bỏ qua)")
        else:
            dong_cuoi = ra.splitlines()[-1] if ra else "không rõ"
            print(f"  {ten:<18} PHÁ — {dong_cuoi}")
            thu_pham.append(ten)

    # --- 3. Đúng chuỗi nạp của tiến trình thật -------------------------------
    in_muc("3. Đúng chuỗi nạp của scripts.run_asr_batch")
    ok, ra = chay(
        "from src.aic2026.paths import ASR_DIR\n"
        "from src.aic2026.enrich.asr import kiem_onnxruntime\n"
        "print('kiem_onnxruntime ->', kiem_onnxruntime())\n"
        "from faster_whisper import WhisperModel\n"
        "import onnxruntime\n"
        "print('sau khi co faster_whisper: onnxruntime van nap duoc')"
    )
    print(ra if ra else "(không in gì)")

    # --- 4. Tái hiện ĐÚNG chỗ hỏng, và moi lỗi gốc ra ------------------------
    in_muc("4. Gọi thẳng hàm VAD của faster-whisper — chỗ hỏng thật")
    print("get_vad_model() là hàm duy nhất trong faster-whisper chứa câu import")
    print("bị nuốt lỗi. Gọi thẳng nó thì không phải chờ chạy hết một video.\n")
    ok, ra = chay(
        "import traceback\n"
        "from faster_whisper.vad import get_vad_model\n"
        "try:\n"
        "    m = get_vad_model()\n"
        "    print('VAD NAP DUOC BINH THUONG:', type(m).__name__)\n"
        "except Exception as e:\n"
        "    print('VAD HONG:', type(e).__name__, '-', e)\n"
        "    goc = e.__cause__\n"
        "    if goc is None:\n"
        "        print('\\nKhong co loi goc — loi khong den tu cau import.')\n"
        "        traceback.print_exc()\n"
        "    else:\n"
        "        print('\\n--- LOI GOC BI NUOT (day moi la thu can doc) ---')\n"
        "        traceback.print_exception(type(goc), goc, goc.__traceback__)\n"
    )
    print(ra if ra else "(không in gì)")

    # --- 5. Kết luận ---------------------------------------------------------
    in_muc("KẾT LUẬN")
    if thu_pham:
        print(f"Module phá onnxruntime: {', '.join(thu_pham)}")
        print("-> Nạp onnxruntime TRƯỚC module đó. Trong src/aic2026/enrich/asr.py")
        print("   câu import onnxruntime đã nằm ở đầu tệp; nếu module phá là")
        print("   pandas hay pyarrow thì phải dời nó lên TRÊN dòng")
        print("   'from ..frame_map import FrameMap' nữa.")
    else:
        print("Không module nào trong danh sách phá onnxruntime.")
        print("-> Nguyên nhân KHÔNG phải thứ tự nạp. Đọc mục 4: nếu ở đó VAD nạp")
        print("   được bình thường thì lỗi lúc chạy thật đến từ chỗ khác, và cần")
        print("   xem lại lần chạy hỏng đã dùng đúng .venv này chưa.")
    print()
    print("Gửi TOÀN BỘ chữ in ra ở trên — mục 4 là phần quan trọng nhất.")


if __name__ == "__main__":
    main()
