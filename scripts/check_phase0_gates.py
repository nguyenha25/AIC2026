"""
TASK 9 — soát sáu điều kiện đóng Giai đoạn 0.

Chạy:  python -m scripts.check_phase0_gates

Máy tự kiểm được năm điều; điều thứ sáu (bốn máy ra cùng con số) phải
người đối chiếu với nhau trong nhóm chat — lệnh này in sẵn con số để dán.

CÒN MỘT ĐIỀU CHƯA ĐẠT THÌ CHƯA CHIA NHÁNH GIAI ĐOẠN 1.
"""

import csv
import subprocess
import sys

from src.aic2026.frame_map import FrameMap, available_video_ids
from src.aic2026.paths import (
    DATA_ROOT,
    MEDIA_INFO_DIR,
    PHASE0_RAW_DIRS,
    PROJECT_ROOT,
    REQUIRED_DIRS,
    SCHEMA_DIR,
    SUBMISSIONS_DIR,
    list_video_ids,
)

PASS, FAIL, MANUAL = "ĐẠT", "CHƯA ĐẠT", "NGƯỜI SOÁT"


def gate_1_layout():
    """Bốn máy cùng một cấu trúc thư mục."""
    missing = [d for d in REQUIRED_DIRS if not d.is_dir()]
    if PROJECT_ROOT.resolve() in DATA_ROOT.resolve().parents:
        return FAIL, "Gốc dữ liệu nằm trong thư mục mã nguồn"
    if missing:
        return FAIL, f"Thiếu {len(missing)}/{len(REQUIRED_DIRS)} thư mục chuẩn"
    return PASS, f"Đủ {len(REQUIRED_DIRS)} thư mục chuẩn"


def gate_2_data():
    """Dữ liệu BTC đã về đủ, không còn .zip."""
    counts = {
        name: len(list_video_ids(folder, suffix))
        for name, (folder, suffix) in PHASE0_RAW_DIRS.items()
    }
    core = {k: v for k, v in counts.items() if k != "media-info"}
    if not any(core.values()):
        return FAIL, "Chưa giải nén dữ liệu BTC vào raw/"
    if len(set(core.values())) > 1:
        return FAIL, f"Ba loại cốt lõi lệch số video: {core}"
    leftovers = list(DATA_ROOT.rglob("*.zip")) if DATA_ROOT.exists() else []
    if leftovers:
        return FAIL, f"Còn {len(leftovers)} tệp .zip chưa xóa"
    n = next(iter(core.values()))
    thieu = n - counts["media-info"]
    return PASS, f"{n} video mỗi loại, {thieu} video thiếu media-info, không còn .zip"


def gate_3_frame_idx():
    """Cả nhóm hiểu đúng: n KHÁC frame_idx."""
    ids = available_video_ids()
    if not ids:
        return FAIL, "Chưa có map-keyframes"
    worst, checked = 0, 0
    for vid in ids[:20]:
        try:
            worst = max(worst, FrameMap.load(vid).max_drift())
            checked += 1
        except Exception as e:
            return FAIL, f"{vid}: {e}"
    if worst > 1:
        return FAIL, f"frame_idx lệch tới {worst} so với pts_time × fps"
    return PASS, f"Kiểm {checked} video, lệch lớn nhất {worst} (cho phép ≤ 1)"


def gate_4_tests():
    """Bốn máy cùng môi trường: bộ test pass hết."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )
    except Exception as e:
        return FAIL, f"Không chạy được pytest: {e}"
    tail = [l for l in r.stdout.strip().splitlines() if l.strip()]
    summary = tail[-1] if tail else "(không có kết quả)"
    if r.returncode != 0:
        return FAIL, f"pytest KHÔNG pass — {summary}"
    return PASS, summary


def gate_5_schema():
    """Mẫu dữ liệu bốn nhánh đã có tệp ví dụ."""
    if not SCHEMA_DIR.is_dir():
        return FAIL, "Chưa có docs/schema/"
    examples = sorted(p.name for p in SCHEMA_DIR.glob("*.jsonl"))
    docs = sorted(p.name for p in SCHEMA_DIR.glob("*.md"))
    if len(examples) < 4:
        return FAIL, f"Mới có {len(examples)}/4 tệp ví dụ .jsonl"
    if not docs:
        return FAIL, "Thiếu tài liệu mô tả mẫu (.md)"
    return PASS, f"{len(examples)} tệp ví dụ + {len(docs)} tài liệu"


def gate_6_submission():
    """Tệp nộp giả 100 dòng đúng định dạng."""
    expected = {"kis": 2, "qa": 3}
    found = sorted(SUBMISSIONS_DIR.glob("query-*.csv")) if SUBMISSIONS_DIR.is_dir() else []
    if not found:
        return FAIL, "Chưa có tệp nộp giả — chạy scripts.make_dummy_submission"
    problems = []
    for path in found:
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) != 100:
            problems.append(f"{path.name}: {len(rows)} dòng (cần 100)")
            continue
        widths = {len(r) for r in rows}
        if len(widths) != 1:
            problems.append(f"{path.name}: số cột không đều {sorted(widths)}")
            continue
        task = path.stem.rsplit("-", 1)[-1]
        want = expected.get(task)
        if want and widths != {want}:
            problems.append(f"{path.name}: {widths.pop()} cột (cần {want})")
        if len({tuple(r) for r in rows}) != len(rows):
            problems.append(f"{path.name}: có dòng trùng")
    if problems:
        return FAIL, "; ".join(problems)
    return PASS, f"{len(found)} tệp, mỗi tệp 100 dòng, không trùng"


GATES = [
    ("1. Bốn máy cùng một cấu trúc thư mục", gate_1_layout),
    ("2. Dữ liệu BTC về đủ, không còn tệp nén", gate_2_data),
    ("3. Cả nhóm hiểu đúng n khác frame_idx", gate_3_frame_idx),
    ("4. Bốn máy cùng môi trường, test pass hết", gate_4_tests),
    ("5. Mẫu dữ liệu bốn nhánh đã chốt", gate_5_schema),
    ("6. Tệp nộp giả đúng định dạng BTC", gate_6_submission),
]


def main() -> int:
    print("=" * 74)
    print("SÁU ĐIỀU KIỆN ĐÓNG GIAI ĐOẠN 0")
    print("=" * 74)

    results = []
    for title, fn in GATES:
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = FAIL, f"lỗi khi kiểm: {e}"
        results.append(status)
        mark = "v" if status == PASS else "x"
        print(f"[{mark}] {title}")
        print(f"      {status} — {detail}")
        print()

    print("-" * 74)
    print("ĐIỀU KIỆN THỨ BẢY — NGƯỜI SOÁT, MÁY KHÔNG TỰ KIỂM ĐƯỢC:")
    print("  Bốn máy dán con số 'số video' của nhau vào nhóm chat và phải")
    print("  giống hệt. Chạy: python -m scripts.verify_layout")
    print("-" * 74)
    print()

    failed = sum(1 for s in results if s != PASS)
    if failed:
        print(f"=> CHƯA ĐÓNG ĐƯỢC GIAI ĐOẠN 0: còn {failed}/6 điều chưa đạt.")
        print("   Chưa chia nhánh Giai đoạn 1.")
        return 1

    print("=> SÁU ĐIỀU KIỆN ĐỀU ĐẠT. Được phép chia nhánh Giai đoạn 1.")
    print("   Bốn nhánh: index/ (tra cứu), enrich/ (OCR+ASR),")
    print("              rank/ (xếp hạng, lọc trùng), submit/+eval/ (nộp, tự chấm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
