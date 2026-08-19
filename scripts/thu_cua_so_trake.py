"""
THÍ NGHIỆM — cửa sổ lọc trùng ảnh hưởng thế nào tới điểm TRAKE.

    python -u -m scripts.thu_cua_so_trake

MỤC ĐÍCH
--------
Lấy số liệu để đề xuất Ngân chỉnh cửa sổ lọc trùng cho dạng TRAKE. Hướng dẫn Giai
đoạn 1 (Việc 4) đã ghi sẵn: "Con số 10 giây là mốc khởi đầu, chỉnh lại sau khi có
bộ câu hỏi tự chấm." Bộ câu hỏi giờ đã có, nên đây là lúc lấy con số.

KHÔNG SỬA TỆP CỦA AI
--------------------
Script này KHÔNG chạm vào `rank/search.py` hay `rank/config.py` — cả hai là tệp
của Ngân, và mục 5.1 của hướng dẫn ghi rõ chỉ người đứng tên mới được sửa.

Thay vào đó nó thay thuộc tính `cua_so_giay` trong namespace của module search
LÚC CHẠY. Được phép vì `search.py` nhập hàm đó theo tên:

    from .config import (..., cua_so_giay, ...)

Nên gán `search.cua_so_giay = ...` đổi cùng lúc CẢ HAI chỗ dùng nó:

    bước 3   loc_trung(..., cua_so_giay=cua_so_giay())
    cổng 5   tim_cap_qua_gan(ket_qua, cua_so_giay=cua_so_giay())

Hai chỗ này BẮT BUỘC đi đôi. Thu hẹp cửa sổ ở bước 3 mà để cổng 5 vẫn soát theo
10 giây thì mọi câu TRAKE sẽ báo TRƯỢT — không phải lỗi thật, chỉ là hai bên đang
dùng hai luật khác nhau. Đây chính là cái bẫy cần kiểm tra trước khi đề xuất.

CÁCH ĐỌC KẾT QUẢ
----------------
Khoảng lấy keyframe của bộ dữ liệu này là ~65–90 khung hình, tức 2,6–3,5 giây.
Nên MỌI cửa sổ dưới ~2,6 giây đều tương đương TẮT HẲN lọc trùng cho TRAKE: hai
keyframe kề nhau vốn đã cách xa hơn thế. Lựa chọn thực chất là nhị phân — giữ
10 giây, hay bỏ. Không có vùng ở giữa để tinh chỉnh.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

for _luong in (sys.stdout, sys.stderr):
    try:
        _luong.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:  # noqa: BLE001
        pass

# torch TRƯỚC pandas — xem chú thích trong scripts/benchmark_rrf.py
import torch  # noqa: F401,E402

import pandas as pd  # noqa: E402

from aic2026.paths import DERIVED_DIR, DEV_QUERIES_PATH, SUBMISSIONS_DIR  # noqa: E402
from aic2026.rank import search as _search  # noqa: E402
from aic2026.rank.config import so_anh_toi_da_moi_video  # noqa: E402
from aic2026.submit import TRAKE  # noqa: E402
from scripts.run_search import doc_tep_cau_hoi  # noqa: E402

# 10.0 là mốc hiện hành (để đối chiếu). 2.5 trở xuống ~ tắt lọc trùng.
CAC_CUA_SO = (10.0, 5.0, 3.0, 2.0, 1.0)


def doc_cau_trake() -> list[dict]:
    """
    Chỉ lấy câu dạng chuỗi sự kiện.

    Dùng doc_tep_cau_hoi() của Ngân để SỐ MỐC được phân giải đúng (đếm
    cac_giai_doan), thay vì rơi về hằng số 4 — câu 07 chỉ có 2 mốc, câu 08 có 5,
    nộp sai số cột là 0 điểm bảo đảm và sẽ che mất tác động của cửa sổ.
    """
    tat_ca = doc_tep_cau_hoi(DEV_QUERIES_PATH)
    return [
        m
        for m in tat_ca
        if not m.get("__normalize_error__") and m.get("task") == TRAKE
    ]


def chay_mot_cua_so(cua_so: float, danh_sach: list[dict]) -> dict:
    """Chạy 8 câu TRAKE với một giá trị cửa sổ, trả về số liệu."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for p in SUBMISSIONS_DIR.glob("*.csv"):
        p.unlink()

    goc = _search.cua_so_giay
    _search.cua_so_giay = lambda: cua_so  # đổi CẢ bước 3 lẫn cổng 5

    try:
        ket_qua = _search.run_queries(danh_sach, ghi_tep=True)
    finally:
        _search.cua_so_giay = goc  # trả lại ngay, đừng để rò sang lượt sau

    so_dong = [kq.so_dong_nop for kq in ket_qua]
    truot_cong5 = [kq.query_id for kq in ket_qua if not kq.dat_cong_5]

    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    chay = subprocess.run(
        [sys.executable, "-m", "scripts.run_scoring"],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        env=env,
        encoding="utf-8",
        errors="replace",
    )

    if chay.returncode != 0:
        print(f"    [LỖI] run_scoring thất bại ở cửa sổ {cua_so}s:", file=sys.stderr)
        print(chay.stderr, file=sys.stderr)
        return {
            "cua_so": cua_so,
            "diem": 0.0,
            "so_dong_tb": sum(so_dong) / max(len(so_dong), 1),
            "truot_cong5": truot_cong5,
            "loi": True,
        }

    tep = DERIVED_DIR / "eval" / "scoring_results.csv"
    diem = 0.0
    chi_tiet: dict[str, float] = {}

    if tep.exists():
        df = pd.read_csv(tep)
        cot = "final_score" if "final_score" in df.columns else df.columns[-1]
        if "task" in df.columns:
            df = df[df["task"].astype(str).str.lower() == TRAKE]
        diem = float(df[cot].sum())
        chi_tiet = {str(r["query_id"]): float(r[cot]) for _, r in df.iterrows()}
        df.to_csv(
            DERIVED_DIR / "eval" / f"scoring_trake_cuaso_{str(cua_so).replace('.', 'p')}.csv",
            index=False,
        )

    return {
        "cua_so": cua_so,
        "diem": diem,
        "so_dong_tb": sum(so_dong) / max(len(so_dong), 1),
        "truot_cong5": truot_cong5,
        "chi_tiet": chi_tiet,
        "loi": False,
    }


def main() -> None:
    print("=" * 88)
    print("THÍ NGHIỆM — CỬA SỔ LỌC TRÙNG ẢNH HƯỞNG THẾ NÀO TỚI ĐIỂM TRAKE")
    print("=" * 88)

    danh_sach = doc_cau_trake()
    if not danh_sach:
        print("Không thấy câu TRAKE nào trong bộ dev.", file=sys.stderr)
        return

    tran = so_anh_toi_da_moi_video()
    print(f"Số câu TRAKE: {len(danh_sach)}")
    for m in danh_sach:
        print(
            f"    câu {m['query_id']}: {m.get('so_moc_trake')} mốc "
            f"({m.get('nguon_so_moc', '?')})"
        )
    print(f"Trần ảnh mỗi video (chỗ 2): {tran}")

    if tran is not None and tran < 4:
        print(
            f"  [CẢNH BÁO] trần {tran} NHỎ HƠN số mốc TRAKE (4). Cửa sổ có thu hẹp\n"
            "  đến đâu cũng không gom đủ mốc — phải nới trần trước, nếu không thí\n"
            "  nghiệm này sẽ ra 0 ở mọi giá trị và dẫn tới kết luận sai.",
            file=sys.stderr,
        )

    print()

    ket: list[dict] = []
    for cua_so in CAC_CUA_SO:
        print(f"[cửa sổ {cua_so:>5.1f}s] đang chạy...", end=" ")
        moc = time.time()
        r = chay_mot_cua_so(cua_so, danh_sach)
        ket.append(r)
        print(
            f"điểm TRAKE {r['diem']:.3f} | {r['so_dong_tb']:.1f} dòng/câu "
            f"| trượt cổng 5: {len(r['truot_cong5'])} ({time.time() - moc:.1f}s)"
        )

    print("\n" + "=" * 88)
    print(f"{'CỬA SỔ':>8} | {'ĐIỂM TRAKE':>11} | {'DÒNG/CÂU':>9} | {'TRƯỢT CỔNG 5':>13}")
    print("-" * 88)
    for r in ket:
        print(
            f"{r['cua_so']:>7.1f}s | {r['diem']:>11.3f} | {r['so_dong_tb']:>9.1f} | "
            f"{len(r['truot_cong5']):>13}"
        )
    print("=" * 88)

    goc = next((r for r in ket if r["cua_so"] == 10.0), None)
    tot = max(ket, key=lambda r: r["diem"])

    if goc is not None:
        print(f"\nMốc hiện hành 10.0s : {goc['diem']:.3f} điểm / {len(danh_sach)} câu")
    print(f"Tốt nhất {tot['cua_so']:.1f}s      : {tot['diem']:.3f} điểm")

    if goc is not None and tot["diem"] > goc["diem"]:
        print(
            f"\n  => Thu hẹp cửa sổ cho TRAKE làm điểm tăng {tot['diem'] - goc['diem']:+.3f}.\n"
            "     Con số này để ĐỀ XUẤT Ngân, không phải để tự sửa: cửa sổ 10 giây\n"
            "     nằm trong định nghĩa CỔNG THOÁT SỐ 5, đổi nó là đổi luôn tiêu chí\n"
            "     đóng giai đoạn — thuộc quyền Ngân."
        )
    elif goc is not None and tot["diem"] == goc["diem"]:
        print(
            "\n  => Thu hẹp cửa sổ KHÔNG đổi điểm. Nút thắt nằm chỗ khác (trần ảnh\n"
            "     mỗi video, số ứng viên, hoặc khung hình đáp án không có trong kho)."
        )

    if any(r["truot_cong5"] for r in ket):
        print(
            "\n  LƯU Ý: có câu TRƯỢT cổng 5. Trong thí nghiệm này KHÔNG phải lỗi —\n"
            "  bước lọc trùng và phép soát cổng 5 được đổi CÙNG một giá trị, nên nếu\n"
            "  vẫn trượt thì là dữ liệu thật sự có hai dòng quá gần. Còn khi triển khai\n"
            "  thật, đổi một chỗ mà quên chỗ kia sẽ khiến MỌI câu TRAKE báo trượt oan."
        )

    print(
        "\n  ĐỌC SỐ: khoảng lấy keyframe ~2,6–3,5 giây, nên cửa sổ dưới ~2,6s tương\n"
        "  đương TẮT lọc trùng. Nếu 3.0s, 2.0s, 1.0s cho cùng một điểm thì đó là dấu\n"
        "  hiệu đã chạm đáy — lựa chọn chỉ là GIỮ (10s) hay BỎ, không có vùng giữa."
    )


if __name__ == "__main__":
    main()