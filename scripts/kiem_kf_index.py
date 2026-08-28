"""
Nghiệm thu Việc 3c — lấy ngẫu nhiên 20 keyframe, đối chiếu n -> frame_idx.

CÁCH CHẠY:
    python -u -m scripts.kiem_kf_index
    python -u -m scripts.kiem_kf_index --so-mau 200 --video L23_V001

Thoát 0 = đạt. Thoát 1 = có lệch, ĐỪNG nộp bài cho tới khi sửa xong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.kf_index import (          # noqa: E402
    kiem_gia_dinh_n_bang_i_cong_1,
    kiem_ngau_nhien,
    kiem_ten_tep_anh,
)
from aic2026.paths import RUNS_DIR      # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--so-mau", type=int, default=20)
    p.add_argument("--video", nargs="*", default=None, help="kiểm tên tệp ảnh thật")
    p.add_argument("--seed", type=int, default=20260822)
    args = p.parse_args()

    print("1. Ba đường tính frame_idx phải cho cùng một số")
    kq = kiem_ngau_nhien(args.so_mau, seed=args.seed)
    for m in kq["mau"][:10]:
        dau = "OK  " if m["khop"] else "LỆCH"
        print(
            f"   {dau} {m['video_id']}  i={m['i']:<5} n={m['n']:<5} "
            f"frame_idx={m['frame_idx_kf_index']}"
        )
    if len(kq["mau"]) > 10:
        print(f"   ... còn {len(kq['mau']) - 10} mẫu nữa")
    print(f"   -> {kq['so_mau']} mẫu, {kq['so_lech']} lệch\n")

    print("2. Giả định n = i + 1 của notebook baseline BTC")
    gd = kiem_gia_dinh_n_bang_i_cong_1()
    print(f"   {gd['ket_luan']}")
    if gd["video_lech"]:
        print(f"   Video lệch: {', '.join(gd['video_lech'])}")
    print()

    ket_qua_anh = {}
    if args.video:
        print("3. Thứ tự tên tệp ảnh thật trên đĩa")
        for vid in args.video:
            r = kiem_ten_tep_anh(vid)
            ket_qua_anh[vid] = r
            if not r["co_anh"]:
                print(f"   {vid}: {r['ly_do']}")
            else:
                dau = "OK" if r["khop_thu_tu"] else "LỆCH"
                print(
                    f"   {dau} {vid}: {r['so_anh_tren_dia']} ảnh trên đĩa / "
                    f"{r['so_anh_theo_frame_map']} theo frame_map"
                )
                for a, b in r["vi_du_lech"]:
                    print(f"        đĩa={a}  frame_map={b}")
        print()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    dich = RUNS_DIR / "kiem_kf_index.json"
    with dich.open("w", encoding="utf-8") as f:
        json.dump(
            {"ngau_nhien": kq, "gia_dinh_n": gd, "ten_tep_anh": ket_qua_anh},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Đã ghi {dich}")

    dat = kq["dat"] and all(
        r.get("khop_thu_tu", True) for r in ket_qua_anh.values() if r.get("co_anh")
    )
    print("\nKẾT LUẬN: " + ("ĐẠT" if dat else "CHƯA ĐẠT — đừng nộp bài"))
    return 0 if dat else 1


if __name__ == "__main__":
    raise SystemExit(main())
