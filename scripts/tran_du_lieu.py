"""
ĐO TRẦN ĐIỂM CỦA BỘ DEV — câu nào KHÔNG THỂ ăn điểm dù truy hồi hoàn hảo.

CÂU HỎI
-------
BTC chấm đúng khi frame nộp nằm TRONG khoảng [frame_start, frame_end]. Ta chỉ
nộp được keyframe CÓ THẬT trong kho. Vậy: khoảng đáp án của câu đó có chứa
keyframe nào không?

Không chứa thì câu đó là 0 điểm CHẮC CHẮN, bất kể mô hình tốt đến đâu. Đó là
trần dữ liệu, không phải trần thuật toán.

VÌ SAO PHẢI CHẠY TRƯỚC MỌI PHÉP ĐO KHÁC
---------------------------------------
Nếu 30% câu dev có trần 0, thì điểm tối đa của bộ dev là 0,70 chứ không phải
1,00 — và mọi cải tiến đang bị đo trên một cái thước sai. Tệ hơn: chúng ta sẽ
đổ công tối ưu những câu không bao giờ ăn điểm được.

Đã đo trên bộ dev 32 câu: trung vị cửa sổ TRAKE là 10 khung (0,40 giây ở
25fps), còn keyframe cách nhau 65-90 khung. Đó là lý do TRAKE bằng 0. Script
này kiểm xem KIS và Q&A có dính cùng vấn đề không.

CÁCH CHẠY
    python -u -m scripts.tran_du_lieu
    python -u -m scripts.tran_du_lieu --dem 0        # không nới khoảng
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

import numpy as np   # noqa: E402

from aic2026.frame_map import load_frame_map      # noqa: E402
from aic2026.paths import DEV_QUERIES_PATH, RUNS_DIR   # noqa: E402


def bang_video(bang, video_id: str) -> np.ndarray:
    """Mảng frame_idx đã sắp của một video. Rỗng nếu chưa tải map-keyframes."""
    nhom = bang[bang["video_id"] == video_id]
    return np.sort(nhom["frame_idx"].to_numpy()) if len(nhom) else np.array([], dtype=int)


def keyframe_trong_khoang(cot: np.ndarray, dau: int, cuoi: int) -> list[int]:
    trai = int(np.searchsorted(cot, dau, side="left"))
    phai = int(np.searchsorted(cot, cuoi, side="right"))
    return [int(x) for x in cot[trai:phai]]


def gan_nhat(cot: np.ndarray, dau: int, cuoi: int) -> int:
    """Khoảng cách từ khoảng đáp án tới keyframe gần nhất, tính bằng khung."""
    if not len(cot):
        return -1
    giua = (dau + cuoi) // 2
    i = int(np.clip(np.searchsorted(cot, giua), 0, len(cot) - 1))
    ung_vien = cot[max(0, i - 1): i + 2]
    return int(min(max(dau - k, k - cuoi, 0) for k in ung_vien))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tep", default=str(DEV_QUERIES_PATH))
    p.add_argument("--dem", type=int, default=0,
                   help="nới khoảng đáp án ra mỗi bên bao nhiêu khung")
    args = p.parse_args()

    cau = [
        json.loads(d) for d in Path(args.tep).open("r", encoding="utf-8") if d.strip()
    ]
    bang = load_frame_map()

    print(f"Bộ dev: {len(cau)} câu | nới khoảng: ±{args.dem} khung\n")

    ket_qua = []
    thieu_video = []

    for q in cau:
        vid = str(q["video_id"])
        cot = bang_video(bang, vid)
        if not len(cot):
            thieu_video.append((q["id"], vid))
            continue

        loai = q.get("loai_truy_van")

        if loai == "chuoi_su_kien":
            moc = []
            for g in q["cac_giai_doan"]:
                dau = int(g["frame_start"]) - args.dem
                cuoi = int(g["frame_end"]) + args.dem
                co = keyframe_trong_khoang(cot, dau, cuoi)
                moc.append((len(co), cuoi - dau, gan_nhat(cot, dau, cuoi)))
            so_trung = sum(1 for c, _, _ in moc if c)
            ket_qua.append(
                {
                    "id": q["id"], "dang": "trake", "video_id": vid,
                    "so_moc": len(moc), "moc_co_keyframe": so_trung,
                    "tran": so_trung / len(moc),
                    "rong_trung_binh": sum(w for _, w, _ in moc) / len(moc),
                    "lech_gan_nhat": max(d for _, _, d in moc),
                }
            )
        else:
            dau = int(q["frame_start"]) - args.dem
            cuoi = int(q["frame_end"]) + args.dem
            co = keyframe_trong_khoang(cot, dau, cuoi)
            ket_qua.append(
                {
                    "id": q["id"],
                    "dang": "kis" if loai == "mo_ta" else "qa",
                    "video_id": vid,
                    "so_keyframe_trong_khoang": len(co),
                    "tran": 1.0 if co else 0.0,
                    "rong_trung_binh": cuoi - dau,
                    "lech_gan_nhat": gan_nhat(cot, dau, cuoi),
                }
            )

    if thieu_video:
        print(f"BỎ QUA {len(thieu_video)} câu — chưa có map-keyframes của video:")
        for i, v in thieu_video[:5]:
            print(f"    câu {i}: {v}")
        print()

    print(f"{'câu':<5}{'dạng':<7}{'video':<12}{'trần':>6}  chi tiết")
    for r in sorted(ket_qua, key=lambda x: (x["dang"], x["id"])):
        if r["dang"] == "trake":
            ct = (
                f"{r['moc_co_keyframe']}/{r['so_moc']} mốc có keyframe | "
                f"rộng TB {r['rong_trung_binh']:.0f} khung | "
                f"lệch xa nhất {r['lech_gan_nhat']} khung"
            )
        else:
            ct = (
                f"{r['so_keyframe_trong_khoang']} keyframe trong khoảng | "
                f"rộng {r['rong_trung_binh']:.0f} khung"
                + (f" | lệch {r['lech_gan_nhat']} khung"
                   if not r["so_keyframe_trong_khoang"] else "")
            )
        dau_hieu = "  " if r["tran"] > 0 else " !"
        print(f"{r['id']:<5}{r['dang']:<7}{r['video_id']:<12}{r['tran']:>6.2f}{dau_hieu}{ct}")

    print("\n" + "=" * 74)
    print("\nTRẦN ĐIỂM THEO DẠNG — điểm cao nhất bộ dev có thể đạt\n")

    for dang in ("kis", "qa", "trake"):
        nhom = [r for r in ket_qua if r["dang"] == dang]
        if not nhom:
            continue
        tran = sum(r["tran"] for r in nhom) / len(nhom)
        so_khong = sum(1 for r in nhom if r["tran"] == 0)
        print(
            f"  {dang:<6} {len(nhom):>2} câu | trần {tran:.4f} | "
            f"{so_khong} câu KHÔNG THỂ ăn điểm"
        )

    tong = sum(r["tran"] for r in ket_qua) / len(ket_qua) if ket_qua else 0
    print(f"\n  Trần toàn bộ: {tong:.4f}")
    print(
        "\n  Đọc con số này thế nào: điểm đo được phải chia cho TRẦN mới ra tỉ lệ\n"
        "  khai thác thật. Ví dụ trần KIS 0,75 mà đo được 0,3833 nghĩa là mạch\n"
        "  đang lấy được 51% những gì CÓ THỂ lấy, không phải 38%.\n"
        "\n  Câu có trần 0 nên LOẠI khỏi bộ dev, hoặc nới khoảng đáp án cho đúng —\n"
        "  giữ lại chỉ làm loãng mọi phép đo và kéo mọi cải tiến xuống."
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    dich = RUNS_DIR / "tran_du_lieu.json"
    with dich.open("w", encoding="utf-8") as f:
        json.dump({"dem": args.dem, "chi_tiet": ket_qua}, f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi {dich}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
