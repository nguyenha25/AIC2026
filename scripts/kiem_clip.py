"""
Việc 4b — ĐỐI CHIẾU VECTOR TỰ MÃ HOÁ VỚI clip-features-32 CỦA BTC.

CÁCH CHẠY:
    python -u -m scripts.kiem_clip
    python -u -m scripts.kiem_clip --so-mau 200 --video L23_V001 L27_V004

ĐẠT khi cosine trung bình > 0,999 VÀ không mẫu nào dưới 0,99.

VÌ SAO PHẢI KIỂM LẠI SAU VIỆC 3c
--------------------------------
Giai đoạn 1 đã kiểm một lần, đạt 0,9999. Nhưng phép kiểm đó đọc .npy theo
cách cũ. Việc 3c vừa đổi cách xác định "hàng i ứng với ảnh nào". Nếu thứ tự
mới sai thì vector sẽ khớp NHẦM ảnh — và triệu chứng duy nhất là cosine tụt.
Không có ngoại lệ nào được ném ra, không có dòng log nào đỏ.

Cosine tụt xuống khoảng 0,1-0,3 (thay vì gần 1) gần như chắc chắn là LỆCH
THỨ TỰ, không phải khác phiên bản mô hình.

THỨ TỰ IMPORT
-------------
Tệp này KHÔNG chạm faiss, chỉ torch — nên không dính lỗi 0xC0000005. Đừng
thêm import faiss vào đây.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# NẠP torch TRƯỚC MỌI THỨ KHÁC. ĐỪNG DỜI KHỐI NÀY XUỐNG DƯỚI.
#
# Trên Windows, nạp pandas (hoặc faiss, onnxruntime) vào tiến trình TRƯỚC torch
# thì tiến trình chết ngay với 0xC0000005: không traceback, không thông báo,
# màn hình chỉ dừng giữa chừng rồi về dấu nhắc — nhìn y như chạy xong. Mỗi thư
# viện mang một bản OpenMP riêng, bản nạp sau giẫm lên bản nạp trước.
#
# Script này chạm frame_map (pandas) trước khi chạm mô hình (torch), nên nếu
# không ghim thứ tự ở đây thì thứ tự nạp là ngẫu nhiên theo đường mã chạy.
# Cùng một cái bẫy đã ghi ở rank/search.py, hàm tim_ung_vien_clip().
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
except ImportError:
    pass          # máy chưa cài torch: để phần kiểm phụ thuộc báo lỗi tử tế

import numpy as np                                   # noqa: E402

from aic2026.frame_map import load_frame_map         # noqa: E402
from aic2026.kf_index import doc_dac_trung, n_sang_hang   # noqa: E402
from aic2026.paths import (                          # noqa: E402
    KEYFRAMES_DIR,
    RUNS_DIR,
    keyframe_image,
    list_video_ids,
)

NGUONG_TRUNG_BINH = 0.999
NGUONG_TUNG_MAU = 0.99


def _chuan_hoa(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    do_dai = np.linalg.norm(v)
    if do_dai == 0:
        raise ValueError("Vector có độ dài 0")
    return v / do_dai


def lay_mau(so_mau: int, video_ids: list[str] | None, seed: int) -> list[tuple[str, int]]:
    """Chọn (video_id, n) ngẫu nhiên trong số video CÓ ẢNH trên máy này."""
    co_anh = set(list_video_ids(KEYFRAMES_DIR))
    if not co_anh:
        raise FileNotFoundError(
            f"Không thấy ảnh keyframe nào trong {KEYFRAMES_DIR}. "
            "Việc 4b bắt buộc phải có ảnh gốc — tải phần shard của mình trước."
        )

    if video_ids:
        thieu = [v for v in video_ids if v not in co_anh]
        if thieu:
            raise FileNotFoundError(f"Chưa tải ảnh của: {', '.join(thieu)}")
        co_anh = set(video_ids)

    bang = load_frame_map()
    bang = bang[bang["video_id"].isin(co_anh)]

    rng = random.Random(seed)
    vi_tri = rng.sample(range(len(bang)), min(so_mau, len(bang)))
    return [
        (str(bang.iloc[v]["video_id"]), int(bang.iloc[v]["n"])) for v in vi_tri
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--so-mau", type=int, default=200)
    p.add_argument("--video", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=20260822)
    args = p.parse_args()

    mau = lay_mau(args.so_mau, args.video, args.seed)
    print(f"Lấy {len(mau)} mẫu từ {len({v for v, _ in mau})} video\n")

    from aic2026.index.encode.clip_encoder import ClipEncoder   # torch nạp ở đây

    encoder = ClipEncoder()
    print(f"Encoder: {encoder}\n")

    # Gom theo video để mỗi tệp .npy chỉ đọc một lần.
    theo_video: dict[str, list[int]] = {}
    for vid, n in mau:
        theo_video.setdefault(vid, []).append(n)

    chi_tiet: list[dict] = []
    thieu_anh = 0

    for vid, danh_sach_n in theo_video.items():
        btc = doc_dac_trung(vid)          # tự kiểm số hàng khớp frame_map
        for n in sorted(danh_sach_n):
            duong_dan_anh = keyframe_image(vid, n)
            if not duong_dan_anh.exists():
                thieu_anh += 1
                continue

            i = n_sang_hang(vid, n)
            v_tu_ma_hoa = _chuan_hoa(encoder.encode_image_path(duong_dan_anh))
            v_btc = _chuan_hoa(btc[i])

            chi_tiet.append(
                {
                    "video_id": vid,
                    "i": i,
                    "n": n,
                    "anh": duong_dan_anh.name,
                    "cosine": float(np.dot(v_tu_ma_hoa, v_btc)),
                }
            )
            print(f"  {vid} n={n:<5} i={i:<5} cosine={chi_tiet[-1]['cosine']:.5f}")

    if not chi_tiet:
        print("\nKhông đo được mẫu nào — thiếu ảnh.")
        return 1

    cos = np.array([m["cosine"] for m in chi_tiet])
    ket_qua = {
        "so_mau_do": len(chi_tiet),
        "so_mau_thieu_anh": thieu_anh,
        "cosine_trung_binh": float(cos.mean()),
        "cosine_thap_nhat": float(cos.min()),
        "cosine_cao_nhat": float(cos.max()),
        "nguong_trung_binh": NGUONG_TRUNG_BINH,
        "nguong_tung_mau": NGUONG_TUNG_MAU,
        "mo_hinh": str(encoder),
        "mau_thap_nhat": sorted(chi_tiet, key=lambda m: m["cosine"])[:5],
        "chi_tiet": chi_tiet,
    }
    ket_qua["dat"] = bool(
        ket_qua["cosine_trung_binh"] > NGUONG_TRUNG_BINH
        and ket_qua["cosine_thap_nhat"] > NGUONG_TUNG_MAU
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    dich = RUNS_DIR / "kiem_clip.json"
    with dich.open("w", encoding="utf-8") as f:
        json.dump(ket_qua, f, ensure_ascii=False, indent=2)

    print(
        f"\nTrung bình {ket_qua['cosine_trung_binh']:.5f} | "
        f"thấp nhất {ket_qua['cosine_thap_nhat']:.5f} "
        f"({len(chi_tiet)} mẫu)"
    )
    print(f"Đã ghi {dich}")

    if not ket_qua["dat"]:
        print(
            "\nCHƯA ĐẠT. Cosine thấp gần như luôn là LỆCH THỨ TỰ HÀNG .npy, "
            "không phải khác phiên bản mô hình.\n"
            "Chạy trước: python -m scripts.kiem_kf_index"
        )
    else:
        print("\nĐẠT — vector tự mã hoá khớp đúng ảnh.")

    return 0 if ket_qua["dat"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
