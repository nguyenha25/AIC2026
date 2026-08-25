"""
Việc 4b — ĐỐI CHIẾU VECTOR TỰ MÃ HOÁ VỚI clip-features-32 CỦA BTC.

CÁCH CHẠY:
    python -u -m scripts.kiem_clip
    python -u -m scripts.kiem_clip --so-mau 200 --video L23_V001 L27_V004

ĐẠT khi đo ĐỦ 200/200 mẫu, cosine trung bình > 0,999 VÀ không mẫu nào
dưới 0,99.

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
import os
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
from aic2026.index.clip_audit import (                # noqa: E402
    cosine_vector,
    tong_hop_ket_qua,
)
from aic2026.kf_index import doc_dac_trung, n_sang_hang   # noqa: E402
from aic2026.paths import (                          # noqa: E402
    CLIP_FEATURES_DIR,
    KEYFRAMES_DIR,
    RUNS_DIR,
    clip_features_file,
    keyframe_image,
    list_video_ids,
)


def lay_mau(so_mau: int, video_ids: list[str] | None, seed: int) -> list[tuple[str, int]]:
    """Chọn đúng số lượng (video_id, n) có cả ảnh lẫn vector BTC trên máy."""
    if so_mau <= 0:
        raise ValueError("--so-mau phải lớn hơn 0.")

    video_co_anh = set(list_video_ids(KEYFRAMES_DIR))
    if not video_co_anh:
        raise FileNotFoundError(
            f"Không thấy ảnh keyframe nào trong {KEYFRAMES_DIR}. "
            "Việc 4b bắt buộc phải có ảnh gốc — tải phần shard của mình trước."
        )

    video_co_vector = set(list_video_ids(CLIP_FEATURES_DIR, ".npy"))
    if not video_co_vector:
        raise FileNotFoundError(
            f"Không thấy vector BTC nào trong {CLIP_FEATURES_DIR}. "
            "Đã giải nén clip-features-32 chưa?"
        )

    hop_le = video_co_anh & video_co_vector

    if video_ids:
        khong_co_anh = sorted(set(video_ids) - video_co_anh)
        khong_co_vector = sorted(set(video_ids) - video_co_vector)
        if khong_co_anh or khong_co_vector:
            loi = []
            if khong_co_anh:
                loi.append(f"thiếu ảnh: {', '.join(khong_co_anh)}")
            if khong_co_vector:
                loi.append(f"thiếu vector BTC: {', '.join(khong_co_vector)}")
            raise FileNotFoundError("; ".join(loi))
        hop_le = set(video_ids)

    bang = load_frame_map()
    bang = bang[bang["video_id"].isin(hop_le)]

    # Chỉ đưa ảnh THẬT SỰ tồn tại vào tập lấy mẫu. Bản cũ lấy theo thư mục rồi
    # bỏ qua ảnh thiếu ở vòng đo, nên có thể yêu cầu 200 nhưng vẫn ĐẠT với 197.
    ung_vien: list[tuple[str, int]] = []
    for dong in bang.itertuples(index=False):
        video_id = str(dong.video_id)
        n = int(dong.n)
        if keyframe_image(video_id, n).is_file():
            ung_vien.append((video_id, n))

    if len(ung_vien) < so_mau:
        raise RuntimeError(
            f"Chỉ có {len(ung_vien)} keyframe đủ cả ảnh và vector BTC, "
            f"không thể lấy đủ {so_mau} mẫu. Tải thêm shard hoặc giảm "
            "--so-mau chỉ khi đang chẩn đoán; nghiệm thu chính thức phải là 200."
        )

    rng = random.Random(seed)
    return rng.sample(ung_vien, so_mau)


def _ghi_json_an_toan(duong_dan: Path, noi_dung: dict) -> None:
    """Ghi report nguyên tử để lần chạy bị ngắt không để lại JSON cụt."""
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    tam = duong_dan.with_suffix(duong_dan.suffix + ".partial")
    with tam.open("w", encoding="utf-8") as f:
        json.dump(noi_dung, f, ensure_ascii=False, indent=2)
    os.replace(tam, duong_dan)


def _chay(args: argparse.Namespace) -> dict:
    """Chạy phép đo; tách khỏi main để mọi đường lỗi đều ghi được report."""
    mau = lay_mau(args.so_mau, args.video, args.seed)
    print(f"Lấy đủ {len(mau)} mẫu từ {len({v for v, _ in mau})} video\n")

    from aic2026.index.encode.clip_encoder import ClipEncoder   # torch đã nạp

    encoder = ClipEncoder()
    if encoder.dimension != 512:
        raise ValueError(f"Encoder có {encoder.dimension} chiều, cần 512.")
    print(f"Encoder: {encoder}\n")

    # Gom theo video để mỗi tệp .npy chỉ đọc một lần. Việc sắp video ở đây
    # không quyết định ánh xạ: mỗi vector vẫn được tra bằng (video_id, n).
    theo_video: dict[str, list[int]] = {}
    for vid, n in mau:
        theo_video.setdefault(vid, []).append(n)

    chi_tiet: list[dict] = []

    for vid in sorted(theo_video):
        btc = doc_dac_trung(vid)          # tự kiểm số hàng khớp frame_map
        if btc.shape[1] != 512:
            raise ValueError(
                f"{clip_features_file(vid).name} có shape {btc.shape}, "
                "dimension phải là 512."
            )
        if not np.isfinite(btc).all():
            raise ValueError(f"{vid}: tệp .npy chứa NaN hoặc giá trị vô hạn.")

        for n in sorted(theo_video[vid]):
            duong_dan_anh = keyframe_image(vid, n)
            if not duong_dan_anh.is_file():
                # Ảnh có ở lúc lấy mẫu nhưng biến mất trước khi đo: phải DỪNG,
                # tuyệt đối không bỏ qua rồi báo đạt với ít hơn 200 mẫu.
                raise FileNotFoundError(f"Ảnh vừa chọn đã mất: {duong_dan_anh}")

            i = n_sang_hang(vid, n)
            v_tu_ma_hoa = encoder.encode_image_path(duong_dan_anh)
            cosine = cosine_vector(v_tu_ma_hoa, btc[i])

            chi_tiet.append(
                {
                    "video_id": vid,
                    "i": i,
                    "n": n,
                    "anh": duong_dan_anh.name,
                    "cosine": cosine,
                }
            )
            print(f"  {vid} n={n:<5} i={i:<5} cosine={cosine:.6f}")

    thong_ke = tong_hop_ket_qua(
        chi_tiet,
        so_mau_yeu_cau=args.so_mau,
        so_mau_thieu=0,
    )
    return {
        "task": "4b",
        "muc_dich": "doi_chieu_clip_tu_ma_hoa_voi_vector_btc",
        "seed": args.seed,
        "video_gioi_han": args.video,
        "mo_hinh": str(encoder),
        **thong_ke,
        "mau_thap_nhat": sorted(chi_tiet, key=lambda m: m["cosine"])[:5],
        "chi_tiet": chi_tiet,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Việc 4b: đối chiếu ảnh tự mã hoá với clip-features-32 của BTC."
        )
    )
    p.add_argument("--so-mau", type=int, default=200)
    p.add_argument("--video", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument(
        "--dau-ra",
        type=Path,
        default=RUNS_DIR / "kiem_clip.json",
        help="Tệp JSON kết quả (mặc định: DATA_ROOT/runs/kiem_clip.json).",
    )
    args = p.parse_args()

    try:
        ket_qua = _chay(args)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as loi:
        ket_qua = {
            "task": "4b",
            "dat": False,
            "so_mau_yeu_cau": args.so_mau,
            "so_mau_do": 0,
            "seed": args.seed,
            "loai_loi": type(loi).__name__,
            "loi": str(loi),
        }
        _ghi_json_an_toan(args.dau_ra, ket_qua)
        print(f"\nCHƯA ĐẠT: {loi}", file=sys.stderr)
        print(f"Đã ghi {args.dau_ra}")
        return 1

    _ghi_json_an_toan(args.dau_ra, ket_qua)

    print(
        f"\nTrung bình {ket_qua['cosine_trung_binh']:.5f} | "
        f"thấp nhất {ket_qua['cosine_thap_nhat']:.5f} "
        f"({ket_qua['so_mau_do']}/{ket_qua['so_mau_yeu_cau']} mẫu)"
    )
    print(f"Đã ghi {args.dau_ra}")

    if not ket_qua["dat"]:
        print(
            "\nCHƯA ĐẠT. Cosine thấp gần như luôn là LỆCH THỨ TỰ HÀNG .npy, "
            "không phải khác phiên bản mô hình.\n"
            "Chạy trước: python -m scripts.kiem_kf_index"
        )
        for ly_do in ket_qua["ly_do_chua_dat"]:
            print(f"  - {ly_do}")
    else:
        print(
            f"\nĐẠT — đủ {ket_qua['so_mau_yeu_cau']} mẫu và vector "
            "tự mã hoá khớp đúng ảnh."
        )

    return 0 if ket_qua["dat"] else 1


if __name__ == "__main__":
    raise SystemExit(main())