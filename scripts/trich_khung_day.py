"""
Việc 9 — TRÍCH KHUNG DÀY CHO CÁC VIDEO NGHI LÀ ĐÁP ÁN TRAKE.

VÌ SAO CẦN
----------
TRAKE đang 0,000 điểm. Nguyên nhân ở TẦNG DỮ LIỆU, không phải tầng xếp hạng:
keyframe BTC cách nhau 2,6-3,5 giây, còn cửa sổ đáp án chỉ khoảng 0,4 giây.
Ở vòng p1, hai trong ba sự kiện KHÔNG có keyframe nào rơi vào khoảng đúng.
Không thuật toán nào cứu được — chỉ có trích dày mới cứu.

TÊN TỆP LÀ frame_idx, KHÔNG PHẢI SỐ THỨ TỰ
------------------------------------------
    derived/frames_dense/<video_id>/<frame_idx 6 chữ số>.jpg

Đây là quy ước mà src/aic2026/trake_align.py đã chốt. Đặt tên theo frame_idx
chứ không theo số thứ tự là CỐ Ý: số thứ tự chính là thứ đã gây ra cái bẫy
`n` với `frame_idx` suốt dự án. Tên tệp mang thẳng số phải nộp thì không còn
chỗ nào để nhầm.

CÁCH ffmpeg ĐẾM KHUNG
---------------------
Bộ lọc `select='between(n,S,E)'` chọn theo SỐ KHUNG ĐÃ GIẢI MÃ, đếm từ 0.
Với video tốc độ khung cố định, con số đó CHÍNH LÀ frame_idx trong
map-keyframes (đã kiểm: floor(pts_time × fps) = frame_idx đúng 100% số dòng).
Kèm `-start_number S` thì tệp đầu ra được đánh số bắt đầu từ đúng S.

KHÔNG dùng `-ss` để tua nhanh ở chế độ mặc định: tua xong thì bộ đếm khung
của ffmpeg bắt đầu lại từ 0 và mọi tên tệp lệch đi một lượng không biết
trước. Giải mã từ đầu chậm hơn nhưng ĐÚNG. Đây là chỗ không được đổi chậm
lấy nhanh — sai tên tệp là sai âm thầm, y hệt cái bẫy n/frame_idx.

BẮT BUỘC CHẠY --kiem SAU KHI TRÍCH
----------------------------------
`--kiem` lấy các frame_idx mà BTC cũng có keyframe, so ảnh trích được với ảnh
BTC. Giống nhau nghĩa là đánh số đúng. Đây là phép kiểm ĐỘC LẬP duy nhất cho
quy ước tên tệp — bỏ qua nó thì cả Việc 12 chạy trên dữ liệu lệch mà không ai
biết.

CÁCH CHẠY
---------
    python -u -m scripts.trich_khung_day --tu-dev              # lấy khoảng từ bộ dev
    python -u -m scripts.trich_khung_day --video L23_V025 --giay 120 180
    python -u -m scripts.trich_khung_day --kiem L23_V025
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.frame_map import load_frame_map          # noqa: E402
from aic2026.paths import (                           # noqa: E402
    DEV_QUERIES_PATH,
    FRAMES_DENSE_DIR,
    keyframe_image,
    video_file,
)

# Đệm thêm quanh khoảng đáp án. Khoảng đọc bằng PotPlayer ở vòng p1 đã lệch so
# với đáp án thật (câu 21 chỉ trúng nhờ keyframe NGOÀI khoảng đã đọc), nên
# trích rộng hơn khoảng ghi trong bộ dev.
DEM_KHUNG_MAC_DINH = 90


def _fps(video_id: str) -> float:
    bang = load_frame_map()
    nhom = bang[bang["video_id"] == video_id]
    if nhom.empty:
        raise KeyError(f"{video_id} không có trong frame_map.")
    return float(nhom["fps"].iloc[0])


def _co_ffmpeg() -> str:
    duong_dan = shutil.which("ffmpeg")
    if not duong_dan:
        raise FileNotFoundError(
            "Không thấy ffmpeg trong PATH. Cài bằng: winget install Gyan.FFmpeg\n"
            "Rồi mở lại PowerShell."
        )
    return duong_dan


_CO_FPS_MODE: bool | None = None


def _co_khoa_fps_mode(ffmpeg: str) -> bool:
    """ffmpeg 8 đã BỎ HẲN `-vsync`, thay bằng `-fps_mode`.

    Bốn máy của nhóm không chắc cùng phiên bản ffmpeg, và bản cũ thì chưa có
    `-fps_mode`. Nên dò một lần bằng cách chạy thử trên nguồn rỗng, thay vì
    đọc chuỗi phiên bản — chuỗi phiên bản của bản tự dựng không theo khuôn nào.

    Khoá này KHÔNG bỏ được: mặc định ffmpeg có thể nhân bản hoặc bỏ bớt khung
    cho khớp tốc độ đầu ra, mà việc này cần đúng từng khung một.
    """
    global _CO_FPS_MODE
    if _CO_FPS_MODE is not None:
        return _CO_FPS_MODE

    thu = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "quiet",
            "-f", "lavfi", "-i", "nullsrc=s=16x16:d=0.1",
            "-fps_mode", "passthrough", "-f", "null", "-",
        ],
        capture_output=True,
    )
    _CO_FPS_MODE = thu.returncode == 0
    return _CO_FPS_MODE


# ---------------------------------------------------------------------------
# Trích
# ---------------------------------------------------------------------------

def trich(
    video_id: str,
    khung_dau: int,
    khung_cuoi: int,
    chat_luong: int = 2,
    ghi_de: bool = False,
) -> dict:
    """Trích mọi khung từ `khung_dau` đến `khung_cuoi` (bao gồm cả hai đầu)."""
    ffmpeg = _co_ffmpeg()
    nguon = video_file(video_id)
    if not nguon.exists():
        raise FileNotFoundError(
            f"Không thấy video gốc {nguon}. Máy này có tải shard đó không?"
        )

    dich = FRAMES_DENSE_DIR / video_id
    dich.mkdir(parents=True, exist_ok=True)

    da_co = {int(p.stem) for p in dich.glob("*.jpg") if p.stem.isdigit()}
    can = set(range(khung_dau, khung_cuoi + 1))
    if not ghi_de and can <= da_co:
        return {
            "video_id": video_id,
            "bo_qua": True,
            "so_khung": len(can),
            "ly_do": "đã có đủ khung trong khoảng này",
        }

    lenh = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(nguon),
        # KHÔNG có -ss: tua nhanh làm bộ đếm khung bắt đầu lại từ 0.
        "-vf", f"select='between(n\\,{khung_dau}\\,{khung_cuoi})'",
        # Giữ đúng số khung, không nhân bản cũng không bỏ bớt.
        *(["-fps_mode", "passthrough"] if _co_khoa_fps_mode(ffmpeg)
          else ["-vsync", "0"]),
        "-start_number", str(khung_dau),
        "-q:v", str(chat_luong),
        str(dich / "%06d.jpg"),
    ]

    ket_qua = subprocess.run(lenh, capture_output=True, text=True)
    if ket_qua.returncode != 0:
        raise RuntimeError(
            f"ffmpeg lỗi:\n{ket_qua.stderr[:800]}\n\n"
            f"Lệnh đã chạy:\n  {' '.join(lenh)}"
        )

    co_that = {int(p.stem) for p in dich.glob("*.jpg") if p.stem.isdigit()}
    thieu = sorted(can - co_that)

    return {
        "video_id": video_id,
        "bo_qua": False,
        "khung_dau": khung_dau,
        "khung_cuoi": khung_cuoi,
        "so_khung_can": len(can),
        "so_khung_co": len(can & co_that),
        "so_khung_thieu": len(thieu),
        "vi_du_thieu": thieu[:5],
        "thu_muc": str(dich),
    }


# ---------------------------------------------------------------------------
# Kiểm chứng — phép kiểm quan trọng nhất của việc này
# ---------------------------------------------------------------------------

def kiem_danh_so(video_id: str, so_mau: int = 5, ban_kinh: int = 3) -> dict:
    """So ảnh BTC ở frame_idx với các khung trích được QUANH frame_idx đó.

    KHÔNG dùng ngưỡng lệch tuyệt đối. Bản đầu làm vậy (lệch < 12 thì coi là
    khớp) và ĐÃ TRƯỢT phép thử đối kháng: bộ dữ liệu cố tình dịch tên tệp lên
    một khung vẫn được báo ĐẠT, vì khớp thật lệch 0,00 còn lệch một khung chỉ
    lệch 0,45 — cả hai đều dưới 12.

    Bản này hỏi câu hỏi đúng: trong các khung quanh đó, khung nào GIỐNG ảnh
    BTC nhất? Đáp án phải là đúng frame_idx, tức độ dịch bằng 0. Lệch bao
    nhiêu không quan trọng; quan trọng là khung đúng phải thắng hàng xóm.

    Phép này còn TỰ CHẨN ĐOÁN: mọi mẫu cùng ra độ dịch +1 nghĩa là toàn bộ
    tên tệp lệch 1 khung, và báo luôn con số đó để sửa.
    """
    import numpy as np
    from PIL import Image

    thu_muc = FRAMES_DENSE_DIR / video_id
    if not thu_muc.is_dir():
        return {"kiem_duoc": False, "ly_do": f"Chưa trích khung nào cho {video_id}"}

    da_trich = {int(p.stem): p for p in thu_muc.glob("*.jpg") if p.stem.isdigit()}
    if not da_trich:
        return {"kiem_duoc": False, "ly_do": "Thư mục rỗng"}

    bang = load_frame_map()
    nhom = bang[bang["video_id"] == video_id]

    # Chỉ lấy keyframe BTC nào có ĐỦ hàng xóm hai bên trong khoảng đã trích —
    # thiếu hàng xóm thì không so hơn kém được.
    chung = [
        (int(r.n), int(r.frame_idx))
        for r in nhom.itertuples()
        if all(int(r.frame_idx) + d in da_trich for d in range(-ban_kinh, ban_kinh + 1))
    ]
    if not chung:
        return {
            "kiem_duoc": False,
            "ly_do": (
                f"Không keyframe BTC nào có đủ {ban_kinh} khung hàng xóm hai bên "
                "trong khoảng vừa trích. Trích rộng thêm — không kiểm được thì "
                "ĐỪNG tin quy ước đánh số."
            ),
        }

    def _vector(duong_dan):
        with Image.open(duong_dan) as anh:
            return np.asarray(
                anh.convert("L").resize((160, 90)), dtype=np.float64
            )

    mau = chung[:: max(1, len(chung) // so_mau)][:so_mau]
    chi_tiet = []

    for n, frame_idx in mau:
        anh_btc = keyframe_image(video_id, n)
        if not anh_btc.exists():
            continue

        goc = _vector(anh_btc)
        lech = {
            d: float(np.abs(goc - _vector(da_trich[frame_idx + d])).mean())
            for d in range(-ban_kinh, ban_kinh + 1)
        }
        tot_nhat = min(lech, key=lech.get)

        chi_tiet.append(
            {
                "n": n,
                "frame_idx": frame_idx,
                "do_dich": tot_nhat,
                "lech_tai_0": lech[0],
                "lech_tot_nhat": lech[tot_nhat],
                "khop": tot_nhat == 0,
            }
        )

    if not chi_tiet:
        return {
            "kiem_duoc": False,
            "ly_do": f"Chưa tải ảnh keyframe của {video_id} nên không có gì để so",
        }

    do_dich = [c["do_dich"] for c in chi_tiet]
    dich_chung = do_dich[0] if len(set(do_dich)) == 1 else None

    return {
        "kiem_duoc": True,
        "so_mau": len(chi_tiet),
        "so_khop": sum(1 for c in chi_tiet if c["khop"]),
        "dat": all(c["khop"] for c in chi_tiet),
        "do_dich_he_thong": dich_chung if dich_chung else None,
        "chi_tiet": chi_tiet,
    }


def khoang_cach_thuc_te(video_id: str) -> float:
    """Khoảng cách giữa hai khung liền nhau, tính bằng giây."""
    thu_muc = FRAMES_DENSE_DIR / video_id
    so = sorted(int(p.stem) for p in thu_muc.glob("*.jpg") if p.stem.isdigit())
    if len(so) < 2:
        return float("inf")
    buoc = min(b - a for a, b in zip(so, so[1:]))
    return buoc / _fps(video_id)


# ---------------------------------------------------------------------------
# Lấy khoảng từ bộ dev
# ---------------------------------------------------------------------------

def khoang_tu_dev(duong_dan: Path, dem: int = DEM_KHUNG_MAC_DINH) -> list[tuple]:
    """Các câu TRAKE trong bộ dev -> [(video_id, khung_đầu, khung_cuối), ...].

    Lấy khoảng BAO TRÙM mọi giai đoạn của câu đó, cộng đệm hai đầu. Đệm là bắt
    buộc: bài học câu 21 vòng p1 — khoảng frame đọc bằng PotPlayer lệch so với
    đáp án thật, và câu đó chỉ trúng nhờ keyframe nằm NGOÀI khoảng đã đọc.
    """
    ra = []
    for dong in duong_dan.open("r", encoding="utf-8"):
        dong = dong.strip()
        if not dong:
            continue
        q = json.loads(dong)
        if q.get("loai_truy_van") != "chuoi_su_kien":
            continue

        giai_doan = q.get("cac_giai_doan") or []
        if not giai_doan:
            continue

        dau = min(int(g["frame_start"]) for g in giai_doan) - dem
        cuoi = max(int(g["frame_end"]) for g in giai_doan) + dem
        ra.append((str(q["video_id"]), max(0, dau), cuoi, str(q["id"])))
    return ra


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Việc 9 — trích khung dày")
    p.add_argument("--video", default=None)
    p.add_argument("--khung", nargs=2, type=int, metavar=("DAU", "CUOI"))
    p.add_argument("--giay", nargs=2, type=float, metavar=("DAU", "CUOI"))
    p.add_argument("--tu-dev", action="store_true")
    p.add_argument("--tep-dev", default=str(DEV_QUERIES_PATH))
    p.add_argument("--dem", type=int, default=DEM_KHUNG_MAC_DINH)
    p.add_argument("--kiem", default=None, metavar="VIDEO")
    p.add_argument("--ghi-de", action="store_true")
    args = p.parse_args()

    if args.kiem:
        kq = kiem_danh_so(args.kiem)
        if not kq["kiem_duoc"]:
            print(f"KHÔNG KIỂM ĐƯỢC: {kq['ly_do']}")
            return 1
        for c in kq["chi_tiet"]:
            dau = "OK  " if c["khop"] else "LỆCH"
            print(
                f"  {dau} n={c['n']:<5} frame_idx={c['frame_idx']:<8} "
                f"khung giống nhất lệch {c['do_dich']:+d}  "
                f"(tại 0: {c['lech_tai_0']:.2f} | tốt nhất: {c['lech_tot_nhat']:.2f})"
            )
        print(f"\n{kq['so_khop']}/{kq['so_mau']} mẫu có khung đúng thắng hàng xóm")
        print(f"Khoảng cách khung thực tế: {khoang_cach_thuc_te(args.kiem):.3f} giây")

        if kq.get("do_dich_he_thong"):
            d = kq["do_dich_he_thong"]
            print(
                f"\nMỌI mẫu cùng lệch {d:+d} khung -> toàn bộ tên tệp đang trỏ "
                f"sai đúng {abs(d)} khung. Xoá thư mục và trích lại; ffmpeg đã "
                "được tua nhanh bằng -ss ở đâu đó, hoặc video không phải tốc độ "
                "khung cố định."
            )

        print("\nKẾT LUẬN: " + ("ĐẠT — tên tệp trỏ đúng khung" if kq["dat"]
                                else "CHƯA ĐẠT — ĐỪNG dùng cho Việc 12"))
        return 0 if kq["dat"] else 1

    cong_viec: list[tuple] = []

    if args.tu_dev:
        tat_ca = khoang_tu_dev(Path(args.tep_dev), args.dem)
        for vid, dau, cuoi, qid in tat_ca:
            co_video = video_file(vid).exists()
            print(
                f"  câu {qid}: {vid} khung {dau}-{cuoi} "
                + ("" if co_video else "-> KHÔNG CÓ VIDEO trên máy này, bỏ qua")
            )
            if co_video:
                cong_viec.append((vid, dau, cuoi))
        print()
    elif args.video:
        if args.khung:
            cong_viec.append((args.video, args.khung[0], args.khung[1]))
        elif args.giay:
            fps = _fps(args.video)
            cong_viec.append(
                (args.video, int(args.giay[0] * fps), int(args.giay[1] * fps))
            )
        else:
            print("Cần --khung hoặc --giay đi kèm --video")
            return 1
    else:
        print("Cần --tu-dev, --video, hoặc --kiem")
        return 1

    if not cong_viec:
        print("Không có video nào trích được trên máy này.")
        return 1

    for vid, dau, cuoi in cong_viec:
        print(f"Trích {vid} khung {dau}-{cuoi} ({cuoi - dau + 1:,} khung)...")
        print("  (giải mã từ đầu video để đánh số đúng — có thể mất vài phút)")
        kq = trich(vid, dau, cuoi, ghi_de=args.ghi_de)
        if kq["bo_qua"]:
            print(f"  bỏ qua: {kq['ly_do']}")
            continue
        print(
            f"  {kq['so_khung_co']:,}/{kq['so_khung_can']:,} khung"
            + (f" — THIẾU {kq['so_khung_thieu']}: {kq['vi_du_thieu']}"
               if kq["so_khung_thieu"] else "")
        )
        print(f"  {kq['thu_muc']}")
        print(f"  khoảng cách khung: {khoang_cach_thuc_te(vid):.3f} giây\n")

    print("BẮT BUỘC bước tiếp theo — kiểm quy ước đánh số:")
    for vid, _, _ in cong_viec:
        print(f"  python -u -m scripts.trich_khung_day --kiem {vid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
