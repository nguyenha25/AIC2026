"""
Việc 3b — XUẤT GÓI SOI: ảnh + nhãn vật thể + chữ OCR, đóng thành một tệp zip
để nạp vào FiftyOne (chạy trên Colab hoặc chạy tại chỗ).

VÌ SAO CẦN BƯỚC XUẤT NÀY
------------------------
FiftyOne cần ẢNH. Kho keyframe gốc trên máy nặng vài GB, không đưa lên Colab
được. Script này lấy ra đúng phần cần soi — thường là 200-500 ứng viên đầu
của MỘT câu hỏi — thu nhỏ ảnh, gói kèm nhãn, thành tệp cỡ 20-50 MB.

CHẠY TẠI CHỖ THÌ TỐT HƠN
------------------------
Máy có ảnh gốc thì chạy FiftyOne ngay trên Windows, không cần xuất gì:
    pip install fiftyone
    python -m scripts.xuat_goi_soi --cau "..." --tai-cho
Colab chỉ đáng dùng khi ảnh nằm ở máy người khác, hoặc máy mình yếu.

TOẠ ĐỘ HỘP — CHỖ RẤT DỄ SAI
---------------------------
BTC ghi detection_boxes theo thứ tự [ymin, xmin, ymax, xmax], đã chuẩn hoá
0-1. FiftyOne đòi [x, y, rộng, cao], cũng chuẩn hoá. Đổi sai thì hộp vẽ lệch
hẳn sang chỗ khác mà nhìn vẫn "có vẻ hợp lý" — nên phép đổi nằm gọn trong
`hop_sang_fiftyone()` và có test riêng.

CÁCH CHẠY
---------
    # soi ứng viên của một câu hỏi
    python -u -m scripts.xuat_goi_soi --cau "bộ trống đỏ và cây đàn piano" --so 300

    # soi một câu trong bộ dev, kèm đánh dấu khoảng đáp án
    python -u -m scripts.xuat_goi_soi --dev 25

    # soi trọn một video
    python -u -m scripts.xuat_goi_soi --video L23_V025
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# NẠP torch TRƯỚC pandas — xem đầu scripts/chay_trake.py.
try:
    import torch  # noqa: F401
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.paths import (  # noqa: E402
    DERIVED_DIR,
    DEV_QUERIES_PATH,
    keyframe_image,
    objects_file,
    ocr_file,
)

NGUONG_DIEM = 0.4
RONG_ANH = 640          # thu nhỏ để gói nhẹ; vẫn đủ nhìn mặt người và vật thể


def hop_sang_fiftyone(hop: list[float]) -> list[float]:
    """[ymin, xmin, ymax, xmax] của BTC -> [x, y, rộng, cao] của FiftyOne.

    Cả hai đều chuẩn hoá 0-1. Đổi sai thì hộp vẽ lệch sang chỗ khác mà nhìn
    vẫn "có vẻ hợp lý" — không có triệu chứng nào để phát hiện bằng mắt.
    """
    ymin, xmin, ymax, xmax = (float(v) for v in hop[:4])
    return [xmin, ymin, max(0.0, xmax - xmin), max(0.0, ymax - ymin)]


def doc_vat_the(video_id: str, n: int) -> list[dict]:
    tep = objects_file(video_id, n)
    if not tep.exists():
        return []
    try:
        with tep.open("r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    nhan = d.get("detection_class_entities") or []
    diem = d.get("detection_scores") or []
    hop = d.get("detection_boxes") or []

    ra = []
    for i, ten in enumerate(nhan):
        try:
            s = float(diem[i])
        except (IndexError, TypeError, ValueError):
            continue
        if s <= NGUONG_DIEM or i >= len(hop):
            continue
        ra.append(
            {"nhan": str(ten), "diem": s, "hop": hop_sang_fiftyone(hop[i])}
        )
    return ra


def doc_ocr(video_id: str, n: int) -> list[dict]:
    tep = ocr_file(video_id)
    if not tep.exists():
        return []
    for dong in tep.open("r", encoding="utf-8"):
        dong = dong.strip()
        if not dong:
            continue
        try:
            d = json.loads(dong)
        except json.JSONDecodeError:
            continue
        if int(d.get("n", -1)) == int(n):
            return [
                {"chu": b.get("text", ""), "conf": float(b.get("conf", 0.0))}
                for b in (d.get("boxes") or [])
            ]
    return []


def thu_nho(nguon: Path, dich: Path, rong: int = RONG_ANH) -> bool:
    from PIL import Image

    try:
        with Image.open(nguon) as anh:
            anh = anh.convert("RGB")
            if anh.width > rong:
                anh = anh.resize(
                    (rong, round(anh.height * rong / anh.width)), Image.LANCZOS
                )
            anh.save(dich, "JPEG", quality=85)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Chọn khung cần soi
# ---------------------------------------------------------------------------

def khung_tu_cau_hoi(cau: str, so: int) -> list[dict]:
    """Ứng viên đầu bảng của một câu hỏi, đi qua đúng mạch đang chạy thật."""
    from aic2026.index.fts_index import TextSearchIndex
    from aic2026.paths import FTS_DIR
    from aic2026.rank.hop_nhat import tim_ung_vien_gop

    tim = tim_ung_vien_gop(
        kho_chu=TextSearchIndex(FTS_DIR / "text.sqlite"),
        dung_clip=True,
        dung_ocr_fts=True,
        dung_asr=True,
        mo_rong_truy_van=True,
        dang_cau="kis",
    )
    return [
        {
            "video_id": str(h.video_id),
            "n": int(h.n),
            "frame_idx": int(h.frame_idx),
            "pts_time": float(h.pts_time),
            "hang": i + 1,
            "diem": float(h.score),
            "nguon": str(h.source),
        }
        for i, h in enumerate(tim(cau, so)[:so])
    ]


def khung_tu_video(video_id: str, so: int | None) -> list[dict]:
    from aic2026.frame_map import load_frame_map

    bang = load_frame_map()
    nhom = bang[bang["video_id"] == video_id].sort_values("n")
    if nhom.empty:
        raise KeyError(f"{video_id} không có trong frame_map.")
    if so:
        nhom = nhom.head(so)
    return [
        {
            "video_id": video_id,
            "n": int(r.n),
            "frame_idx": int(r.frame_idx),
            "pts_time": float(r.pts_time),
            "hang": i + 1,
            "diem": 0.0,
            "nguon": "video",
        }
        for i, r in enumerate(nhom.itertuples())
    ]


def doc_cau_dev(ma: str) -> dict:
    for dong in DEV_QUERIES_PATH.open("r", encoding="utf-8"):
        dong = dong.strip()
        if dong and json.loads(dong).get("id") == ma:
            return json.loads(dong)
    raise KeyError(f"Không thấy câu {ma} trong {DEV_QUERIES_PATH}")


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Việc 3b — xuất gói soi cho FiftyOne")
    p.add_argument("--cau", default=None, help="câu hỏi tiếng Việt")
    p.add_argument("--dev", default=None, help="mã câu trong bộ dev")
    p.add_argument("--video", default=None, help="soi trọn một video")
    p.add_argument("--so", type=int, default=300, help="số khung tối đa")
    p.add_argument("--rong", type=int, default=RONG_ANH)
    p.add_argument("--ten", default=None, help="tên gói")
    p.add_argument("--tai-cho", action="store_true",
                   help="không nén, mở FiftyOne ngay trên máy này")
    args = p.parse_args()

    cau_hoi = args.cau
    khoang_dap_an = None

    if args.dev:
        q = doc_cau_dev(args.dev)
        cau_hoi = q["cau_hoi"]
        if q.get("loai_truy_van") != "chuoi_su_kien":
            khoang_dap_an = {
                "video_id": q["video_id"],
                "frame_start": int(q["frame_start"]),
                "frame_end": int(q["frame_end"]),
            }
        print(f"Câu dev {args.dev}: {cau_hoi[:90]}")

    if args.video:
        khung = khung_tu_video(args.video, args.so)
        ten = args.ten or f"video_{args.video}"
    elif cau_hoi:
        khung = khung_tu_cau_hoi(cau_hoi, args.so)
        ten = args.ten or ("dev_" + args.dev if args.dev else "cau_hoi")
    else:
        print("Cần --cau, --dev hoặc --video")
        return 1

    if not khung:
        print("Không có khung nào để soi.")
        return 1

    goc = DERIVED_DIR / "goi_soi" / ten
    if goc.exists():
        shutil.rmtree(goc)
    (goc / "anh").mkdir(parents=True)

    print(f"\nGói {len(khung)} khung -> {goc}")

    ban_ghi = []
    thieu_anh = 0
    video_thieu: set[str] = set()

    for k in khung:
        nguon = keyframe_image(k["video_id"], k["n"])
        ten_anh = f"{k['hang']:04d}_{k['video_id']}_{k['n']:04d}.jpg"
        if not nguon.exists() or not thu_nho(nguon, goc / "anh" / ten_anh, args.rong):
            thieu_anh += 1
            video_thieu.add(k["video_id"])
            continue

        muc = dict(k)
        muc["anh"] = ten_anh
        muc["vat_the"] = doc_vat_the(k["video_id"], k["n"])
        muc["ocr"] = doc_ocr(k["video_id"], k["n"])
        if khoang_dap_an and k["video_id"] == khoang_dap_an["video_id"]:
            muc["la_dap_an"] = (
                khoang_dap_an["frame_start"] <= k["frame_idx"]
                <= khoang_dap_an["frame_end"]
            )
        ban_ghi.append(muc)

    # Video đáp án CÓ ảnh trên máy này hay không — khác nhau hoàn toàn về ý nghĩa.
    #
    # Không có ảnh thì gói không thể chứa khung đúng, và câu "không khung nào
    # trúng" nói về SHARD THIẾU chứ không nói mạch trượt. Đã gặp thật: câu dev
    # 25 nằm ở L21_V031, máy chỉ có L23/L27/L30, gói ra 29/300 khung và người
    # đọc tưởng mạch hỏng.
    video_dap_an = khoang_dap_an["video_id"] if khoang_dap_an else None
    dap_an_co_anh = None
    if video_dap_an:
        dap_an_co_anh = keyframe_image(video_dap_an, 1).parent.is_dir()

    with (goc / "du_lieu.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "ten": ten,
                "cau_hoi": cau_hoi,
                "khoang_dap_an": khoang_dap_an,
                "so_khung": len(ban_ghi),
                "so_khung_yeu_cau": len(khung),
                "so_thieu_anh": thieu_anh,
                "video_thieu_anh": sorted(video_thieu)[:30],
                "video_dap_an_co_anh": dap_an_co_anh,
                "khung": ban_ghi,
            },
            f,
            ensure_ascii=False,
            indent=1,
        )

    dung_luong = sum(t.stat().st_size for t in goc.rglob("*")) / 1e6
    print(f"  {len(ban_ghi)}/{len(khung)} khung có ảnh, {thieu_anh} thiếu ảnh")
    print(f"  {dung_luong:.1f} MB")

    if thieu_anh:
        print(
            f"\n  {thieu_anh} khung bị bỏ vì máy này không có ảnh "
            f"({len(video_thieu)} video). Gói chỉ soi được phần shard đang có."
        )
    if dap_an_co_anh is False:
        print(
            f"\n  CHÚ Ý: video đáp án {video_dap_an} KHÔNG có ảnh trên máy này.\n"
            "  Gói này KHÔNG THỂ chứa khung đúng, nên đừng đọc kết quả thành\n"
            "  'mạch trượt' — nó chỉ nói shard thiếu. Nhờ người giữ shard đó\n"
            "  chạy lệnh này rồi gửi tệp zip."
        )

    if args.tai_cho:
        print("\nMở FiftyOne tại chỗ:")
        print(f"  python -c \"import sys; sys.path.insert(0,'.'); "
              f"from scripts.nap_fiftyone import nap; nap(r'{goc}').launch()\"")
        return 0

    tep_zip = goc.with_suffix(".zip")
    with zipfile.ZipFile(tep_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for t in goc.rglob("*"):
            if t.is_file():
                z.write(t, t.relative_to(goc))

    print(f"\nĐã gói: {tep_zip}  ({tep_zip.stat().st_size / 1e6:.1f} MB)")
    print("\nĐưa lên Colab: tải tệp zip này lên, rồi chạy notebook")
    print("  notebooks/soi_ung_vien_colab.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
