"""
Việc 15 — Đóng gói derived/asr/ để tải lên thư mục Drive chung cho Nguyên gộp.

Ra ba tệp, đặt cạnh nhau:
    asr_<tên>_<mã>.zip            các tệp .jsonl (KHÔNG kèm nhật ký, mẫu kiểm)
    asr_<tên>_<mã>.manifest.csv   mỗi video một dòng — để Nguyên soát khi gộp
    asr_<tên>_<mã>.zip.sha256     mã kiểm, định dạng chuẩn của sha256sum

Vì sao cần manifest chứ không chỉ cần zip: gộp bốn phần là ghép bốn thư mục
lại. Rủi ro thật không phải tệp hỏng mà là TRÙNG video_id giữa hai người (một
người chạy nhầm sang phần người khác) hoặc THIẾU video mà không ai nhận ra.
Manifest cho phép so bốn tệp .csv với nhau trong ba mươi giây, không cần giải nén.

KIỂM TRƯỚC KHI GÓI: mọi dòng phải đủ khoá theo docs/schema/README.md mục 3.
Có một dòng sai là dừng, không gói — gửi lên Drive rồi mới phát hiện thì phải
làm phiền cả ba người kia tải lại.

Cách chạy (PowerShell):
    python -m scripts.goi_asr_len_drive --ten nhan
    python -m scripts.goi_asr_len_drive --ten nhan --ra D:\\aic-data\\goi_drive
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

try:
    from src.aic2026.paths import ASR_DIR, DATA_ROOT
except ImportError as e:
    print("Không import được aic2026. Đã chạy 'pip install -e .' chưa?")
    print(f"Chi tiết: {e}")
    sys.exit(1)

KHOA_BAT_BUOC = ("video_id", "start", "end", "text",
                 "frame_idx_start", "frame_idx_end", "lang", "engine")


def bam_sha256(duong_dan: Path) -> str:
    h = hashlib.sha256()
    with duong_dan.open("rb") as f:
        for khoi in iter(lambda: f.read(1024 * 1024), b""):
            h.update(khoi)
    return h.hexdigest()


def soat_mot_tep(duong_dan: Path) -> dict:
    """Đọc và kiểm một tệp .jsonl. Sai mẫu thì ném ValueError kèm số dòng."""
    video_id = duong_dan.stem
    so_dong = 0
    so_co_chu = 0
    giay_dau: float | None = None
    giay_cuoi: float | None = None
    cac_engine: set[str] = set()

    with duong_dan.open("r", encoding="utf-8") as f:
        for stt, dong in enumerate(f, start=1):
            if not dong.strip():
                continue
            so_dong += 1
            try:
                ban_ghi = json.loads(dong)
            except ValueError as e:
                raise ValueError(f"{duong_dan.name} dòng {stt}: JSON hỏng — {e}") from e

            thieu = [k for k in KHOA_BAT_BUOC if k not in ban_ghi]
            if thieu:
                raise ValueError(f"{duong_dan.name} dòng {stt}: thiếu khoá {thieu}")

            if ban_ghi["video_id"] != video_id:
                raise ValueError(
                    f"{duong_dan.name} dòng {stt}: video_id trong dòng là "
                    f"'{ban_ghi['video_id']}' nhưng tên tệp là '{video_id}'"
                )

            s, e = float(ban_ghi["start"]), float(ban_ghi["end"])
            if e < s:
                raise ValueError(f"{duong_dan.name} dòng {stt}: end < start")

            cac_engine.add(ban_ghi["engine"])
            if (ban_ghi["text"] or "").strip():
                so_co_chu += 1
                giay_dau = s if giay_dau is None else min(giay_dau, s)
                giay_cuoi = e if giay_cuoi is None else max(giay_cuoi, e)

    return {
        "video_id": video_id,
        "so_dong": so_dong,
        "so_doan_co_chu": so_co_chu,
        "giay_dau": "" if giay_dau is None else f"{giay_dau:.2f}",
        "giay_cuoi": "" if giay_cuoi is None else f"{giay_cuoi:.2f}",
        "engine": "|".join(sorted(cac_engine)),
        "sha256": bam_sha256(duong_dan),
    }


def main():
    p = argparse.ArgumentParser(description="Đóng gói derived/asr/ để tải lên Drive chung")
    p.add_argument("--ten", required=True, help="Tên người gửi, viết thường không dấu, vd: nhan")
    p.add_argument("--ra", default=None, help="Thư mục ghi gói. Mặc định <gốc dữ liệu>/goi_drive")
    p.add_argument("--tien-to", nargs="+", default=None, help="Chỉ gói các mã này, vd L23 L27")
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    cac_tep = [x for x in sorted(ASR_DIR.glob("*.jsonl")) if not x.name.startswith("_")]
    if args.tien_to:
        cac_tep = [x for x in cac_tep if x.stem.startswith(tuple(args.tien_to))]

    if not cac_tep:
        print(f"Không có tệp .jsonl nào trong {ASR_DIR} để gói.")
        sys.exit(1)

    print(f"Đang soát {len(cac_tep)} tệp trước khi gói ...")
    cac_dong: list[dict] = []
    for tep in cac_tep:
        try:
            cac_dong.append(soat_mot_tep(tep))
        except ValueError as e:
            print(f"\nDỪNG — không gói. {e}")
            print("Sửa tệp đó (hoặc xoá rồi chạy lại Việc 15 cho video đó) rồi gói lại.")
            sys.exit(1)
    print("Soát xong, mọi dòng đúng mẫu.\n")

    ma = sorted({d["video_id"].split("_")[0] for d in cac_dong})
    ten_goi = f"asr_{args.ten}_{'_'.join(ma)}"

    thu_muc_ra = Path(args.ra) if args.ra else DATA_ROOT / "goi_drive"
    thu_muc_ra.mkdir(parents=True, exist_ok=True)

    duong_dan_zip = thu_muc_ra / f"{ten_goi}.zip"
    duong_dan_manifest = thu_muc_ra / f"{ten_goi}.manifest.csv"

    with duong_dan_manifest.open("w", encoding="utf-8", newline="") as f:
        ghi = csv.DictWriter(f, fieldnames=list(cac_dong[0].keys()))
        ghi.writeheader()
        ghi.writerows(cac_dong)

    with zipfile.ZipFile(duong_dan_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for tep in cac_tep:
            z.write(tep, arcname=f"asr/{tep.name}")
        z.write(duong_dan_manifest, arcname=duong_dan_manifest.name)

    bam = bam_sha256(duong_dan_zip)
    # Định dạng chuẩn của sha256sum: "<mã>  <tên tệp>" — hai dấu cách.
    # Kiểm lại được bằng: sha256sum -c <tệp>.sha256 (Git Bash), hoặc
    # Get-FileHash <tệp>.zip -Algorithm SHA256 (PowerShell).
    (thu_muc_ra / f"{ten_goi}.zip.sha256").write_text(
        f"{bam}  {duong_dan_zip.name}\n", encoding="utf-8"
    )

    tong_doan = sum(d["so_doan_co_chu"] for d in cac_dong)
    kich_thuoc = duong_dan_zip.stat().st_size / (1024 * 1024)

    print("=" * 66)
    print("ĐÃ ĐÓNG GÓI")
    print("=" * 66)
    print(f"Gói       : {duong_dan_zip}  ({kich_thuoc:.1f} MB)")
    print(f"Manifest  : {duong_dan_manifest}")
    print(f"Mã kiểm   : {duong_dan_zip.name}.sha256")
    print(f"Nội dung  : {len(cac_dong)} video, {tong_doan} đoạn có chữ, mã {', '.join(ma)}")
    print("=" * 66)
    print("Tải CẢ BA tệp lên thư mục Drive chung, rồi báo Nguyên.")
    print("Nguyên gộp bằng cách chép các .jsonl của bốn người vào cùng derived/asr/,")
    print("so bốn tệp manifest để chắc không trùng video_id và không thiếu video,")
    print("rồi chạy: python -m scripts.nap_asr_vao_fts")


if __name__ == "__main__":
    main()
