"""
scripts/make_thumbnails.py -- Viec 6, lenh chay tay.

Sinh anh thu nho cho phan keyframes co san tren may nay.
Dung nguyen tac: AI TAI PHAN NAO THI XU LY PHAN DO.

Moi quy cach va viec quet nam trong src/aic2026/enrich/thumbnails.py, tep
nay chi lo goi lai ham quet chung, dem so, bat loi va ghi tep ma kiem tra.
Khong tu viet vong quet hay tu ghep duong dan o day.

Cach chay (tu goc repo, KHONG chay truc tiep bang python):
    python -m scripts.make_thumbnails
    python -m scripts.make_thumbnails --hash        # them cot sha256 vao manifest
    python -m scripts.make_thumbnails --overwrite   # sinh lai tu dau
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from src.aic2026.enrich.thumbnails import (
    FRAME_DIGITS,
    THUMBNAILS_DIR,
    frame_number,
    iter_video_keyframes,
    make_thumbnail,
    thumbnail_path,
)

MANIFEST_NAME = "_manifest.csv"
MAX_REPORTED_ERRORS = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dir(paths: list[Path]) -> str:
    """Ma kiem tra cho ca thu muc: bam ten tep + ma bam cua tung tep."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Sinh anh thu nho (viec 6)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Sinh lai ca nhung anh da co")
    parser.add_argument("--hash", action="store_true",
                        help="Tinh sha256 tung thu muc de doi chieu qua Drive")
    parser.add_argument("--limit", type=int, default=0,
                        help="Chi xu ly N video dau, de chay thu")
    args = parser.parse_args()

    try:
        # Goi mot lan duy nhat ham quet chung -- khong tu ghep
        # iter_video_dirs() + iter_keyframes() rieng o day nua.
        video_data = list(iter_video_keyframes())
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    if args.limit:
        video_data = video_data[: args.limit]

    if not video_data:
        print("Khong tim thay thu muc video nao. Kiem lai da giai nen anh chua.",
              file=sys.stderr)
        return 1

    print(f"Tim thay {len(video_data)} thu muc video.")

    total_created = 0
    total_skipped = 0
    errors: list[str] = []
    name_warnings: list[str] = []
    manifest_rows: list[dict[str, object]] = []

    for i, (video_id, keyframes) in enumerate(video_data, 1):
        if not keyframes:
            name_warnings.append(f"{video_id}: thu muc rong")

        created = skipped = failed = 0
        outputs: list[Path] = []

        for keyframe in keyframes:
            # frame_number() la noi DUY NHAT kiem tra quy cach ten tep
            # (xem enrich/thumbnails.py) -- o day khong kiem tra lai.
            n = frame_number(keyframe)
            if n is None:
                name_warnings.append(
                    f"{video_id}/{keyframe.name}: ten khong dung quy cach {FRAME_DIGITS} chu so"
                )
                continue

            out_path = thumbnail_path(video_id, n)
            outputs.append(out_path)

            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            # Bat loi TUNG anh rieng: mot anh hong khong duoc lam vang
            # ca luot chay hang chuc nghin anh.
            try:
                make_thumbnail(keyframe, out_path)
                created += 1
            except Exception as err:
                failed += 1
                errors.append(f"{video_id}/{keyframe.name}: {type(err).__name__}: {err}")

        total_created += created
        total_skipped += skipped

        existing = [p for p in outputs if p.exists()]
        manifest_rows.append({
            "video_id": video_id,
            "n_files": len(existing),
            "bytes": sum(p.stat().st_size for p in existing),
            "sha256": sha256_dir(existing) if args.hash else "",
        })

        note = f" | LOI {failed}" if failed else ""
        print(f"[{i}/{len(video_data)}] {video_id}: "
              f"moi {created}, bo qua {skipped}, tong {len(keyframes)}{note}")

    manifest_path = THUMBNAILS_DIR / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "n_files", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    total_files = sum(int(row["n_files"]) for row in manifest_rows)
    total_bytes = sum(int(row["bytes"]) for row in manifest_rows)

    print(f"\n--- Tong ket ---")
    print(f"Video          : {len(video_data)}")
    print(f"Anh tao moi    : {total_created}")
    print(f"Anh bo qua     : {total_skipped}")
    print(f"Anh hien co    : {total_files} ({total_bytes / 1e9:.2f} GB)")
    print(f"Ma kiem tra    : {manifest_path}")

    if name_warnings:
        print(f"\nCANH BAO TEN TEP: {len(name_warnings)} cho. "
              f"Nhung anh nay CHUA duoc sinh thu nho:")
        for line in name_warnings[:MAX_REPORTED_ERRORS]:
            print(f"  {line}")
        if len(name_warnings) > MAX_REPORTED_ERRORS:
            print(f"  ... con {len(name_warnings) - MAX_REPORTED_ERRORS} dong nua")

    if errors:
        print(f"\nLOI KHI SINH ANH: {len(errors)} tep:")
        for line in errors[:MAX_REPORTED_ERRORS]:
            print(f"  {line}")
        if len(errors) > MAX_REPORTED_ERRORS:
            print(f"  ... con {len(errors) - MAX_REPORTED_ERRORS} dong nua")
        print("Chay lai lenh nay de thu lai dung nhung tep con thieu.")

    return 1 if (errors or name_warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
