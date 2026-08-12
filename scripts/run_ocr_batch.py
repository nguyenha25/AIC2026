"""
Task 8 -- Chay OCR hang loat tren keyframes.

Chay dung quy uoc cua nhom:
    python -m scripts.run_ocr_batch --dry-run      # xem truoc, KHONG OCR gi ca
    python -m scripts.run_ocr_batch --limit 3       # thu that tren 3 video
    python -m scripts.run_ocr_batch                 # chay that toan bo

GIA DINH INTERFACE (xem chi tiet trong src/aic2026/enrich/ocr.py):
    aic2026.enrich.thumbnails.iter_video_keyframes(keyframes_base: Path)
        -> Iterable[Tuple[str, List[Path]]]
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_ocr_batch")


def is_jsonl_valid(filepath: Path) -> bool:
    """Kiem tra tep jsonl co ton tai, khong rong va dong cuoi la JSON hop le."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return False
    try:
        with open(filepath, "rb") as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode("utf-8")
            data = json.loads(last_line)
            return "video_id" in data and "frame_idx" in data
    except Exception:
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Task 8 - chay OCR hang loat tren keyframes")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Chi liet ke video/so anh se xu ly, KHONG khoi tao EasyOCR, KHONG OCR gi ca.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chi xu ly N video dau tien (de thu truoc khi chay full tren Colab).",
    )
    p.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT", r"D:\aic-data"))
    p.add_argument("--languages", nargs="+", default=["vi", "en"])
    return p.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    keyframes_base = data_root / "raw" / "keyframes"
    output_dir = Path("derived/ocr")
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = output_dir / "_errors.log"

    # Scanner DUNG CHUNG voi Task 6 -- tu dong canh bao (warnings.warn) khi
    # mot video_id xuat hien o nhieu nhanh thu muc, khong tu viet lai logic
    # quet nua.
    from aic2026.enrich.thumbnails import iter_video_keyframes

    logger.info("Quet thu muc keyframes: %s", keyframes_base)
    all_videos = list(iter_video_keyframes(keyframes_base))
    total_videos = len(all_videos)
    total_images = sum(len(paths) for _, paths in all_videos)
    logger.info("--> Tim thay %d video, %d anh tong cong.", total_videos, total_images)

    if args.limit:
        all_videos = all_videos[: args.limit]
        logger.info("--limit=%d -> chi xet %d video", args.limit, len(all_videos))

    pending = []
    skip_count = 0
    for video_id, img_paths in all_videos:
        out_file = output_dir / f"{video_id}.jsonl"
        if is_jsonl_valid(out_file):
            skip_count += 1
            continue
        pending.append((video_id, img_paths))

    logger.info("Da xong (skip): %d | Can xu ly: %d", skip_count, len(pending))

    if args.dry_run:
        print("\n=== DRY RUN -- khong co gi duoc OCR ===")
        print(f"Tong video trong nguon: {total_videos} ({total_images} anh)")
        print(f"Da co ket qua hop le (skip): {skip_count}")
        print(
            f"Se xu ly trong lan chay that: {len(pending)} video "
            f"({sum(len(p) for _, p in pending)} anh)"
        )
        if pending:
            sample_n = min(5, len(pending))
            print(f"\nMau {sample_n} video se chay dau tien:")
            for video_id, img_paths in pending[:sample_n]:
                print(f"  - {video_id}: {len(img_paths)} anh")
        return

    from aic2026.frame_map import FrameMap
    from aic2026.enrich.ocr import OCRExtractor

    import torch

    use_gpu = torch.cuda.is_available()
    logger.info("Khoi tao EasyOCR (GPU: %s)...", use_gpu)
    extractor = OCRExtractor(languages=args.languages, gpu=use_gpu)

    start_time = time.time()
    success_count = 0
    fail_count = 0

    for idx, (video_id, img_paths) in enumerate(pending, 1):
        t0 = time.time()
        try:
            frame_map = FrameMap.load(video_id)
            extractor.process_video_keyframes(
                video_id=video_id,
                img_paths=img_paths,
                output_dir=output_dir,
                frame_map=frame_map,
                error_log_path=error_log_path,
            )
            success_count += 1
            dt = time.time() - t0
            elapsed = time.time() - start_time
            avg = elapsed / idx
            eta_min = avg * (len(pending) - idx) / 60
            logger.info(
                "[%d/%d] %s OK (%.1fs) | ETA con lai: %.1f phut",
                idx,
                len(pending),
                video_id,
                dt,
                eta_min,
            )
        except Exception as e:
            fail_count += 1
            logger.error("[%d/%d] %s LOI: %r", idx, len(pending), video_id, e)
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"{video_id}: VIDEO LOI TOAN BO - {e!r}\n")

    elapsed = time.time() - start_time
    print(
        f"\nXONG! Thanh cong: {success_count}, Loi: {fail_count}, "
        f"Da skip truoc do: {skip_count}, Tong nguon: {total_videos}"
    )
    print(f"Thoi gian: {elapsed/60:.2f} phut")
    if fail_count or error_log_path.exists():
        print(f"Xem chi tiet loi tai: {error_log_path}")


if __name__ == "__main__":
    main()