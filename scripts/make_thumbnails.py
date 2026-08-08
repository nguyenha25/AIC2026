"""
Task 6 — Sinh ảnh thu nhỏ (cạnh dài 224px) cho phần keyframes đã tải.
Chỉ xử lý phần có sẵn trong raw/keyframes/ trên máy này (đúng nguyên tắc:
AI TẢI PHẦN NÀO THÌ XỬ LÝ PHẦN ĐÓ).

Quy cách đã chốt:
  - Cạnh dài = 224px, giữ tỉ lệ gốc
  - JPEG quality = 85
  - Tên file giữ nguyên
  - Cấu trúc: derived/thumbnails/<video_id>/<tên_file_gốc>.jpg
"""
from PIL import Image
from src.aic2026.paths import KEYFRAMES_DIR, THUMBNAILS_DIR, VIDEO_ID_RE
THUMB_LONG_EDGE = 224
JPEG_QUALITY = 85


def make_thumbnail(src_path, dst_path):
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if w >= h:
            new_w = THUMB_LONG_EDGE
            new_h = round(h * THUMB_LONG_EDGE / w)
        else:
            new_h = THUMB_LONG_EDGE
            new_w = round(w * THUMB_LONG_EDGE / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, "JPEG", quality=JPEG_QUALITY)


def main():
    video_dirs = sorted(
        d for d in KEYFRAMES_DIR.iterdir()
        if d.is_dir() and VIDEO_ID_RE.match(d.name)
    )
    print(f"Tìm thấy {len(video_dirs)} thư mục video trong {KEYFRAMES_DIR}")

    total_done = 0
    total_skipped = 0
    for i, video_dir in enumerate(video_dirs, 1):
        out_dir = THUMBNAILS_DIR / video_dir.name
        jpgs = sorted(video_dir.glob("*.jpg"))
        for jpg in jpgs:
            out_path = out_dir / jpg.name
            if out_path.exists():
                total_skipped += 1
                continue
            make_thumbnail(jpg, out_path)
            total_done += 1
        print(f"[{i}/{len(video_dirs)}] {video_dir.name}: {len(jpgs)} ảnh")

    print(f"\nXong. Đã tạo mới {total_done} ảnh, bỏ qua {total_skipped} ảnh đã có sẵn.")


if __name__ == "__main__":
    main()