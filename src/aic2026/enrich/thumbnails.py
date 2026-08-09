"""
src/aic2026/enrich/thumbnails.py -- Nghi giu, khong ai sua truc tiep.

Sinh anh thu nho va cung cap duong dan chuan cho ca nhom.
Hop dong muc 5.3: thumbnail_path(video_id, n) -> Path

Quy cach da chot:
  - Canh dai 224 diem anh, giu ti le goc
  - JPEG chat luong 85
  - Ten tep giu nguyen so thu tu bon chu so cua anh goc
  - Cau truc: derived/thumbnails/<video_id>/<n bon chu so>.jpg

File nay la NGUON DUY NHAT cho ca viec quet thu muc va viec sinh anh.
Task 8 (OCR) phai goi lai iter_video_keyframes() o day, KHONG tu viet
vong quet rieng -- tranh moi ben quet mot kieu roi lech ket qua nhau.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterator

from PIL import Image

# Giu dung kieu import ma cac tep khac trong src/ dang dung.
from src.aic2026.paths import KEYFRAMES_DIR, THUMBNAILS_DIR, VIDEO_ID_RE

THUMB_LONG_EDGE = 224
JPEG_QUALITY = 85
FRAME_DIGITS = 4  # quy tac dat ten so 2: so thu tu anh luon bon chu so
MAX_SCAN_DEPTH = 4  # zip cua BTC giai nen ra thu muc long nhieu tang


# --------------------------------------------------------------------------
# Hop dong voi cac nhanh khac
# --------------------------------------------------------------------------

def thumbnail_path(video_id: str, n: int) -> Path:
    """Duong dan anh thu nho cua anh thu n trong video.

    Day la NGUON DUY NHAT sinh duong dan anh thu nho. Giao dien cua Thi
    va moi cho khac deu goi ham nay, khong ai tu ghep chuoi duong dan.
    """
    return THUMBNAILS_DIR / video_id / f"{n:0{FRAME_DIGITS}d}.jpg"


def frame_number(keyframe_path: Path) -> int | None:
    """Doc so thu tu anh tu ten tep.

    Day la NGUON DUY NHAT kiem tra quy cach ten tep anh goc (so 2 o dau
    file nay). Tra None khi ten khong phai so HOAC khong dung du
    FRAME_DIGITS chu so -- ca hai truong hop deu bi coi la "lech quy
    cach" va khong ai duoc tu kiem tra lai o noi khac (truoc day
    make_thumbnails.py tu kiem tra them mot lan, de sai lech neu ai do
    sua quy tac o day ma quen sua ben script).
    """
    stem = keyframe_path.stem
    if not stem.isdigit():
        return None
    if len(stem) != FRAME_DIGITS:
        return None
    return int(stem)


# --------------------------------------------------------------------------
# Vong quet dung chung -- viec 8 (OCR) goi lai dung cac ham nay,
# KHONG viet vong quet thu hai.
# --------------------------------------------------------------------------

def iter_video_dirs(root: Path | None = None) -> list[Path]:
    """Liet ke thu muc video trong kho anh goc, quet de quy toi 4 tang.

    Tra ve danh sach da sap xep theo video_id, khong trung. Khong duoc
    gia dinh KEYFRAMES_DIR.iterdir() ra thang thu muc video: zip cua
    BTC hoac cach giai nen cua tung nguoi co the con long them tang
    (vd raw/keyframes/Keyframes_L23/keyframes/L23_V001/...).
    """
    root = root or KEYFRAMES_DIR
    if not root.exists():
        raise FileNotFoundError(
            f"Khong thay thu muc anh goc: {root}\n"
            f"Kiem lai .env va xem da giai nen anh chua."
        )

    found: dict[str, Path] = {}
    duplicates: list[str] = []

    def walk(current: Path, depth: int) -> None:
        if depth > MAX_SCAN_DEPTH:
            return
        for child in sorted(current.iterdir()):
            if not child.is_dir():
                continue
            # fullmatch, khong dung match(): tranh lot ten kieu
            # L21_V001_backup (match() chi can khop tu dau chuoi).
            if VIDEO_ID_RE.fullmatch(child.name):
                if child.name in found and found[child.name] != child:
                    duplicates.append(f"{child.name}: {found[child.name]} va {child}")
                else:
                    found.setdefault(child.name, child)
            else:
                walk(child, depth + 1)

    walk(root, 0)

    if duplicates:
        # Truoc day cho nay dung setdefault() va lang le bo qua ban
        # trung -- nguy hiem vi neu ai do chua don phang thu muc long
        # (dung nhu ca cua Nghi o Task 6), mot video co the bi quet
        # tu hai noi khac nhau ma khong ai biet chi mot ban duoc dung.
        warnings.warn(
            "Trung video_id o nhieu thu muc, chi dung ban tim thay dau tien:\n  "
            + "\n  ".join(duplicates),
            stacklevel=2,
        )

    return [found[video_id] for video_id in sorted(found)]


def iter_keyframes(video_dir: Path) -> Iterator[Path]:
    """Liet ke anh trong mot thu muc video, da sap xep theo ten."""
    for path in sorted(video_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg"):
            yield path


def iter_video_keyframes(root: Path | None = None) -> Iterator[tuple[str, list[Path]]]:
    """Quet mot lan, dung chung cho Task 6 (thumbnail) va Task 8 (OCR).

    Sinh (video_id, danh sach anh jpg da sap xep) cho tung video. Day la
    diem vao duy nhat ma script nen goi: khong ai duoc tu ghep
    iter_video_dirs() + iter_keyframes() rieng trong script cua minh nua,
    vi lam vay de moi ben ghep logic khac nhau (vd mot ben quen loc
    _backup, mot ben quen sort).

    Loi doc tung thu muc rieng le (vd mat quyen doc, symlink hong)
    KHONG duoc lam vang ca luot quet -- chi bo qua thu muc do va canh
    bao, giong nguyen tac "mot anh hong khong duoc lam vang ca lo".
    """
    for video_dir in iter_video_dirs(root):
        try:
            frames = list(iter_keyframes(video_dir))
        except OSError as err:
            warnings.warn(f"Khong doc duoc thu muc {video_dir}: {err}", stacklevel=2)
            continue
        yield video_dir.name, frames


# --------------------------------------------------------------------------
# Sinh anh
# --------------------------------------------------------------------------

def make_thumbnail(src_path: Path, dst_path: Path) -> None:
    """Sinh mot anh thu nho. Ghi nguyen tu: ghi .tmp roi doi ten.

    Ghi thang vao dst_path la sai: dut giua chung se de lai anh cut,
    va lan chay sau se coi nhu da xong ma bo qua vinh vien.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")

    try:
        with Image.open(src_path) as img:
            # draft() bao bo giai ma JPEG tra ve ban nho hon san,
            # nhanh khoang gap doi ma khong doi chat luong sau khi resize.
            img.draft("RGB", (THUMB_LONG_EDGE, THUMB_LONG_EDGE))
            img = img.convert("RGB")

            w, h = img.size
            if w >= h:
                new_w = THUMB_LONG_EDGE
                new_h = max(1, round(h * THUMB_LONG_EDGE / w))
            else:
                new_h = THUMB_LONG_EDGE
                new_w = max(1, round(w * THUMB_LONG_EDGE / h))

            img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(tmp_path, "JPEG", quality=JPEG_QUALITY, optimize=True)

        os.replace(tmp_path, dst_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
