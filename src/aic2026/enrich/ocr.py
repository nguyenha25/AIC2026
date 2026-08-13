"""
Task 8 -- Trich xuat OCR tu keyframes.

GIA DINH VE INTERFACE (kiem tra lai neu khac thuc te trong repo cua ban):

- aic2026.enrich.thumbnails.iter_video_keyframes(keyframes_base: Path)
      -> Iterable[Tuple[str, List[Path]]]
  Ham quet dung chung voi Task 6, tra ve (video_id, danh sach duong dan .jpg
  da sap xep theo n), quet de quy toi da MAX_SCAN_DEPTH, va tu phat
  warnings.warn() khi mot video_id xuat hien o nhieu nhanh thu muc (trung lap).
  ==> ocr.py KHONG con tu viet lai logic quet thu muc nua, de tranh xu ly
      trung video_id ma khong duoc canh bao.

- aic2026.frame_map.FrameMap.frame_idx_of(n: int) -> int
  Co the raise (KeyError/IndexError/ValueError/...) neu n khong co trong map.
  Ham nay duoc goi ben trong try/except o muc TUNG FRAME, khong phai muc
  toan video, de mot frame loi khong lam mat ca video.

Neu chu ky thuc te khac (vi du iter_video_keyframes tra ve them thong tin
khac, hoac frame_map co ten khac), chinh lai phan goi ham ben duoi cho khop.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENGINE_NAME = "easyocr-1.7.1"


def _atomic_write_jsonl(records: Iterable[Dict[str, Any]], out_file: Path) -> int:
    """Ghi jsonl an toan: ghi vao {out_file}.tmp truoc, chi os.replace() sang
    ten that KHI DA GHI XONG toan bo. Nho vay neu tien trinh bi ngat giua
    chung (mat dien, Colab timeout, Ctrl+C...), {out_file} khong bao gio o
    trang thai do dang -- is_jsonl_valid() se khong bi danh lua tuong da
    xong, va lan chay sau se tu dong lam lai dung video do.
    """
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    count = 0
    with open(tmp_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp_file, out_file)
    return count


class OCRExtractor:
    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        # import tre (lazy) de --dry-run trong run_ocr_batch.py khong can
        # cai/khoi tao easyocr (nang va cham tren may khong GPU).
        import easyocr

        self.reader = easyocr.Reader(languages or ["vi", "en"], gpu=gpu)

    def _ocr_one_image(self, img_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Chay OCR tren 1 anh. CO THE RAISE -- ben goi chiu trach nhiem
        co lap loi (xem process_video_keyframes)."""
        results = self.reader.readtext(str(img_path))
        boxes: List[Dict[str, Any]] = []
        text_list: List[str] = []
        for bbox, text, conf in results:
            text_list.append(text)
            x1, y1 = int(bbox[0][0]), int(bbox[0][1])
            x2, y2 = int(bbox[2][0]), int(bbox[2][1])
            boxes.append({"text": text, "bbox": [x1, y1, x2, y2], "conf": float(conf)})
        return " ".join(text_list), boxes

    def process_video_keyframes(
        self,
        video_id: str,
        img_paths: List[Path],
        output_dir: Path,
        frame_map,
        error_log_path: Optional[Path] = None,
    ) -> Path:
        """
        Trich xuat chu cho 1 video, ghi ra {output_dir}/{video_id}.jsonl.

        CO LAP LOI THEO TUNG ANH (khac ban cu -- ban cu 1 anh loi la mat ca
        video): neu OCR 1 anh bi loi (anh hong, doc khong duoc...), ham ghi
        lai 1 record voi text/boxes rong kem field "error", RoI TIEP TUC cac
        anh con lai -- khong huy ca video.

        Rieng loi tra cuu frame_idx_of(n): neu n khong co trong FrameMap,
        frame do bi BO QUA khoi output (KHONG duoc doan/gan frame_idx gia,
        vi frame_idx sai se lam hong bai nop) va duoc ghi vao error_log de
        ban kiem tra thu cong.
        """
        if not img_paths:
            raise FileNotFoundError(f"Khong co anh .jpg nao cho video_id={video_id}")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{video_id}.jsonl"
        errors: List[str] = []

        def _records():
            for img_path in sorted(img_paths, key=lambda p: p.stem):
                try:
                    n = int(img_path.stem)
                except ValueError:
                    errors.append(f"{video_id}: ten anh khong phai so: {img_path.name}")
                    continue

                try:
                    frame_idx = int(frame_map.frame_idx_of(n))
                except Exception as e:  # noqa: BLE001 - day la bien co lap loi co chu dich
                    errors.append(
                        f"{video_id} n={n}: frame_idx_of loi ({e!r}) -- BO QUA frame nay, "
                        f"KHONG suy doan frame_idx."
                    )
                    continue

                try:
                    full_text, boxes = self._ocr_one_image(img_path)
                    yield {
                        "video_id": video_id,
                        "n": n,
                        "frame_idx": frame_idx,
                        "text": full_text,
                        "boxes": boxes,
                        "engine": ENGINE_NAME,
                    }
                except Exception as e:  # noqa: BLE001
                    errors.append(
                        f"{video_id} n={n}: OCR loi ({e!r}) -- ghi record rong, giu du frame."
                    )
                    yield {
                        "video_id": video_id,
                        "n": n,
                        "frame_idx": frame_idx,
                        "text": "",
                        "boxes": [],
                        "engine": ENGINE_NAME,
                        "error": repr(e),
                    }

        count = _atomic_write_jsonl(_records(), out_file)

        if errors:
            logger.warning("%s: %d loi trong qua trinh OCR (xem error log)", video_id, len(errors))
            if error_log_path:
                error_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(error_log_path, "a", encoding="utf-8") as ef:
                    for line in errors:
                        ef.write(line + "\n")

        if count == 0:
            # Toan bo frame deu loi frame_idx_of -> khong co gi de ghi.
            # Khong de lai file rong gay hieu nham la "da xong".
            if out_file.exists():
                out_file.unlink()
            raise RuntimeError(
                f"{video_id}: 0 record ghi duoc (toan bo {len(img_paths)} frame loi "
                f"frame_idx_of). Kiem tra FrameMap cho video nay truoc khi chay lai."
            )

        return out_file