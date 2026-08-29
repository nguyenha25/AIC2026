"""Hàm thuần cho N-00: kiểm schema, chia theo video và đóng băng tệp.

Module này cố ý không import model, FAISS hay đường dẫn dữ liệu. Nhờ vậy phần
logic quan trọng nhất của N-00 kiểm thử được trên máy chưa có kho AIC.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable


LOAI_HOP_LE = {"mo_ta", "hoi_dap", "chuoi_su_kien"}


class LoiDongBang(ValueError):
    """Dữ liệu/cấu hình không đủ an toàn để đóng băng."""


def doc_jsonl(duong_dan: Path) -> list[dict]:
    """Đọc JSONL và báo đúng số dòng hỏng; không bỏ qua âm thầm."""
    if not duong_dan.is_file():
        raise LoiDongBang(f"Không thấy tệp dev: {duong_dan}")

    ket_qua: list[dict] = []
    with duong_dan.open("r", encoding="utf-8-sig") as f:
        for so_dong, dong in enumerate(f, 1):
            if not dong.strip():
                continue
            try:
                ban_ghi = json.loads(dong)
            except json.JSONDecodeError as exc:
                raise LoiDongBang(
                    f"{duong_dan.name} dòng {so_dong} sai JSON: {exc}"
                ) from exc
            if not isinstance(ban_ghi, dict):
                raise LoiDongBang(
                    f"{duong_dan.name} dòng {so_dong} phải là object JSON"
                )
            ket_qua.append(ban_ghi)
    if not ket_qua:
        raise LoiDongBang(f"{duong_dan.name} không có câu hỏi")
    return ket_qua


def _so_nguyen_khong_am(gia_tri: object, ten: str, query_id: str) -> int:
    if isinstance(gia_tri, bool):
        raise LoiDongBang(f"Câu {query_id}: {ten} phải là số nguyên")
    try:
        so = int(gia_tri)
    except (TypeError, ValueError) as exc:
        raise LoiDongBang(f"Câu {query_id}: {ten} phải là số nguyên") from exc
    if so < 0:
        raise LoiDongBang(f"Câu {query_id}: {ten} không được âm")
    return so


def kiem_schema(cac_cau: Iterable[dict]) -> list[dict]:
    """Kiểm schema thật của ``dev_questions.jsonl`` và ID không trùng."""
    da_thay: set[str] = set()
    ket_qua: list[dict] = []

    for vi_tri, q in enumerate(cac_cau, 1):
        query_id = str(q.get("id", "")).strip()
        if not query_id:
            raise LoiDongBang(f"Dòng dữ liệu {vi_tri}: thiếu id")
        if query_id in da_thay:
            raise LoiDongBang(f"Trùng id câu hỏi: {query_id}")
        da_thay.add(query_id)

        loai = str(q.get("loai_truy_van", "")).strip()
        if loai not in LOAI_HOP_LE:
            raise LoiDongBang(
                f"Câu {query_id}: loai_truy_van={loai!r}, cần một trong "
                f"{sorted(LOAI_HOP_LE)}"
            )
        if not str(q.get("cau_hoi", "")).strip():
            raise LoiDongBang(f"Câu {query_id}: cau_hoi rỗng")
        if not str(q.get("video_id", "")).strip():
            raise LoiDongBang(f"Câu {query_id}: thiếu video_id")

        if loai in {"mo_ta", "hoi_dap"}:
            dau = _so_nguyen_khong_am(q.get("frame_start"), "frame_start", query_id)
            cuoi = _so_nguyen_khong_am(q.get("frame_end"), "frame_end", query_id)
            if dau > cuoi:
                raise LoiDongBang(
                    f"Câu {query_id}: frame_start={dau} lớn hơn frame_end={cuoi}"
                )
            if loai == "hoi_dap" and not str(q.get("cau_tra_loi", "")).strip():
                raise LoiDongBang(f"Câu {query_id}: cau_tra_loi rỗng")
        else:
            cac_giai_doan = q.get("cac_giai_doan")
            if not isinstance(cac_giai_doan, list) or not cac_giai_doan:
                raise LoiDongBang(f"Câu {query_id}: cac_giai_doan phải là list không rỗng")
            for i, giai_doan in enumerate(cac_giai_doan, 1):
                if not isinstance(giai_doan, dict):
                    raise LoiDongBang(f"Câu {query_id}: event {i} phải là object")
                dau = _so_nguyen_khong_am(
                    giai_doan.get("frame_start"), f"event {i}.frame_start", query_id
                )
                cuoi = _so_nguyen_khong_am(
                    giai_doan.get("frame_end"), f"event {i}.frame_end", query_id
                )
                if dau > cuoi:
                    raise LoiDongBang(
                        f"Câu {query_id}: event {i} có frame_start > frame_end"
                    )
        ket_qua.append(dict(q))
    return ket_qua


def tach_tap_sach(
    cac_cau: Iterable[dict],
    kiem_bang_chung: Callable[[dict], tuple[bool, str]],
) -> tuple[list[dict], list[dict]]:
    """Giữ câu có evidence GT thật; câu loại luôn mang lý do."""
    sach: list[dict] = []
    loai: list[dict] = []
    for q in cac_cau:
        dat, ly_do = kiem_bang_chung(q)
        if dat:
            sach.append(dict(q))
        else:
            loai.append(
                {
                    "id": str(q["id"]),
                    "loai_truy_van": q["loai_truy_van"],
                    "video_id": q["video_id"],
                    "reason": ly_do or "khong_ro",
                }
            )
    return sach, loai


def _khoa_hash(seed: str, video_id: str) -> str:
    return hashlib.sha256(f"{seed}|{video_id}".encode("utf-8")).hexdigest()


def chia_theo_video(
    cac_cau: Iterable[dict],
    *,
    ti_le_holdout: float = 0.30,
    seed: str = "aic2026-n00-v1",
) -> tuple[list[dict], list[dict]]:
    """Chia tất định theo ``video_id`` và cố giữ mỗi loại ở cả hai tập.

    Không một video nào được xuất hiện ở cả tune và holdout. Với một loại chỉ
    có đúng một video thì không thể có mặt ở cả hai; manifest sẽ phản ánh cỡ
    mẫu thật thay vì nhân bản câu gây leakage.
    """
    if not 0 < ti_le_holdout < 1:
        raise LoiDongBang("ti_le_holdout phải nằm trong (0, 1)")

    cau = list(cac_cau)
    if not cau:
        raise LoiDongBang("Tập sạch rỗng, không thể chia")

    theo_video: dict[str, list[dict]] = defaultdict(list)
    for q in cau:
        theo_video[str(q["video_id"])].append(q)
    video = sorted(theo_video, key=lambda v: (_khoa_hash(seed, v), v))
    if len(video) == 1:
        return list(cau), []

    video_theo_loai: dict[str, set[str]] = defaultdict(set)
    for v, nhom in theo_video.items():
        for q in nhom:
            video_theo_loai[str(q["loai_truy_van"])].add(v)

    def bo_sung_duoc(v: str) -> bool:
        # Không lấy hết mọi video của bất kỳ loại nào sang holdout.
        for loai in {str(q["loai_truy_van"]) for q in theo_video[v]}:
            tat_ca = video_theo_loai[loai]
            if len(tat_ca) >= 2 and tat_ca <= (holdout | {v}):
                return False
        return True

    holdout: set[str] = set()
    # Mỗi loại có từ hai video trở lên phải có ít nhất một video holdout,
    # nhưng không được vì vậy mà lấy hết video của một loại khác.
    for loai in sorted(video_theo_loai):
        ung_vien = [v for v in video if v in video_theo_loai[loai]]
        if len(ung_vien) < 2 or any(v in holdout for v in ung_vien):
            continue
        chon = next((v for v in ung_vien if bo_sung_duoc(v)), None)
        if chon is not None:
            holdout.add(chon)

    muc_tieu = max(1, round(len(cau) * ti_le_holdout))

    def so_cau_holdout() -> int:
        return sum(len(theo_video[v]) for v in holdout)

    for v in video:
        if so_cau_holdout() >= muc_tieu:
            break
        if v not in holdout and bo_sung_duoc(v):
            holdout.add(v)

    if not holdout:
        holdout.add(video[0])
    if len(holdout) == len(video):
        holdout.remove(video[-1])

    tune = [q for q in cau if str(q["video_id"]) not in holdout]
    hold = [q for q in cau if str(q["video_id"]) in holdout]
    if set(str(q["video_id"]) for q in tune) & set(str(q["video_id"]) for q in hold):
        raise AssertionError("Lỗi nội bộ: split bị leakage video")
    return tune, hold


def dem_theo_loai(cac_cau: Iterable[dict]) -> dict[str, int]:
    dem = Counter(str(q["loai_truy_van"]) for q in cac_cau)
    return {k: dem.get(k, 0) for k in sorted(LOAI_HOP_LE)}


def sha256_file(duong_dan: Path) -> str:
    bam = hashlib.sha256()
    with duong_dan.open("rb") as f:
        for khoi in iter(lambda: f.read(1024 * 1024), b""):
            bam.update(khoi)
    return bam.hexdigest()


def ghi_json_atomic(duong_dan: Path, du_lieu: object) -> None:
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    tam = duong_dan.with_suffix(duong_dan.suffix + ".partial")
    tam.write_text(
        json.dumps(du_lieu, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tam, duong_dan)


def ghi_jsonl_atomic(duong_dan: Path, cac_dong: Iterable[dict]) -> None:
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    tam = duong_dan.with_suffix(duong_dan.suffix + ".partial")
    with tam.open("w", encoding="utf-8", newline="\n") as f:
        for dong in cac_dong:
            f.write(json.dumps(dong, ensure_ascii=False) + "\n")
    os.replace(tam, duong_dan)
