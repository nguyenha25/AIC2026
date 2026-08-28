"""Nghiệm thu Việc 9: frames_dense tồn tại, đúng tên và không có lỗ >= 0,5 s."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.paths import FRAMES_DENSE_DIR  # noqa: E402
from scripts.trich_khung_day import (  # noqa: E402
    NGUONG_DAY_GIAY,
    TEN_MANIFEST,
    _fps,
    doc_danh_sach_video,
    kiem_danh_so,
)


def doc_cau_trake_dev(duong_dan: Path) -> list[dict]:
    """Đọc đúng các record ``chuoi_su_kien`` có nhãn ``cac_giai_doan``."""
    ra = []
    for so_dong, dong in enumerate(
        duong_dan.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not dong.strip():
            continue
        try:
            q = json.loads(dong)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{duong_dan.name} dòng {so_dong} sai JSON.") from exc
        if q.get("loai_truy_van") != "chuoi_su_kien":
            continue
        if not q.get("video_id") or not isinstance(q.get("cac_giai_doan"), list):
            raise ValueError(f"Câu {q.get('id', so_dong)} thiếu video_id/cac_giai_doan.")
        ra.append(q)
    if not ra:
        raise ValueError(f"{duong_dan} không có câu chuoi_su_kien nào.")
    return ra


def kiem_bao_phu_trake(duong_dan: Path) -> dict:
    """Mỗi cửa sổ đáp án TRAKE phải chứa ít nhất một ảnh dense."""
    cau = doc_cau_trake_dev(duong_dan)
    tong = 0
    trung = 0
    truot = []
    cache: dict[str, list[int]] = {}
    for q in cau:
        video_id = str(q["video_id"])
        if video_id not in cache:
            thu_muc = FRAMES_DENSE_DIR / video_id
            cache[video_id] = sorted(
                int(p.stem) for p in thu_muc.glob("*.jpg") if p.stem.isdigit()
            )
        chi_so = cache[video_id]
        for thu_tu, g in enumerate(q["cac_giai_doan"], 1):
            tong += 1
            dau, cuoi = int(g["frame_start"]), int(g["frame_end"])
            mau = [x for x in chi_so if dau <= x <= cuoi]
            if mau:
                trung += 1
            else:
                truot.append(
                    {
                        "query_id": str(q.get("id", "?")),
                        "video_id": video_id,
                        "event": thu_tu,
                        "frame_start": dau,
                        "frame_end": cuoi,
                    }
                )
    return {
        "so_cau": len(cau),
        "so_video": len(cache),
        "tong_cua_so": tong,
        "trung_cua_so": trung,
        "truot": truot,
        "dat": tong > 0 and trung == tong,
    }


def _doc_manifest(thu_muc: Path) -> tuple[dict | None, str | None]:
    tep = thu_muc / TEN_MANIFEST
    if not tep.exists():
        return None, f"thiếu {TEN_MANIFEST}"
    try:
        du_lieu = json.loads(tep.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{TEN_MANIFEST} hỏng: {exc}"
    if not isinstance(du_lieu, dict) or not isinstance(du_lieu.get("ranges"), list):
        return None, f"{TEN_MANIFEST} sai schema"
    return du_lieu, None


def kiem_video(video_id: str, kiem_anh: bool = True) -> dict:
    thu_muc = FRAMES_DENSE_DIR / video_id
    loi: list[str] = []
    canh_bao: list[str] = []
    if not thu_muc.is_dir():
        return {"video_id": video_id, "dat": False, "loi": ["chưa có thư mục"], "canh_bao": []}

    tep_anh = list(thu_muc.glob("*.jpg"))
    tep_sai_ten = sorted(p.name for p in tep_anh if not p.stem.isdigit())
    if tep_sai_ten:
        loi.append(f"{len(tep_sai_ten)} ảnh không mang tên frame_idx: {tep_sai_ten[:3]}")
    chi_so = sorted(int(p.stem) for p in tep_anh if p.stem.isdigit())
    if len(chi_so) < 2:
        loi.append("có ít hơn 2 ảnh")

    try:
        fps = _fps(video_id)
    except Exception as exc:
        fps = float("nan")
        loi.append(f"không đọc được fps từ frame_map: {exc}")

    manifest, loi_manifest = _doc_manifest(thu_muc)
    if loi_manifest:
        loi.append(loi_manifest)

    khoang_lon_nhat = 0.0
    so_vung = 0
    if manifest and math.isfinite(fps) and len(chi_so) >= 2:
        if str(manifest.get("video_id")) != video_id:
            loi.append(f"manifest ghi video_id={manifest.get('video_id')!r}")
        for i, vung in enumerate(manifest["ranges"], 1):
            try:
                dau = int(vung["frame_start"])
                cuoi = int(vung["frame_end"])
            except (KeyError, TypeError, ValueError):
                loi.append(f"vùng {i} trong manifest thiếu frame_start/frame_end")
                continue
            trong = [x for x in chi_so if dau <= x <= cuoi]
            if not trong or trong[0] != dau or trong[-1] != cuoi:
                loi.append(f"vùng {i} chưa phủ đủ hai đầu [{dau}, {cuoi}]")
                continue
            if len(trong) < 2:
                loi.append(f"vùng {i} có ít hơn 2 ảnh")
                continue
            so_vung += 1
            lon_nhat_khung = max(b - a for a, b in zip(trong, trong[1:]))
            khoang = lon_nhat_khung / fps
            khoang_lon_nhat = max(khoang_lon_nhat, khoang)
            if khoang >= NGUONG_DAY_GIAY:
                loi.append(
                    f"vùng {i} có lỗ lớn nhất {khoang:.3f}s "
                    f"({lon_nhat_khung} khung), cần < {NGUONG_DAY_GIAY}s"
                )
        if so_vung == 0:
            loi.append("manifest không có vùng trích hợp lệ")

    kiem_anh_kq = None
    if kiem_anh and not loi:
        kiem_anh_kq = kiem_danh_so(video_id)
        if kiem_anh_kq.get("kiem_duoc"):
            if not kiem_anh_kq.get("dat"):
                loi.append(
                    f"đối chiếu ảnh chỉ đúng {kiem_anh_kq.get('so_khop', 0)}/"
                    f"{kiem_anh_kq.get('so_mau', 0)} mẫu"
                )
        else:
            # Input Task 9 chỉ bắt buộc videos + map-keyframes; raw/keyframes
            # có thể chưa nằm trên máy. Vì vậy thiếu ảnh đối chiếu là cảnh báo,
            # không làm sai tiêu chí khoảng cách và schema.
            canh_bao.append(f"không đối chiếu ảnh BTC: {kiem_anh_kq.get('ly_do')}")

    return {
        "video_id": video_id,
        "dat": not loi,
        "so_anh": len(chi_so),
        "so_vung": so_vung,
        "khoang_lon_nhat_giay": khoang_lon_nhat,
        "loi": loi,
        "canh_bao": canh_bao,
        "kiem_anh": kiem_anh_kq,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Nghiệm thu Việc 9 — frames_dense")
    p.add_argument("--video", action="append", default=[], help="có thể lặp nhiều lần")
    p.add_argument("--danh-sach", type=Path, help="txt/csv/json/jsonl chứa video_id")
    p.add_argument(
        "--tep-dev",
        type=Path,
        help="dev_questions.jsonl; kiểm mỗi cửa sổ TRAKE có ít nhất một ảnh",
    )
    p.add_argument("--tat-kiem-anh", action="store_true", help="không so với keyframe BTC")
    args = p.parse_args()

    video_ids = list(args.video)
    if args.danh_sach:
        video_ids.extend(doc_danh_sach_video(args.danh_sach))
    if args.tep_dev:
        video_ids.extend(str(q["video_id"]) for q in doc_cau_trake_dev(args.tep_dev))
    video_ids = list(dict.fromkeys(video_ids))
    if not video_ids:
        video_ids = sorted(p.name for p in FRAMES_DENSE_DIR.iterdir() if p.is_dir()) \
            if FRAMES_DENSE_DIR.is_dir() else []
    if not video_ids:
        print(f"CHƯA ĐẠT: chưa có video nào trong {FRAMES_DENSE_DIR}")
        return 1

    ket_qua = [kiem_video(v, kiem_anh=not args.tat_kiem_anh) for v in video_ids]
    print("VIỆC 9 — NGHIỆM THU FRAMES_DENSE\n")
    for kq in ket_qua:
        nhan = "ĐẠT" if kq["dat"] else "CHƯA ĐẠT"
        print(
            f"{nhan:<9} {kq['video_id']:<12} {kq.get('so_anh', 0):>7,} ảnh | "
            f"{kq.get('so_vung', 0)} vùng | lỗ lớn nhất "
            f"{kq.get('khoang_lon_nhat_giay', float('inf')):.3f}s"
        )
        for loi in kq["loi"]:
            print(f"  LỖI: {loi}")
        for canh_bao in kq["canh_bao"]:
            print(f"  CẢNH BÁO: {canh_bao}")

    dat = sum(1 for kq in ket_qua if kq["dat"])
    print(f"\nKẾT LUẬN: {dat}/{len(ket_qua)} video đạt tiêu chí khoảng cách < 0,5 giây.")
    dat_dev = True
    if args.tep_dev:
        phu = kiem_bao_phu_trake(args.tep_dev)
        print(
            f"BAO PHỦ TRAKE: {phu['trung_cua_so']}/{phu['tong_cua_so']} cửa sổ "
            f"thuộc {phu['so_cau']} câu, {phu['so_video']} video."
        )
        for muc in phu["truot"][:20]:
            print(
                f"  TRƯỢT câu {muc['query_id']} E{muc['event']}: "
                f"{muc['video_id']} [{muc['frame_start']}, {muc['frame_end']}]"
            )
        dat_dev = bool(phu["dat"])
    return 0 if dat == len(ket_qua) and dat_dev else 1


if __name__ == "__main__":
    raise SystemExit(main())
