"""Việc 9 — trích khung dày cho các video nghi là đáp án TRAKE.

Đầu ra chuẩn::

    derived/frames_dense/<video_id>/<frame_idx 6 chữ số>.jpg

Tên ảnh là ``frame_idx`` phải nộp, không phải số thứ tự keyframe ``n``.
Mặc định lấy một ảnh mỗi 0,16 giây (luôn < 0,5 giây), đồng thời giữ thêm các
keyframe BTC nằm trong vùng trích để có thể đối chiếu độc lập cách đánh số.

Cách chạy thường dùng trên Windows PowerShell::

    # Một video, trích toàn bộ
    python -u -m scripts.trich_khung_day --video L23_V025

    # Nhiều video: tệp txt/csv/json/jsonl có cột hoặc khóa video_id
    python -u -m scripts.trich_khung_day --danh-sach config/trake_dense_videos.txt

    # Chỉ một đoạn để thử nhanh
    python -u -m scripts.trich_khung_day --video L23_V025 --giay 120 180

    # Bộ dev có nhãn thời gian: chỉ trích quanh các cửa sổ đáp án
    python -u -m scripts.trich_khung_day --tu-dev

Sau khi trích, chạy nghiệm thu::

    python -u -m scripts.verify_task9_acceptance --video L23_V025
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.frame_map import load_frame_map  # noqa: E402
from aic2026.paths import (  # noqa: E402
    DEV_QUERIES_PATH,
    FRAMES_DENSE_DIR,
    keyframe_image,
    video_file,
)

# dev_questions hiện có cửa sổ TRAKE ngắn nhất chỉ 4 frame. Bước 0,4 giây
# (10 frame ở 25 fps) đã bỏ lọt 6/50 cửa sổ. Bước 0,16 giây cho tối đa 4
# frame ở 25/30 fps, nên mọi cửa sổ dài >= 4 frame đều có ít nhất một mẫu,
# không phụ thuộc pha bắt đầu của lịch lấy mẫu.
BUOC_GIAY_MAC_DINH = 0.16
NGUONG_DAY_GIAY = 0.5
DEM_KHUNG_MAC_DINH = 90
TEN_MANIFEST = "manifest.json"


def _fps(video_id: str) -> float:
    bang = load_frame_map()
    nhom = bang[bang["video_id"] == video_id]
    if nhom.empty:
        raise KeyError(f"{video_id} không có trong frame_map.")
    fps = float(nhom["fps"].iloc[0])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps của {video_id} không hợp lệ: {fps!r}")
    return fps


def tinh_buoc_khung(fps: float, buoc_giay: float = BUOC_GIAY_MAC_DINH) -> int:
    """Đổi bước theo giây sang số khung, bảo đảm khoảng cách thật < 0,5 s."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps phải là số dương.")
    if not math.isfinite(buoc_giay) or buoc_giay <= 0:
        raise ValueError("--buoc-giay phải là số dương.")
    if buoc_giay >= NGUONG_DAY_GIAY:
        raise ValueError(
            f"--buoc-giay phải nhỏ hơn {NGUONG_DAY_GIAY}, nhận {buoc_giay}."
        )

    # floor giúp khoảng cách thực không vượt quá giá trị người chạy yêu cầu.
    buoc = max(1, int(math.floor(fps * buoc_giay + 1e-9)))
    if buoc / fps >= NGUONG_DAY_GIAY:
        buoc = max(1, int(math.ceil(fps * NGUONG_DAY_GIAY) - 1))
    if buoc / fps >= NGUONG_DAY_GIAY:
        raise ValueError(
            f"Video chỉ có {fps:g} fps nên không thể lấy khung cách nhau "
            f"dưới {NGUONG_DAY_GIAY} giây."
        )
    return buoc


def tao_chi_so_khung(
    khung_dau: int,
    khung_cuoi: int,
    buoc_khung: int,
    moc_bat_buoc: Iterable[int] = (),
) -> list[int]:
    """Danh sách frame_idx tăng dần; luôn có hai đầu và các mốc kiểm chứng."""
    if khung_dau < 0:
        raise ValueError("Khung đầu không được âm.")
    if khung_cuoi < khung_dau:
        raise ValueError("Khung cuối phải lớn hơn hoặc bằng khung đầu.")
    if buoc_khung < 1:
        raise ValueError("Bước khung phải >= 1.")

    chi_so = set(range(khung_dau, khung_cuoi + 1, buoc_khung))
    chi_so.add(khung_cuoi)
    chi_so.update(int(x) for x in moc_bat_buoc if khung_dau <= int(x) <= khung_cuoi)
    return sorted(chi_so)


def _moc_keyframe(video_id: str, khung_dau: int, khung_cuoi: int) -> list[int]:
    bang = load_frame_map()
    nhom = bang[
        (bang["video_id"] == video_id)
        & (bang["frame_idx"] >= khung_dau)
        & (bang["frame_idx"] <= khung_cuoi)
    ]
    return [int(x) for x in nhom["frame_idx"].tolist()]


def _co_chuong_trinh(ten: str, goi_y: str) -> str:
    duong_dan = shutil.which(ten)
    if not duong_dan:
        raise FileNotFoundError(
            f"Không thấy {ten} trong PATH. Cài bằng: winget install Gyan.FFmpeg\n"
            f"Rồi mở lại PowerShell. {goi_y}"
        )
    return duong_dan


_CO_FPS_MODE: bool | None = None


def _co_khoa_fps_mode(ffmpeg: str) -> bool:
    """Dò ``-fps_mode`` để tương thích cả ffmpeg cũ lẫn mới."""
    global _CO_FPS_MODE
    if _CO_FPS_MODE is not None:
        return _CO_FPS_MODE
    thu = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "quiet",
            "-f",
            "lavfi",
            "-i",
            "nullsrc=s=16x16:d=0.1",
            "-fps_mode",
            "passthrough",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
    )
    _CO_FPS_MODE = thu.returncode == 0
    return _CO_FPS_MODE


def _phan_so(s: str | None) -> float | None:
    if not s or s in {"N/A", "0/0"}:
        return None
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def thong_tin_video(duong_dan: Path) -> dict:
    """Đọc fps và số khung bằng ffprobe, không giải mã toàn bộ video."""
    ffprobe = _co_chuong_trinh("ffprobe", "ffprobe thường đi kèm ffmpeg.")
    lenh = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,r_frame_rate,duration",
        "-of",
        "json",
        str(duong_dan),
    ]
    kq = subprocess.run(lenh, capture_output=True, text=True)
    if kq.returncode != 0:
        raise RuntimeError(f"ffprobe lỗi với {duong_dan.name}:\n{kq.stderr[:800]}")
    try:
        stream = json.loads(kq.stdout)["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise RuntimeError(f"ffprobe không đọc được luồng hình của {duong_dan}.") from exc

    fps = _phan_so(stream.get("avg_frame_rate")) or _phan_so(stream.get("r_frame_rate"))
    try:
        so_khung = int(stream["nb_frames"])
    except (KeyError, TypeError, ValueError):
        so_khung = 0
    if so_khung <= 0:
        try:
            thoi_luong = float(stream["duration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Không suy ra được số khung của {duong_dan.name}; "
                "hãy dùng --khung DAU CUOI."
            ) from exc
        if not fps:
            raise RuntimeError(f"Không đọc được fps của {duong_dan.name}.")
        so_khung = max(1, int(math.floor(thoi_luong * fps + 1e-6)))

    return {"fps": fps, "so_khung": so_khung}


def _bieu_thuc_select(
    khung_dau: int,
    khung_cuoi: int,
    buoc_khung: int,
    moc_bat_buoc: Iterable[int],
) -> str:
    # Không dùng -ss: bộ đếm n phải bắt đầu ở đúng khung 0 của video.
    noi = [f"not(mod(n-{khung_dau}\\,{buoc_khung}))", f"eq(n\\,{khung_cuoi})"]
    noi.extend(f"eq(n\\,{int(x)})" for x in moc_bat_buoc)
    bieu_thuc = (
        f"between(n\\,{khung_dau}\\,{khung_cuoi})*"
        f"({'+'.join(noi)})"
    )
    return f"select='{bieu_thuc}'"


def _doc_manifest(thu_muc: Path) -> dict:
    tep = thu_muc / TEN_MANIFEST
    if not tep.exists():
        return {"schema_version": 1, "ranges": []}
    try:
        du_lieu = json.loads(tep.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "ranges": []}
    if not isinstance(du_lieu, dict):
        return {"schema_version": 1, "ranges": []}
    du_lieu.setdefault("ranges", [])
    return du_lieu


def _ghi_manifest(
    thu_muc: Path,
    video_id: str,
    fps: float,
    khung_dau: int,
    khung_cuoi: int,
    buoc_khung: int,
    so_khung_vung: int,
) -> None:
    manifest = _doc_manifest(thu_muc)
    vung_moi = {
        "frame_start": khung_dau,
        "frame_end": khung_cuoi,
        "step_frames": buoc_khung,
        "max_regular_gap_seconds": buoc_khung / fps,
        "frame_count": so_khung_vung,
    }
    vung_cu = [
        r
        for r in manifest.get("ranges", [])
        if not (
            int(r.get("frame_start", -1)) == khung_dau
            and int(r.get("frame_end", -1)) == khung_cuoi
        )
    ]
    manifest.update(
        {
            "schema_version": 1,
            "video_id": video_id,
            "fps": fps,
            "threshold_seconds": NGUONG_DAY_GIAY,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ranges": vung_cu + [vung_moi],
            "frame_count_total": len(list(thu_muc.glob("*.jpg"))),
        }
    )
    tam = thu_muc / f"{TEN_MANIFEST}.partial"
    tam.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tam.replace(thu_muc / TEN_MANIFEST)


def trich(
    video_id: str,
    khung_dau: int,
    khung_cuoi: int,
    chat_luong: int = 2,
    ghi_de: bool = False,
    buoc_giay: float = BUOC_GIAY_MAC_DINH,
) -> dict:
    """Trích khung theo bước dày trong đoạn đóng ``[khung_dau, khung_cuoi]``."""
    ffmpeg = _co_chuong_trinh("ffmpeg", "")
    nguon = video_file(video_id)
    if not nguon.exists():
        raise FileNotFoundError(f"Không thấy video gốc {nguon}.")

    fps = _fps(video_id)
    buoc_khung = tinh_buoc_khung(fps, buoc_giay)
    moc = _moc_keyframe(video_id, khung_dau, khung_cuoi)
    can = tao_chi_so_khung(khung_dau, khung_cuoi, buoc_khung, moc)

    dich = FRAMES_DENSE_DIR / video_id
    dich.mkdir(parents=True, exist_ok=True)
    da_co = {int(p.stem) for p in dich.glob("*.jpg") if p.stem.isdigit()}
    if not ghi_de and set(can) <= da_co:
        _ghi_manifest(dich, video_id, fps, khung_dau, khung_cuoi, buoc_khung, len(can))
        return {
            "video_id": video_id,
            "bo_qua": True,
            "so_khung": len(can),
            "buoc_khung": buoc_khung,
            "khoang_cach_giay": buoc_khung / fps,
            "ly_do": "đã có đủ khung trong khoảng này",
        }

    FRAMES_DENSE_DIR.mkdir(parents=True, exist_ok=True)
    thu_muc_tam = Path(tempfile.mkdtemp(prefix=f".{video_id}_", dir=FRAMES_DENSE_DIR))
    try:
        lenh = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(nguon),
            "-vf",
            _bieu_thuc_select(khung_dau, khung_cuoi, buoc_khung, moc),
            *(
                ["-fps_mode", "passthrough"]
                if _co_khoa_fps_mode(ffmpeg)
                else ["-vsync", "0"]
            ),
            "-start_number",
            "0",
            "-q:v",
            str(chat_luong),
            str(thu_muc_tam / "%09d.jpg"),
        ]
        ket_qua = subprocess.run(lenh, capture_output=True, text=True)
        if ket_qua.returncode != 0:
            raise RuntimeError(
                f"ffmpeg lỗi:\n{ket_qua.stderr[:1000]}\n\n"
                "Lưu ý: script cố ý không dùng -ss để giữ đúng frame_idx."
            )

        tep_tam = sorted(thu_muc_tam.glob("*.jpg"))
        if len(tep_tam) != len(can):
            raise RuntimeError(
                f"ffmpeg sinh {len(tep_tam)} ảnh nhưng lịch lấy mẫu cần {len(can)}. "
                f"Đoạn yêu cầu {khung_dau}-{khung_cuoi}; có thể khung cuối vượt "
                "quá độ dài video hoặc video bị lỗi giải mã. Không chép ảnh dở vào đầu ra."
            )

        for tep, frame_idx in zip(tep_tam, can):
            muc_tieu = dich / f"{frame_idx:06d}.jpg"
            if muc_tieu.exists() and not ghi_de:
                continue
            tam = dich / f".{frame_idx:06d}.jpg.partial"
            shutil.move(str(tep), str(tam))
            tam.replace(muc_tieu)
    finally:
        shutil.rmtree(thu_muc_tam, ignore_errors=True)

    co_that = {int(p.stem) for p in dich.glob("*.jpg") if p.stem.isdigit()}
    thieu = sorted(set(can) - co_that)
    if thieu:
        raise RuntimeError(f"Thiếu {len(thieu)} ảnh sau khi chép: {thieu[:5]}")

    _ghi_manifest(dich, video_id, fps, khung_dau, khung_cuoi, buoc_khung, len(can))
    return {
        "video_id": video_id,
        "bo_qua": False,
        "khung_dau": khung_dau,
        "khung_cuoi": khung_cuoi,
        "so_khung_can": len(can),
        "so_khung_co": len(set(can) & co_that),
        "so_khung_thieu": 0,
        "buoc_khung": buoc_khung,
        "khoang_cach_giay": buoc_khung / fps,
        "thu_muc": str(dich),
    }


def _vector_anh(duong_dan: Path):
    import numpy as np
    from PIL import Image

    with Image.open(duong_dan) as anh:
        return np.asarray(anh.convert("L").resize((160, 90)), dtype=np.float64)


def kiem_danh_so(video_id: str, so_mau: int = 7) -> dict:
    """Đối chiếu ảnh dense tại frame_idx với keyframe BTC cùng frame_idx."""
    import numpy as np

    thu_muc = FRAMES_DENSE_DIR / video_id
    da_trich = {int(p.stem): p for p in thu_muc.glob("*.jpg") if p.stem.isdigit()}
    if not da_trich:
        return {"kiem_duoc": False, "ly_do": f"Chưa trích khung cho {video_id}."}

    bang = load_frame_map()
    nhom = bang[bang["video_id"] == video_id].sort_values("frame_idx")
    chung = [
        (int(r.n), int(r.frame_idx))
        for r in nhom.itertuples()
        if int(r.frame_idx) in da_trich and keyframe_image(video_id, int(r.n)).exists()
    ]
    if not chung:
        return {
            "kiem_duoc": False,
            "ly_do": "Không có keyframe BTC cùng frame_idx để đối chiếu ảnh.",
        }

    mau = chung[:: max(1, len(chung) // so_mau)][:so_mau]
    chi_so_dense = sorted(da_trich)
    vi_tri = {x: i for i, x in enumerate(chi_so_dense)}
    chi_tiet = []
    for n, frame_idx in mau:
        i = vi_tri[frame_idx]
        lan_can = chi_so_dense[max(0, i - 2) : i + 3]
        goc = _vector_anh(keyframe_image(video_id, n))
        sai_so = {
            fi: float(np.abs(goc - _vector_anh(da_trich[fi])).mean()) for fi in lan_can
        }
        tot_nhat = min(sai_so, key=sai_so.get)
        chi_tiet.append(
            {
                "n": n,
                "frame_idx": frame_idx,
                "frame_giong_nhat": tot_nhat,
                "do_dich": tot_nhat - frame_idx,
                "lech_tai_0": sai_so[frame_idx],
                "lech_tot_nhat": sai_so[tot_nhat],
                "khop": tot_nhat == frame_idx,
            }
        )

    do_dich = [c["do_dich"] for c in chi_tiet]
    dich_chung = do_dich[0] if do_dich and len(set(do_dich)) == 1 else None
    return {
        "kiem_duoc": True,
        "so_mau": len(chi_tiet),
        "so_khop": sum(1 for c in chi_tiet if c["khop"]),
        "dat": all(c["khop"] for c in chi_tiet),
        "do_dich_he_thong": dich_chung if dich_chung not in (None, 0) else None,
        "chi_tiet": chi_tiet,
    }


def _cac_vung_manifest(video_id: str) -> list[tuple[int, int]]:
    manifest = _doc_manifest(FRAMES_DENSE_DIR / video_id)
    ra = []
    for vung in manifest.get("ranges", []):
        try:
            ra.append((int(vung["frame_start"]), int(vung["frame_end"])))
        except (KeyError, TypeError, ValueError):
            continue
    return ra


def khoang_cach_thuc_te(video_id: str) -> float:
    """Khoảng cách LỚN NHẤT trong các vùng đã trích, tính bằng giây."""
    thu_muc = FRAMES_DENSE_DIR / video_id
    so = sorted(int(p.stem) for p in thu_muc.glob("*.jpg") if p.stem.isdigit())
    if len(so) < 2:
        return float("inf")
    vung = _cac_vung_manifest(video_id) or [(so[0], so[-1])]
    lon_nhat = 0
    for dau, cuoi in vung:
        trong = [x for x in so if dau <= x <= cuoi]
        if len(trong) < 2:
            return float("inf")
        lon_nhat = max(lon_nhat, max(b - a for a, b in zip(trong, trong[1:])))
    return lon_nhat / _fps(video_id)


def khoang_tu_dev(duong_dan: Path, dem: int = DEM_KHUNG_MAC_DINH) -> list[tuple]:
    """Các câu TRAKE trong bộ dev -> (video_id, đầu, cuối, query_id)."""
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


def _video_id_tu_muc(muc) -> list[str]:
    if isinstance(muc, str):
        return [muc.strip()] if muc.strip() else []
    if isinstance(muc, dict):
        if "video_id" in muc:
            return [str(muc["video_id"]).strip()]
        if isinstance(muc.get("videos"), list):
            return [str(x).strip() for x in muc["videos"]]
    return []


def doc_danh_sach_video(duong_dan: Path) -> list[str]:
    """Đọc txt/csv/json/jsonl; loại trùng nhưng giữ thứ tự xuất hiện."""
    if not duong_dan.exists():
        raise FileNotFoundError(f"Không thấy danh sách video: {duong_dan}")
    ra: list[str] = []
    duoi = duong_dan.suffix.lower()
    if duoi == ".csv":
        with duong_dan.open("r", encoding="utf-8-sig", newline="") as f:
            for muc in csv.DictReader(f):
                ra.extend(_video_id_tu_muc(muc))
    elif duoi == ".json":
        du_lieu = json.loads(duong_dan.read_text(encoding="utf-8"))
        if isinstance(du_lieu, list):
            for muc in du_lieu:
                ra.extend(_video_id_tu_muc(muc))
        else:
            ra.extend(_video_id_tu_muc(du_lieu))
    elif duoi == ".jsonl":
        for so_dong, dong in enumerate(duong_dan.read_text(encoding="utf-8").splitlines(), 1):
            if not dong.strip():
                continue
            try:
                ra.extend(_video_id_tu_muc(json.loads(dong)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{duong_dan.name} dòng {so_dong} sai JSON.") from exc
    else:
        for dong in duong_dan.read_text(encoding="utf-8-sig").splitlines():
            dong = dong.split("#", 1)[0].strip()
            if dong:
                ra.extend(x for x in re.split(r"[\s,;]+", dong) if x)

    ket_qua = []
    da_gap = set()
    for video_id in ra:
        if video_id and video_id not in da_gap:
            da_gap.add(video_id)
            ket_qua.append(video_id)
    if not ket_qua:
        raise ValueError(f"{duong_dan} không chứa video_id nào.")
    return ket_qua


def _kiem_fps_video(video_id: str, fps_map: float, fps_video: float | None) -> None:
    if fps_video is None:
        return
    sai_so = abs(fps_map - fps_video)
    if sai_so > max(0.02, fps_map * 0.001):
        raise RuntimeError(
            f"{video_id}: fps video={fps_video:.6f} khác fps map={fps_map:.6f}. "
            "Dừng để tránh đặt sai tên frame_idx."
        )


def _in_kiem(video_id: str) -> bool:
    kq = kiem_danh_so(video_id)
    print(f"\n{video_id}")
    if not kq["kiem_duoc"]:
        print(f"  KHÔNG KIỂM ẢNH ĐƯỢC: {kq['ly_do']}")
        return False
    for c in kq["chi_tiet"]:
        dau = "OK  " if c["khop"] else "LỆCH"
        print(
            f"  {dau} n={c['n']:<5} frame_idx={c['frame_idx']:<8} "
            f"ảnh giống nhất mang số {c['frame_giong_nhat']} "
            f"(dịch {c['do_dich']:+d})"
        )
    print(
        f"  {kq['so_khop']}/{kq['so_mau']} mẫu đúng | "
        f"khoảng cách lớn nhất {khoang_cach_thuc_te(video_id):.3f}s"
    )
    return bool(kq["dat"])


def main() -> int:
    p = argparse.ArgumentParser(description="Việc 9 — trích frames_dense cho TRAKE")
    p.add_argument("--video", action="append", default=[], help="có thể lặp nhiều lần")
    p.add_argument("--danh-sach", type=Path, help="txt/csv/json/jsonl chứa video_id")
    p.add_argument("--top-video", type=int, default=None, help="chỉ lấy N video đầu")
    p.add_argument("--khung", nargs=2, type=int, metavar=("DAU", "CUOI"))
    p.add_argument("--giay", nargs=2, type=float, metavar=("DAU", "CUOI"))
    p.add_argument("--tu-dev", action="store_true")
    p.add_argument("--tep-dev", default=str(DEV_QUERIES_PATH))
    p.add_argument("--dem", type=int, default=DEM_KHUNG_MAC_DINH)
    p.add_argument("--buoc-giay", type=float, default=BUOC_GIAY_MAC_DINH)
    p.add_argument("--kiem", action="append", default=[], metavar="VIDEO")
    p.add_argument("--ghi-de", action="store_true")
    args = p.parse_args()

    if args.kiem:
        ket_qua_kiem = [_in_kiem(v) for v in args.kiem]
        return 0 if all(ket_qua_kiem) else 1

    video_ids = list(args.video)
    if args.danh_sach:
        video_ids.extend(doc_danh_sach_video(args.danh_sach))
    video_ids = list(dict.fromkeys(video_ids))
    if args.top_video is not None:
        if args.top_video < 1:
            p.error("--top-video phải >= 1")
        video_ids = video_ids[: args.top_video]

    cong_viec: list[tuple[str, int, int, str]] = []
    if args.tu_dev:
        for vid, dau, cuoi, qid in khoang_tu_dev(Path(args.tep_dev), args.dem):
            if video_file(vid).exists():
                cong_viec.append((vid, dau, cuoi, f"câu {qid}"))
            else:
                print(f"BỎ QUA câu {qid}: không có video {vid} trên máy.")

    if (args.khung or args.giay) and len(video_ids) != 1:
        p.error("--khung/--giay chỉ dùng khi chọn đúng một --video.")

    for vid in video_ids:
        nguon = video_file(vid)
        if not nguon.exists():
            print(f"BỎ QUA {vid}: không thấy {nguon}")
            continue
        info = thong_tin_video(nguon)
        fps = _fps(vid)
        _kiem_fps_video(vid, fps, info.get("fps"))
        khung_cuoi_video = int(info["so_khung"]) - 1
        if args.khung:
            dau, cuoi = args.khung
        elif args.giay:
            dau = int(math.floor(args.giay[0] * fps))
            cuoi = int(math.floor(args.giay[1] * fps))
        else:
            dau, cuoi = 0, khung_cuoi_video
        if cuoi > khung_cuoi_video:
            raise ValueError(
                f"{vid}: khung cuối {cuoi} vượt video (khung cuối {khung_cuoi_video})."
            )
        cong_viec.append((vid, dau, cuoi, "danh sách nghi ngờ"))

    if not cong_viec:
        print("Không có việc nào. Dùng --video, --danh-sach hoặc --tu-dev.")
        return 1

    loi = 0
    for vid, dau, cuoi, nguon_chon in cong_viec:
        print(f"\n{vid}: {dau}-{cuoi} ({nguon_chon})")
        try:
            kq = trich(
                vid,
                dau,
                cuoi,
                ghi_de=args.ghi_de,
                buoc_giay=args.buoc_giay,
            )
        except Exception as exc:
            loi += 1
            print(f"  LỖI {type(exc).__name__}: {exc}")
            continue
        if kq["bo_qua"]:
            print(f"  Bỏ qua: {kq['ly_do']} ({kq['so_khung']:,} ảnh)")
        else:
            print(
                f"  Đã có {kq['so_khung_co']:,} ảnh; bước đều "
                f"{kq['buoc_khung']} khung = {kq['khoang_cach_giay']:.3f}s"
            )
            print(f"  {kq['thu_muc']}")

    print("\nNghiệm thu bắt buộc:")
    lenh = "python -u -m scripts.verify_task9_acceptance"
    for vid in dict.fromkeys(v for v, _, _, _ in cong_viec):
        lenh += f" --video {vid}"
    print(f"  {lenh}")
    return 1 if loi else 0


if __name__ == "__main__":
    raise SystemExit(main())
