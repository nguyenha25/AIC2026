"""N-00 — đóng băng baseline sạch, split theo video và cấu hình tham chiếu.

Lệnh chính (chạy baseline Q&A local và đóng Gate 0):

    python -u -m scripts.dong_bang_n00

Lệnh chỉ chuẩn bị để xem trước split, không được tính là hoàn tất N-00:

    python -u -m scripts.dong_bang_n00 --chi-chuan-bi --run-id n00_preview

Output nằm trong ``DATA_ROOT/runs/<run_id>/``. Script không ghi đè một run đã
tồn tại; muốn thử cấu hình khác phải dùng run_id mới để lịch sử không bị sửa.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.frame_map import FrameMap  # noqa: E402
from aic2026.n00_freeze import (  # noqa: E402
    LoiDongBang,
    chia_theo_video,
    dem_theo_loai,
    doc_jsonl,
    ghi_json_atomic,
    ghi_jsonl_atomic,
    kiem_schema,
    sha256_file,
    tach_tap_sach,
)
from aic2026.paths import (  # noqa: E402
    CONFIG_DIR,
    DATA_ROOT,
    DEV_QUERIES_PATH,
    FRAMES_DENSE_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    keyframe_image,
    resolve,
)


N00_CONFIG = CONFIG_DIR / "n00_baseline.yaml"
SETTINGS_CONFIG = CONFIG_DIR / "settings.yaml"
RRF_CONFIG = CONFIG_DIR / "rrf_weights.yaml"
CAC_BIEN_API = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)


def _doc_yaml(path: Path) -> dict:
    if not path.is_file():
        raise LoiDongBang(f"Thiếu cấu hình: {path}")
    du_lieu = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(du_lieu, dict):
        raise LoiDongBang(f"{path.name} phải chứa một object YAML")
    return du_lieu


def _kiem_offline(n00: dict, settings: dict, rrf: dict) -> dict:
    offline = n00.get("offline", {})
    if offline.get("bat_buoc") is not True:
        raise LoiDongBang("n00_baseline.yaml phải đặt offline.bat_buoc=true")

    dang_co = [ten for ten in CAC_BIEN_API if os.getenv(ten)]
    if dang_co:
        raise LoiDongBang(
            "Đang có biến API key trong môi trường: " + ", ".join(dang_co)
            + ". Xóa các biến này rồi chạy lại N-00."
        )

    cho_phep = set(offline.get("nguon_mo_rong_cho_phep", []))
    if not cho_phep or "llm" in cho_phep:
        raise LoiDongBang(
            "offline.nguon_mo_rong_cho_phep phải chỉ gồm nguồn local, không có llm"
        )
    nguon_settings = (
        settings.get("mo_rong_truy_van", {}).get("nguon")
    )
    if nguon_settings not in cho_phep:
        raise LoiDongBang(
            f"settings.yaml đang dùng mo_rong_truy_van.nguon={nguon_settings!r}; "
            f"N-00 chỉ cho phép {sorted(cho_phep)}"
        )

    so_ung_vien = settings.get("tra_cuu", {}).get("so_ung_vien_moi_nguon")
    mong_doi_k = int(n00.get("qa_baseline", {}).get("so_ung_vien", 0))
    if int(so_ung_vien or 0) != mong_doi_k:
        raise LoiDongBang(
            f"K_source lệch: settings={so_ung_vien}, n00={mong_doi_k}"
        )

    thuc_te = rrf.get("theo_dang", {}).get("qa") or {}
    mong_doi = n00.get("qa_baseline", {}).get("rrf_weights", {})
    sai = {
        ten: {"expected": float(gia_tri), "actual": float(thuc_te.get(ten, 0.0))}
        for ten, gia_tri in mong_doi.items()
        if float(thuc_te.get(ten, 0.0)) != float(gia_tri)
    }
    if sai:
        raise LoiDongBang(f"Trọng số QA khác cấu hình N-00: {sai}")
    return {
        "offline_only": True,
        "query_expansion_source": nguon_settings,
        "api_keys_present": [],
        "qa_rrf_weights": {k: float(thuc_te.get(k, 0.0)) for k in mong_doi},
        "k_source": mong_doi_k,
    }


def _chi_so_dense(video_id: str) -> tuple[set[int], Path]:
    thu_muc = resolve(FRAMES_DENSE_DIR, video_id) or FRAMES_DENSE_DIR / video_id
    if not thu_muc.is_dir():
        return set(), thu_muc
    chi_so = {
        int(p.stem)
        for p in thu_muc.glob("*.jpg")
        if p.stem.isdigit()
    }
    return chi_so, thu_muc


def tao_ham_kiem_bang_chung():
    """Tạo checker có cache để không đọc lại frame_map cho từng câu."""
    map_cache: dict[str, FrameMap | Exception] = {}
    dense_cache: dict[str, tuple[set[int], Path]] = {}

    def kiem(q: dict) -> tuple[bool, str]:
        video_id = str(q["video_id"])
        loai = str(q["loai_truy_van"])

        if loai in {"mo_ta", "hoi_dap"}:
            if video_id not in map_cache:
                try:
                    map_cache[video_id] = FrameMap.load(video_id)
                except (FileNotFoundError, ValueError) as exc:
                    map_cache[video_id] = exc
            bang = map_cache[video_id]
            if isinstance(bang, Exception):
                return False, "thieu_hoac_hong_frame_map"
            dau, cuoi = int(q["frame_start"]), int(q["frame_end"])
            trong_cua_so = [r for r in bang.rows if dau <= r.frame_idx <= cuoi]
            if not trong_cua_so:
                return False, "khong_co_keyframe_trong_cua_so_gt"
            if not any(keyframe_image(video_id, r.n).is_file() for r in trong_cua_so):
                return False, "thieu_anh_gt_trong_cua_so"
            return True, "ok"

        if video_id not in dense_cache:
            dense_cache[video_id] = _chi_so_dense(video_id)
        chi_so, thu_muc = dense_cache[video_id]
        if not chi_so:
            return False, "thieu_frames_dense"
        if not (thu_muc / "manifest.json").is_file():
            return False, "thieu_manifest_dense"
        for i, giai_doan in enumerate(q["cac_giai_doan"], 1):
            dau = int(giai_doan["frame_start"])
            cuoi = int(giai_doan["frame_end"])
            if not any(dau <= frame_idx <= cuoi for frame_idx in chi_so):
                return False, f"dense_khong_phu_event_{i}"
        return True, "ok"

    return kiem


def _nhan_duong_dan(path: Path) -> str:
    path = path.resolve()
    for ten, goc in (("DATA_ROOT", DATA_ROOT), ("PROJECT_ROOT", PROJECT_ROOT)):
        try:
            return f"{ten}/{path.relative_to(goc.resolve()).as_posix()}"
        except ValueError:
            pass
    return path.name


def _git_commit(fallback: str | None) -> str:
    try:
        kq = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if kq.returncode == 0 and kq.stdout.strip():
            return kq.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return fallback or "unknown"


def _copy_snapshot(nguon: Path, dich: Path) -> dict:
    dich.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nguon, dich)
    return {
        "source": _nhan_duong_dan(nguon),
        "snapshot": dich.relative_to(dich.parents[1]).as_posix(),
        "sha256": sha256_file(dich),
        "size_bytes": dich.stat().st_size,
    }


def _doc_bao_cao_qa(path: Path, so_cau_qa_holdout: int) -> dict:
    if not path.is_file():
        raise LoiDongBang(f"Không thấy báo cáo QA: {path}")
    try:
        bao_cao = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise LoiDongBang(f"Báo cáo QA hỏng JSON: {path}") from exc
    thieu = [k for k in ("so_cau", "tran", "that", "so_khop") if k not in bao_cao]
    if thieu:
        raise LoiDongBang(f"Báo cáo QA thiếu khóa: {thieu}")
    if int(bao_cao["so_cau"]) != so_cau_qa_holdout:
        raise LoiDongBang(
            f"Báo cáo có {bao_cao['so_cau']} câu, dev_holdout có "
            f"{so_cau_qa_holdout} câu QA; có thể đang dùng báo cáo cũ"
        )
    return bao_cao


def _chay_qa(dev_holdout: Path, n00: dict) -> Path:
    qa = n00["qa_baseline"]
    bao_cao = RUNS_DIR / "do_qa.json"
    mtime_cu = bao_cao.stat().st_mtime_ns if bao_cao.exists() else None
    lenh = [
        sys.executable,
        "-u",
        "-m",
        "scripts.do_qa",
        "--tep",
        str(dev_holdout),
        "--so-ung-vien",
        str(int(qa["so_ung_vien"])),
        "--so-doc",
        str(int(qa["so_doc"])),
        "--nguon-mo-rong",
        str(qa["nguon_mo_rong"]),
        "--nguon-clip-l",
        str(qa["nguon_clip_l"]),
    ]
    moi_truong = os.environ.copy()
    for ten in CAC_BIEN_API:
        moi_truong.pop(ten, None)
    print("\nChạy baseline QA local trên dev_holdout:\n  " + " ".join(lenh) + "\n")
    kq = subprocess.run(lenh, cwd=PROJECT_ROOT, env=moi_truong)
    if kq.returncode != 0:
        raise LoiDongBang(f"Baseline QA thất bại với mã thoát {kq.returncode}")
    if not bao_cao.is_file():
        raise LoiDongBang("do_qa chạy xong nhưng không tạo DATA_ROOT/runs/do_qa.json")
    if mtime_cu is not None and bao_cao.stat().st_mtime_ns <= mtime_cu:
        raise LoiDongBang("do_qa.json không được cập nhật; không dùng báo cáo cũ")
    return bao_cao


def main() -> int:
    parser = argparse.ArgumentParser(description="N-00 — đóng băng Gate 0")
    parser.add_argument("--tep", default=str(DEV_QUERIES_PATH))
    parser.add_argument("--cau-hinh", default=str(N00_CONFIG))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--ra", default=None, help="thư mục run; mặc định DATA_ROOT/runs/<run_id>")
    parser.add_argument(
        "--chi-chuan-bi",
        action="store_true",
        help="không chạy baseline QA; output chỉ để xem trước, Gate 0 chưa đạt",
    )
    parser.add_argument(
        "--bao-cao-qa",
        default=None,
        help="dùng báo cáo do_qa.json đã chạy đúng dev_holdout thay vì chạy lại",
    )
    args = parser.parse_args()

    try:
        n00_path = Path(args.cau_hinh)
        n00 = _doc_yaml(n00_path)
        settings = _doc_yaml(SETTINGS_CONFIG)
        rrf = _doc_yaml(RRF_CONFIG)
        offline = _kiem_offline(n00, settings, rrf)

        run_id = args.run_id or str(n00.get("run_id", "n00_baseline_20260829"))
        if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
            raise LoiDongBang("run_id chỉ được chứa chữ, số, gạch dưới và gạch ngang")
        run_dir = Path(args.ra) if args.ra else RUNS_DIR / run_id
        if run_dir.exists():
            raise LoiDongBang(
                f"Run đã tồn tại: {run_dir}. N-00 không ghi đè; dùng --run-id mới."
            )

        dev_path = Path(args.tep)
        tat_ca = kiem_schema(doc_jsonl(dev_path))
        sach, bi_loai = tach_tap_sach(tat_ca, tao_ham_kiem_bang_chung())
        if not sach:
            raise LoiDongBang(
                "Không câu nào có evidence GT đầy đủ. Kiểm tra keyframes/frame_map/frames_dense."
            )

        split = n00.get("split", {})
        tune, holdout = chia_theo_video(
            sach,
            ti_le_holdout=float(split.get("holdout_ratio", 0.30)),
            seed=str(split.get("seed", "aic2026-n00-v1")),
        )
        video_tune = {str(q["video_id"]) for q in tune}
        video_holdout = {str(q["video_id"]) for q in holdout}
        if video_tune & video_holdout:
            raise LoiDongBang("Split có leakage video")
        so_qa_holdout = sum(q["loai_truy_van"] == "hoi_dap" for q in holdout)
        if not args.chi_chuan_bi and so_qa_holdout == 0:
            raise LoiDongBang(
                "dev_holdout không có câu QA; không thể đóng baseline Q&A. "
                "Cần bổ sung evidence sạch ở ít nhất hai video QA."
            )

        run_dir.mkdir(parents=True)
        dev_clean = run_dir / "dev_clean.jsonl"
        dev_tune = run_dir / "dev_tune.jsonl"
        dev_holdout = run_dir / "dev_holdout.jsonl"
        dev_excluded = run_dir / "dev_excluded.jsonl"
        ghi_jsonl_atomic(dev_clean, sach)
        ghi_jsonl_atomic(dev_tune, tune)
        ghi_jsonl_atomic(dev_holdout, holdout)
        ghi_jsonl_atomic(dev_excluded, bi_loai)

        snapshots = {
            "n00_baseline": _copy_snapshot(n00_path, run_dir / "config" / n00_path.name),
            "settings": _copy_snapshot(SETTINGS_CONFIG, run_dir / "config" / SETTINGS_CONFIG.name),
            "rrf_weights": _copy_snapshot(RRF_CONFIG, run_dir / "config" / RRF_CONFIG.name),
        }

        baseline = {"status": "pending"}
        if not args.chi_chuan_bi:
            bao_cao_path = (
                Path(args.bao_cao_qa)
                if args.bao_cao_qa
                else _chay_qa(dev_holdout, n00)
            )
            bao_cao = _doc_bao_cao_qa(bao_cao_path, so_qa_holdout)
            dich_bao_cao = run_dir / "baseline_qa.json"
            shutil.copy2(bao_cao_path, dich_bao_cao)
            baseline = {
                "status": "completed",
                "report": dich_bao_cao.name,
                "sha256": sha256_file(dich_bao_cao),
                "so_cau": int(bao_cao["so_cau"]),
                "retrieval_ceiling": float(bao_cao["tran"]),
                "end_to_end": float(bao_cao["that"]),
                "answer_matches": int(bao_cao["so_khop"]),
            }

        ly_do_loai = Counter(str(x["reason"]) for x in bi_loai)
        manifest = {
            "schema_version": "1.1",
            "run_id": run_id,
            "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
            "status": "completed" if baseline["status"] == "completed" else "prepared",
            "deadline": str(n00.get("deadline")),
            "source_commit": _git_commit(n00.get("source_snapshot")),
            "dataset": {
                "source": _nhan_duong_dan(dev_path),
                "sha256": sha256_file(dev_path),
                "total_queries": len(tat_ca),
                "clean_queries": len(sach),
                "source_evidence_coverage": len(sach) / len(tat_ca),
                "clean_evidence_coverage": 1.0,
                "excluded_queries": len(bi_loai),
                "excluded_by_reason": dict(sorted(ly_do_loai.items())),
            },
            "split": {
                "seed": str(split.get("seed")),
                "holdout_ratio_target": float(split.get("holdout_ratio")),
                "group_key": "video_id",
                "no_video_leakage": not bool(video_tune & video_holdout),
                "tune": {
                    "queries": len(tune),
                    "videos": len(video_tune),
                    "by_task": dem_theo_loai(tune),
                    "file": dev_tune.name,
                    "sha256": sha256_file(dev_tune),
                },
                "holdout": {
                    "queries": len(holdout),
                    "videos": len(video_holdout),
                    "by_task": dem_theo_loai(holdout),
                    "file": dev_holdout.name,
                    "sha256": sha256_file(dev_holdout),
                },
            },
            "offline": offline,
            "config_snapshots": snapshots,
            "baseline_qa": baseline,
            "gates": {
                "schema_valid": True,
                "clean_evidence_coverage_100pct": True,
                "no_video_leakage": not bool(video_tune & video_holdout),
                "offline_config_valid": True,
                "baseline_qa_completed": baseline["status"] == "completed",
            },
        }
        ghi_json_atomic(run_dir / "manifest.json", manifest)

        print("\n" + "=" * 78)
        print("N-00 — KẾT QUẢ ĐÓNG BĂNG")
        print("=" * 78)
        print(f"Run          : {run_dir}")
        print(f"Dev gốc      : {len(tat_ca)}")
        print(f"Dev sạch     : {len(sach)} ({len(sach) / len(tat_ca):.1%})")
        print(f"dev_tune     : {len(tune)} câu / {len(video_tune)} video / {dem_theo_loai(tune)}")
        print(f"dev_holdout  : {len(holdout)} câu / {len(video_holdout)} video / {dem_theo_loai(holdout)}")
        print(f"Bị loại      : {len(bi_loai)} / {dict(sorted(ly_do_loai.items()))}")
        print("Video leakage: 0")
        print("API key      : không dùng")
        if baseline["status"] == "completed":
            print(
                "Baseline QA   : "
                f"trần={baseline['retrieval_ceiling']:.4f}, "
                f"thật={baseline['end_to_end']:.4f}, "
                f"khớp={baseline['answer_matches']}/{baseline['so_cau']}"
            )
            print("=> N-00 ĐẠT. Có thể dùng manifest này làm mốc so sánh Gate 1-5.")
        else:
            print("Baseline QA   : CHƯA CHẠY (--chi-chuan-bi)")
            print("=> N-00 CHƯA ĐÓNG. Đây chỉ là preview split.")
        return 0
    except (LoiDongBang, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"\nN-00 KHÔNG ĐẠT: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
