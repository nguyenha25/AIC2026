"""N-03 — KIỂM KÊ EVIDENCE Q&A VÀ TRAKE. Chủ tệp: Nguyên.

Bối cảnh (Tai_lieu_ky_thuat_SAGE_QA_TRAKE_AIC2026.docx, mục "Hiện trạng
Phase 2"): phép đo Q&A cũ chạy 18 câu, 13/18 THIẾU ẢNH VIDEO GT — điểm thấp
lẫn lộn giữa lỗi dữ liệu và lỗi model. N-03 dứt điểm việc này TRƯỚC khi bất
cứ ai bắt đầu sửa parser/reader, bằng cách kiểm tra CẢ BỐN loại bằng chứng
cho từng câu trong dev_questions.jsonl:

    - keyframe gốc   raw/keyframes/<video_id>/
    - OCR            derived/ocr/<video_id>.jsonl   (paths.ocr_file)
    - ASR            derived/asr/<video_id>.jsonl   (paths.asr_file)
    - dense frame    derived/frames_dense/<video_id>/  (CHỈ bắt buộc cho
                      chuoi_su_kien — mỗi giai_doan phải có >=1 ảnh dense
                      rơi vào đúng [frame_start, frame_end])

Câu nào ĐỦ cả các loại bằng chứng mà loại câu đó cần thì được đánh dấu
"clean" và đưa vào dev_baseline_clean.jsonl — đây là input tham chiếu
Ngân dùng ở Gate 0 tối nay (20:00). Câu "missing" KHÔNG bị xoá khỏi bộ
dev gốc, chỉ bị loại khỏi benchmark cho tới khi bổ sung xong bằng chứng.

Chạy (PowerShell, từ thư mục gốc dự án)::

    python -u -m scripts.kiem_ke_evidence
    python -u -m scripts.kiem_ke_evidence --tep-dev D:\\aic-data\\dev\\dev_questions.jsonl

Output::

    runs/<run_id>/contracts/evidence_audit.jsonl   một dòng / câu, đúng
                                                     quy ước schema_version
                                                     1.1 trong handbook
    runs/<run_id>/manifest.json                     run_id, thời gian,
                                                     git_commit, tổng hợp
    dev/dev_questions_baseline_clean.jsonl          NGUYÊN VĂN các dòng
                                                     dev gốc đã "clean"

Mã thoát
    0   audit chạy hết, có in được bảng tổng hợp (dù coverage < 100%)
    1   không đọc được dev_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.aic2026.frame_map import FrameMap  # noqa: E402
from src.aic2026.paths import (  # noqa: E402
    ASR_DIR,
    DEV_DIR,
    DEV_QUERIES_PATH,
    FRAMES_DENSE_DIR,
    KEYFRAMES_DIR,
    OCR_DIR,
    RUNS_DIR,
    asr_file,
    ocr_file,
    resolve,
)

SCHEMA_VERSION = "1.1"

LOAI_SANG_NHAN = {
    "mo_ta": "kis",
    "hoi_dap": "qa",
    "chuoi_su_kien": "trake",
}


# ---------------------------------------------------------------------------
# Đọc dev_questions.jsonl — GIỮ NGUYÊN dòng thô để có thể ghi lại y hệt vào
# baseline_clean (không đi qua json.dumps lại, tránh đổi thứ tự khoá/khoảng
# trắng của người viết gốc).
# ---------------------------------------------------------------------------
def doc_dev_tho(duong_dan: Path) -> list[tuple[str, dict]]:
    """Trả về [(dòng thô đã strip, dict đã parse), ...]."""
    if not duong_dan.exists():
        raise FileNotFoundError(
            f"Không thấy {duong_dan}. Chưa có bộ dev hợp nhất — kiểm tra "
            "lại đường dẫn --tep-dev hoặc Task 11 đã xong chưa."
        )
    ra: list[tuple[str, dict]] = []
    with duong_dan.open("r", encoding="utf-8-sig") as f:
        for so_dong, dong in enumerate(f, start=1):
            dong = dong.strip()
            if not dong:
                continue
            try:
                q = json.loads(dong)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{duong_dan.name} dòng {so_dong} không phải JSON hợp lệ "
                    f"({exc}). Sửa trước khi audit — bỏ qua âm thầm sẽ làm "
                    "sai tổng số câu của baseline_clean."
                ) from exc
            ra.append((dong, q))
    if not ra:
        raise ValueError(f"{duong_dan.name} không có dòng nào.")
    return ra


# ---------------------------------------------------------------------------
# Kiểm tra từng loại bằng chứng
# ---------------------------------------------------------------------------
def co_keyframe_goc(video_id: str) -> bool:
    """Dùng resolve() của paths.py — TỰ CHUI QUA TẦNG BỌC do giải nén sinh
    ra, giống hệt cách keyframe_image()/objects_file() đã làm. Không tự
    ghép KEYFRAMES_DIR / video_id, vì ảnh thật có thể nằm sâu hơn một tầng
    (raw/keyframes/keyframes-aic25-b1/L28_V001/...) — ghép tay là rơi đúng
    cái bẫy README đã cảnh báo và báo THIẾU oan cho video đang có sẵn."""
    thu_muc = resolve(KEYFRAMES_DIR, video_id)
    if thu_muc is None:
        return False
    return any(thu_muc.glob("*.jpg"))


def co_ocr(video_id: str) -> bool:
    tep = ocr_file(video_id)
    return tep.exists() and tep.stat().st_size > 0


def co_asr(video_id: str) -> bool:
    tep = asr_file(video_id)
    return tep.exists() and tep.stat().st_size > 0


def _khung_dense_co_san(video_id: str) -> set[int]:
    """Tập frame_idx (suy từ tên tệp 6 chữ số) đang có trong frames_dense."""
    thu_muc = FRAMES_DENSE_DIR / video_id
    if not thu_muc.is_dir():
        return set()
    ra = set()
    for tep in thu_muc.glob("*.jpg"):
        ten = tep.stem
        if ten.isdigit():
            ra.add(int(ten))
    return ra


def dense_phu_su_kien(video_id: str, cac_giai_doan: list[dict]) -> list[dict]:
    """Với mỗi giai_doan, báo có >=1 ảnh dense rơi vào [frame_start, frame_end]
    hay không. Trả về danh sách chi tiết từng sự kiện — không chỉ true/false
    gộp, vì một sự kiện thiếu là đủ để coi cả câu TRAKE là 'missing'."""
    co_san = _khung_dense_co_san(video_id)
    ket_qua = []
    for i, gd in enumerate(cac_giai_doan):
        s, e = gd["frame_start"], gd["frame_end"]
        phu = any(s <= f <= e for f in co_san)
        ket_qua.append(
            {
                "su_kien_thu": i,
                "su_kien": gd.get("su_kien"),
                "frame_start": s,
                "frame_end": e,
                "co_dense": phu,
            }
        )
    return ket_qua


def kiem_frame_map(video_id: str) -> bool:
    """frame_map.parquet có dữ liệu cho video này không — điều kiện nền cho
    MỌI thứ khác (không có frame_map thì không dịch được s/e sang gì cả)."""
    try:
        fm = FrameMap.load(video_id)
        return len(fm.rows) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Kiểm kê một câu
# ---------------------------------------------------------------------------
def kiem_ke_mot_cau(q: dict) -> dict:
    query_id = str(q["id"])
    loai = q.get("loai_truy_van")
    nhan = LOAI_SANG_NHAN.get(loai, loai)
    video_id = q["video_id"]

    # evidence: LUÔN ghi đủ cả bốn mục để báo cáo minh bạch, kể cả mục
    # không bắt buộc với loại câu này (vd OCR của một câu KIS) — nhưng
    # "thieu" (quyết định status) chỉ tính đúng mục THẬT SỰ bắt buộc theo
    # loại câu, tránh báo oan một câu KIS là "thiếu OCR".
    evidence: dict[str, object] = {
        "frame_map": kiem_frame_map(video_id),
        "keyframe_goc": co_keyframe_goc(video_id),
        "ocr": co_ocr(video_id),
        "asr": co_asr(video_id),
    }

    if nhan == "trake":
        chi_tiet_dense = dense_phu_su_kien(video_id, q.get("cac_giai_doan", []))
        evidence["dense_theo_su_kien"] = chi_tiet_dense
        evidence["dense_du"] = bool(chi_tiet_dense) and all(
            sk["co_dense"] for sk in chi_tiet_dense
        )

    # Bằng chứng TỐI THIỂU bắt buộc theo loại câu (khớp routing trong
    # handbook: OCR/ASR là nguồn CHÍNH tuỳ câu, nhưng frame_map + keyframe
    # gốc là điều kiện nền cho MỌI loại; QA cần thêm ít nhất một trong
    # OCR/ASR để reader có khả năng trả lời — không đòi cả hai; TRAKE
    # cần thêm dense đủ cho mọi giai_doan).
    thieu: list[str] = []
    if not evidence["frame_map"]:
        thieu.append("frame_map")
    if not evidence["keyframe_goc"]:
        thieu.append("keyframe_goc")
    if nhan == "qa" and not (evidence["ocr"] or evidence["asr"]):
        thieu.append("ocr_hoac_asr")
    if nhan == "trake" and not evidence.get("dense_du"):
        thieu.append("dense")

    status = "ok" if not thieu else "missing"

    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": query_id,
        "task": nhan,
        "video_id": video_id,
        "evidence": evidence,
        "missing": thieu,
        "status": status,
        "error": None,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="N-03 — kiểm kê evidence Q&A/TRAKE.")
    ap.add_argument(
        "--tep-dev",
        default=str(DEV_QUERIES_PATH),
        help="Đường dẫn dev_questions.jsonl (mặc định lấy từ paths.py).",
    )
    doi_so = ap.parse_args()
    duong_dan_dev = Path(doi_so.tep_dev)

    print("=" * 72)
    print("N-03 — KIỂM KÊ EVIDENCE Q&A VÀ TRAKE (Nguyên, D1 29/08)")
    print("=" * 72)

    try:
        dong_tho = doc_dev_tho(duong_dan_dev)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LỖI: {exc}")
        return 1

    ban_ghi = []
    dong_clean = []
    for dong, q in dong_tho:
        try:
            bg = kiem_ke_mot_cau(q)
        except KeyError as exc:
            bg = {
                "schema_version": SCHEMA_VERSION,
                "query_id": str(q.get("id", "?")),
                "task": q.get("loai_truy_van"),
                "video_id": q.get("video_id"),
                "evidence": {},
                "missing": [f"truong_thieu:{exc}"],
                "status": "error",
                "error": str(exc),
            }
        ban_ghi.append(bg)
        if bg["status"] == "ok":
            dong_clean.append(dong)

    # ---- ghi output ----
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    run_id = f"{ts}_N03_kiem_ke_evidence"
    run_folder = RUNS_DIR / run_id
    contracts_folder = run_folder / "contracts"
    contracts_folder.mkdir(parents=True, exist_ok=True)

    audit_path = contracts_folder / "evidence_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as f:
        for bg in ban_ghi:
            f.write(json.dumps(bg, ensure_ascii=False) + "\n")

    DEV_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = DEV_DIR / "dev_questions_baseline_clean.jsonl"
    with clean_path.open("w", encoding="utf-8") as f:
        for dong in dong_clean:
            f.write(dong + "\n")

    # ---- tổng hợp theo loại câu ----
    tong = len(ban_ghi)
    theo_loai: dict[str, dict[str, int]] = {}
    thieu_dem: dict[str, int] = {}
    video_thieu: dict[str, set[str]] = {}
    for bg in ban_ghi:
        nhan = bg["task"] or "?"
        d = theo_loai.setdefault(nhan, {"tong": 0, "ok": 0, "missing": 0})
        d["tong"] += 1
        d["ok" if bg["status"] == "ok" else "missing"] += 1
        for lydo in bg["missing"]:
            thieu_dem[lydo] = thieu_dem.get(lydo, 0) + 1
            video_thieu.setdefault(lydo, set()).add(bg["video_id"] or "?")

    so_ok = sum(1 for bg in ban_ghi if bg["status"] == "ok")

    manifest = {
        "run_id": run_id,
        "task": "N-03",
        "owner": "Nguyen",
        "git_commit": _git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dev_source": str(duong_dan_dev),
        "tong_so_cau": tong,
        "so_cau_clean": so_ok,
        "ty_le_clean": round(so_ok / tong, 4) if tong else 0.0,
        "theo_loai": theo_loai,
        "ly_do_thieu": thieu_dem,
        "output": {
            "evidence_audit": str(audit_path.relative_to(RUNS_DIR.parent)),
            "dev_baseline_clean": str(clean_path.relative_to(RUNS_DIR.parent)),
        },
    }
    manifest_path = run_folder / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- in báo cáo ----
    print(f"\nTổng số câu trong bộ dev : {tong}")
    print(f"Số câu ĐỦ evidence (clean): {so_ok} ({so_ok/tong:.1%})" if tong else "")
    print()
    print(f"{'Loại câu':<14}{'Tổng':>6}{'OK':>6}{'Thiếu':>8}")
    for nhan, d in sorted(theo_loai.items()):
        print(f"{nhan:<14}{d['tong']:>6}{d['ok']:>6}{d['missing']:>8}")

    if thieu_dem:
        print("\nLý do thiếu (số câu bị ảnh hưởng / số video liên quan):")
        for lydo, dem in sorted(thieu_dem.items(), key=lambda kv: -kv[1]):
            n_video = len(video_thieu[lydo])
            print(f"  - {lydo:<16} {dem:>3} câu, {n_video:>3} video")
            vd = sorted(video_thieu[lydo])[:8]
            print(f"      vd: {', '.join(vd)}" + (" ..." if n_video > 8 else ""))

    print(f"\nĐã ghi: {audit_path}")
    print(f"Đã ghi: {clean_path}  ({len(dong_clean)} câu)")
    print(f"Manifest: {manifest_path}")

    if tong and so_ok == tong:
        print("\nKẾT LUẬN: ĐẠT — 100% evidence coverage trên toàn bộ dev.")
    elif so_ok == 0:
        print("\nKẾT LUẬN: KHÔNG ĐẠT — không có câu nào đủ evidence.")
    else:
        print(
            "\nKẾT LUẬN: dev_questions_baseline_clean.jsonl là tập ĐÃ LỌC "
            f"({so_ok}/{tong} câu), coverage trong tập này = 100% theo định "
            "nghĩa 'tập đánh giá sạch'. Dùng tệp này cho Gate 0, KHÔNG dùng "
            "dev_questions.jsonl gốc để đo baseline."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())