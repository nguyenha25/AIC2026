"""
TR-E1 — TẠO TẬP ĐÁNH GIÁ TRAKE SẠCH.

Chỉ xét các câu loai_truy_van == "chuoi_su_kien".

Một câu TRAKE được coi là clean khi:
    - có video_id;
    - có ít nhất một giai_doan;
    - mọi giai_doan có frame_start <= frame_end;
    - mỗi GT window [frame_start, frame_end] chứa >= 1 dense frame.

KHÔNG dùng keyframe thưa để quyết định clean/missing.

Output:
    runs/<run_id>/contracts/trake_clean.jsonl
    runs/<run_id>/contracts/trake_audit.jsonl
    runs/<run_id>/manifest.json
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

from aic2026.paths import DEV_QUERIES_PATH, FRAMES_DENSE_DIR, RUNS_DIR  # noqa: E402

SCHEMA_VERSION = "1.1"


def doc_dev(duong_dan: Path) -> list[dict]:
    if not duong_dan.exists():
        raise FileNotFoundError(f"Không thấy {duong_dan}")

    ra: list[dict] = []

    with duong_dan.open("r", encoding="utf-8-sig") as f:
        for so_dong, dong in enumerate(f, 1):
            dong = dong.strip()
            if not dong:
                continue

            try:
                q = json.loads(dong)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{duong_dan.name} dòng {so_dong} sai JSON."
                ) from exc

            if q.get("loai_truy_van") == "chuoi_su_kien":
                ra.append(q)

    if not ra:
        raise ValueError(
            f"{duong_dan} không có câu loai_truy_van=chuoi_su_kien."
        )

    return ra


def khung_dense(video_id: str) -> list[int]:
    thu_muc = FRAMES_DENSE_DIR / video_id

    if not thu_muc.is_dir():
        return []

    return sorted(
        int(p.stem)
        for p in thu_muc.glob("*.jpg")
        if p.stem.isdigit()
    )


def kiem_mot_cau(q: dict) -> dict:
    query_id = str(q.get("id", "?"))
    video_id = str(q.get("video_id", ""))

    cac_giai_doan = q.get("cac_giai_doan")
    loi: list[str] = []
    evidence: list[dict] = []

    if not video_id:
        loi.append("thieu_video_id")

    if not isinstance(cac_giai_doan, list) or not cac_giai_doan:
        loi.append("thieu_cac_giai_doan")
        cac_giai_doan = []

    dense = khung_dense(video_id) if video_id else []

    if not dense:
        loi.append("khong_co_dense")

    for thu_tu, gd in enumerate(cac_giai_doan, 1):
        try:
            dau = int(gd["frame_start"])
            cuoi = int(gd["frame_end"])
        except (KeyError, TypeError, ValueError):
            loi.append(f"event_{thu_tu}_sai_gt")
            continue

        if dau > cuoi:
            loi.append(f"event_{thu_tu}_frame_start_lon_hon_frame_end")
            continue

        trong = [f for f in dense if dau <= f <= cuoi]

        muc = {
            "event": thu_tu,
            "su_kien": gd.get("su_kien"),
            "frame_start": dau,
            "frame_end": cuoi,
            "dense_frame_count": len(trong),
            "dense_frame_idx": trong,
            "first_dense_frame": trong[0] if trong else None,
            "last_dense_frame": trong[-1] if trong else None,
            "covered": bool(trong),
        }
        evidence.append(muc)

        if not trong:
            loi.append(f"event_{thu_tu}_khong_co_dense")

    status = "clean" if not loi and evidence else "missing"

    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": query_id,
        "task": "trake",
        "video_id": video_id,
        "events": evidence,
        "event_count": len(evidence),
        "covered_event_count": sum(1 for e in evidence if e["covered"]),
        "status": status,
        "missing": loi,
        "error": None,
    }


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
    ap = argparse.ArgumentParser(
        description="TR-E1 — tạo tập đánh giá TRAKE sạch."
    )
    ap.add_argument(
        "--tep-dev",
        default=str(DEV_QUERIES_PATH),
        help="Đường dẫn dev_questions.jsonl.",
    )
    args = ap.parse_args()

    duong_dan_dev = Path(args.tep_dev)

    print("=" * 72)
    print("TR-E1 — TẠO TẬP ĐÁNH GIÁ TRAKE SẠCH")
    print("=" * 72)

    try:
        cau_trake = doc_dev(duong_dan_dev)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LỖI: {exc}")
        return 1

    audit = [kiem_mot_cau(q) for q in cau_trake]
    clean = [x for x in audit if x["status"] == "clean"]

    tong_event = sum(x["event_count"] for x in audit)
    covered_event = sum(x["covered_event_count"] for x in audit)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    run_id = f"{ts}_TR-E1_tap_trake_sach"

    run_folder = RUNS_DIR / run_id
    contracts = run_folder / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)

    clean_path = contracts / "trake_clean.jsonl"
    audit_path = contracts / "trake_audit.jsonl"

    with clean_path.open("w", encoding="utf-8") as f:
        for bg in clean:
            f.write(json.dumps(bg, ensure_ascii=False) + "\n")

    with audit_path.open("w", encoding="utf-8") as f:
        for bg in audit:
            f.write(json.dumps(bg, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "task": "TR-E1",
        "owner": "Nguyen",
        "git_commit": _git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dev_source": str(duong_dan_dev),
        "definition": (
            "TRAKE clean iff every GT event window contains at least "
            "one dense frame; sparse keyframes are not used as a "
            "clean/missing criterion."
        ),
        "tong_cau_trake": len(audit),
        "so_cau_clean": len(clean),
        "tong_event": tong_event,
        "covered_event": covered_event,
        "query_coverage": (
            round(len(clean) / len(audit), 4)
            if audit
            else 0.0
        ),
        "event_coverage": (
            round(covered_event / tong_event, 4)
            if tong_event
            else 0.0
        ),
        "output": {
            "trake_clean": str(clean_path.relative_to(RUNS_DIR.parent)),
            "trake_audit": str(audit_path.relative_to(RUNS_DIR.parent)),
        },
    }

    manifest_path = run_folder / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Câu TRAKE       : {len(audit)}")
    print(f"Câu clean       : {len(clean)}/{len(audit)}")
    print(f"GT event covered: {covered_event}/{tong_event}")

    for bg in audit:
        nhan = "ĐẠT" if bg["status"] == "clean" else "THIẾU"
        print(
            f"{nhan:<6} câu {bg['query_id']:<3} "
            f"{bg['video_id']:<12} "
            f"{bg['covered_event_count']}/{bg['event_count']} event"
        )

        for ly_do in bg["missing"]:
            print(f"       - {ly_do}")

    print(f"\nĐã ghi: {clean_path}")
    print(f"Audit  : {audit_path}")
    print(f"Manifest: {manifest_path}")

    dat = (
        len(audit) > 0
        and len(clean) == len(audit)
        and tong_event > 0
        and covered_event == tong_event
    )

    if dat:
        print(
            "\nKẾT LUẬN: ĐẠT — 100% câu TRAKE và 100% GT event "
            "window có dense evidence."
        )
        return 0

    print(
        "\nKẾT LUẬN: CHƯA ĐẠT — còn câu/event TRAKE thiếu dense evidence."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())