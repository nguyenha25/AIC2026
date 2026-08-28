"""Nghiệm thu Việc 10: caption phủ đúng keyframe và đã được nạp vào FTS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.enrich.caption import CaptionSearchIndex  # noqa: E402
from aic2026.kf_index import hang_sang_moc, so_hang  # noqa: E402
from aic2026.paths import CAPTIONS_DIR  # noqa: E402


def doc_jsonl(tep: Path) -> tuple[list[dict], list[str]]:
    ra, loi = [], []
    if not tep.exists():
        return ra, ["chưa có tệp caption"]
    for so_dong, dong in enumerate(tep.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not dong.strip():
            continue
        try:
            d = json.loads(dong)
        except json.JSONDecodeError as exc:
            loi.append(f"dòng {so_dong} sai JSON: {exc.msg}")
            continue
        if not isinstance(d, dict):
            loi.append(f"dòng {so_dong} không phải object")
            continue
        ra.append(d)
    return ra, loi


def kiem_video(video_id: str, ti_le_toi_thieu: float = 0.95) -> dict:
    tep = CAPTIONS_DIR / f"{video_id}.jsonl"
    ban_ghi, loi = doc_jsonl(tep)
    try:
        cac_moc = [hang_sang_moc(video_id, i) for i in range(so_hang(video_id))]
    except Exception as exc:
        return {"video_id": video_id, "dat": False, "loi": loi + [f"không đọc được frame_map: {exc}"]}

    theo_n: dict[int, dict] = {}
    trung_n: list[int] = []
    for d in ban_ghi:
        try:
            n = int(d["n"])
        except (KeyError, TypeError, ValueError):
            loi.append("có bản ghi thiếu/sai n")
            continue
        if n in theo_n:
            trung_n.append(n)
        theo_n[n] = d
    if trung_n:
        loi.append(f"trùng n: {sorted(set(trung_n))[:10]}")

    for m in cac_moc:
        d = theo_n.get(m.n)
        if d is None:
            loi.append(f"thiếu n={m.n}")
            continue
        try:
            if str(d.get("video_id")) != video_id:
                loi.append(f"n={m.n} sai video_id")
            if int(d.get("frame_idx")) != m.frame_idx:
                loi.append(f"n={m.n} sai frame_idx")
            if abs(float(d.get("pts_time")) - m.pts_time) > 1e-3:
                loi.append(f"n={m.n} sai pts_time")
            if d.get("status") not in ("ok", "missing_image", "empty_caption"):
                loi.append(f"n={m.n} thiếu/sai status")
        except (TypeError, ValueError):
            loi.append(f"n={m.n} sai kiểu dữ liệu")

    n_hop_le = {m.n for m in cac_moc}
    n_thua = sorted(set(theo_n) - n_hop_le)
    if n_thua:
        loi.append(f"thừa n: {n_thua[:10]}")
    if len(ban_ghi) != len(cac_moc):
        loi.append(f"số dòng {len(ban_ghi)} != số keyframe {len(cac_moc)}")

    so_caption = sum(bool(str(theo_n.get(m.n, {}).get("caption") or "").strip()) for m in cac_moc)
    ti_le = so_caption / len(cac_moc) if cac_moc else 0.0
    if ti_le < ti_le_toi_thieu:
        loi.append(f"caption không rỗng chỉ {ti_le:.1%}, cần >= {ti_le_toi_thieu:.1%}")

    # Không in hàng trăm lỗi cùng loại; báo cáo vẫn đủ để xác định video cần chạy lại.
    loi = list(dict.fromkeys(loi))
    return {
        "video_id": video_id,
        "dat": not loi,
        "so_keyframe": len(cac_moc),
        "so_caption": so_caption,
        "ti_le": ti_le,
        "loi": loi[:20],
    }


def doc_cau_mo_ta(tep: Path) -> list[dict]:
    ra = []
    for dong in tep.read_text(encoding="utf-8-sig").splitlines():
        if not dong.strip():
            continue
        q = json.loads(dong)
        if q.get("loai_truy_van") == "mo_ta":
            ra.append(q)
    return ra


def main() -> int:
    p = argparse.ArgumentParser(description="Nghiệm thu Việc 10 — captions")
    p.add_argument("--video", action="append", default=[], help="có thể lặp nhiều lần")
    p.add_argument("--shard", action="append", default=[], help="ví dụ L23")
    p.add_argument("--tep-dev", type=Path, help="báo phủ nhóm câu mo_ta của dev v2")
    p.add_argument("--ti-le-toi-thieu", type=float, default=0.95)
    p.add_argument("--khong-kiem-fts", action="store_true")
    args = p.parse_args()

    if not 0.0 <= args.ti_le_toi_thieu <= 1.0:
        p.error("--ti-le-toi-thieu phải nằm trong [0, 1]")

    video_ids = list(args.video)
    co_tep = sorted(x.stem for x in CAPTIONS_DIR.glob("*.jsonl")) if CAPTIONS_DIR.exists() else []
    if args.shard:
        video_ids.extend(v for v in co_tep if any(v.startswith(s) for s in args.shard))
    if not video_ids:
        video_ids = co_tep
    video_ids = list(dict.fromkeys(video_ids))
    if not video_ids:
        print(f"CHƯA ĐẠT: chưa có caption JSONL trong {CAPTIONS_DIR}")
        return 1

    ket_qua = [kiem_video(v, args.ti_le_toi_thieu) for v in video_ids]
    print("VIỆC 10 — NGHIỆM THU CAPTION\n")
    for kq in ket_qua:
        nhan = "ĐẠT" if kq["dat"] else "CHƯA ĐẠT"
        print(
            f"{nhan:<10} {kq['video_id']:<12} "
            f"{kq.get('so_caption', 0):>5,}/{kq.get('so_keyframe', 0):<5,} "
            f"caption ({kq.get('ti_le', 0):.1%})"
        )
        for loi in kq.get("loi", []):
            print(f"  LỖI: {loi}")

    dat = sum(x["dat"] for x in ket_qua)
    dat_fts = True
    if not args.khong_kiem_fts:
        tk = CaptionSearchIndex().thong_ke()
        dat_fts = tk["so_ban_ghi_caption"] > 0
        print(
            f"\nFTS: {tk['so_ban_ghi_caption']:,} caption thuộc "
            f"{tk['so_video_caption']:,} video" + ("" if dat_fts else " — CHƯA ĐẠT")
        )

    dat_dev = True
    if args.tep_dev:
        cau = doc_cau_mo_ta(args.tep_dev)
        dat_video = {x["video_id"] for x in ket_qua if x["dat"]}
        co_dap_an = [q for q in cau if str(q.get("video_id")) in dat_video]
        dat_dev = len(co_dap_an) == len(cau) and bool(cau)
        print(
            f"DEV THUẦN THỊ GIÁC: đáp án đã có caption đạt cho "
            f"{len(co_dap_an)}/{len(cau)} câu" + ("" if dat_dev else " — chưa phủ đủ")
        )

    print(f"\nKẾT LUẬN: {dat}/{len(ket_qua)} video đạt schema và độ phủ caption.")
    return 0 if dat == len(ket_qua) and dat_fts and dat_dev else 1


if __name__ == "__main__":
    raise SystemExit(main())
