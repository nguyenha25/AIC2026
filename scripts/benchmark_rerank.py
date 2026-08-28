"""Nghiệm thu Việc 4: baseline CLIP B/32 so với rerank top-100.

Chạy khi có dev v2:

    python -u -m scripts.benchmark_rerank --tep D:/aic-data/dev/dev_v2.jsonl

Lệnh tự làm bốn việc trên CÙNG tập ứng viên thô:

1. chấm baseline CLIP B/32;
2. rerank top-100 bằng mô hình mạnh hơn và chấm lần 1;
3. xoá cache vector ảnh, mã hoá lại và chấm lần 2;
4. chỉ công nhận nghiệm thu nếu hai lượt có cùng thứ tự + điểm và không thiếu ảnh.

Không có dev v2 vẫn có thể smoke-test một vài câu bằng ``--so-cau 2``. Báo cáo
sẽ ghi rõ cỡ mẫu chưa đủ để kết luận điểm tăng.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Trên Windows phải nạp torch trước faiss; xem rank/search.py.
try:
    import torch  # noqa: F401, E402
except ImportError:
    pass

from aic2026.eval import compute_final_score  # noqa: E402
from aic2026.paths import (
    DEV_QUERIES_PATH,
    RUNS_DIR,
    THUMBNAILS_DIR,
    bang_tep_theo_so,
)  # noqa: E402
from aic2026.rank.config import so_ung_vien_moi_nguon  # noqa: E402
from aic2026.rank.search import run_query, tim_ung_vien_clip  # noqa: E402
from aic2026.rerank import Reranker  # noqa: E402
from aic2026.submit import KIS, QA  # noqa: E402
from scripts.run_scoring import build_gt, doc_dev_questions  # noqa: E402


CO_MAU_KET_LUAN_TOI_THIEU = 40


def _duong_dan_thumbnail(video_id: str, n: int) -> Path:
    base = THUMBNAILS_DIR / str(video_id)
    found = bang_tep_theo_so(base, ".jpg").get(int(n))
    return found if found is not None else base / f"{int(n):03d}.jpg"



def _chu_ky(hits) -> list[tuple[str, int, float]]:
    return [
        (str(h.video_id), int(h.n), float(np.float32(h.score)))
        for h in hits
    ]


def _bam(cac_chu_ky: list[list[tuple[str, int, float]]]) -> str:
    du_lieu = json.dumps(
        cac_chu_ky,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(du_lieu).hexdigest()


def _cham(q: dict, hits: list) -> float:
    qid, task, gt = build_gt(q)
    cau = str(q["cau_hoi"])

    def _tra_da_co(_cau: str, _k: int):
        return list(hits)

    ket_qua = run_query(
        cau,
        query_id=qid,
        task=task,
        dap_an=gt.get("gt_answer"),
        ghi_tep=False,
        tim_ung_vien=_tra_da_co,
    )
    dong = []
    for h in ket_qua.ket_qua:
        d = {"video_id": h.video_id, "frame_id": h.frame_idx}
        if task == QA:
            # Việc 4 chỉ đo tìm ảnh, không trộn chất lượng sinh đáp án Việc 11.
            d["answer"] = gt["gt_answer"]
        dong.append(d)
    return float(compute_final_score(gt, dong, task)["final_score"])


def _cau_cho_clip(cau_goc: str, nguon_mo_rong: str | None) -> str:
    if not nguon_mo_rong:
        return cau_goc
    from aic2026.query_expand import mo_rong

    return mo_rong(cau_goc, nguon=nguon_mo_rong, bat_buoc=True).cum_chinh


def _trung_binh(ds: list[float]) -> float:
    return float(sum(ds) / len(ds)) if ds else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description="Nghiệm thu Task 4 — rerank top-100")
    p.add_argument("--tep", default=str(DEV_QUERIES_PATH))
    p.add_argument("--so-dau", type=int, default=100)
    p.add_argument("--cach-gop", choices=["thay", "rrf"], default=None)
    p.add_argument("--so-cau", type=int, default=0,
                   help="0 = toàn bộ; số dương chỉ dùng smoke-test")
    p.add_argument("--nguon-mo-rong", choices=["tu_dien", "marian", "llm"])
    p.add_argument("--cho-phep-thieu-anh", action="store_true",
                   help="chỉ debug; báo cáo có ảnh thiếu không được dùng chốt điểm")
    p.add_argument(
        "--dung-thumbnail",
        action="store_true",
        help="dung derived/thumbnails thay cho raw/keyframes",
    )
    p.add_argument("--ra", default=None, help="đường dẫn báo cáo JSON")
    args = p.parse_args()

    if args.so_dau <= 0:
        print("LỖI: --so-dau phải lớn hơn 0")
        return 2

    try:
        tat_ca = doc_dev_questions(Path(args.tep))
    except (FileNotFoundError, ValueError) as exc:
        print(f"LỖI BỘ DEV: {exc}")
        return 2

    cau_hoi = []
    for q in tat_ca:
        try:
            _, task, _ = build_gt(q)
        except (KeyError, ValueError) as exc:
            print(f"Bỏ câu {q.get('id', '?')}: {exc}")
            continue
        if task in (KIS, QA):
            cau_hoi.append(q)
    if args.so_cau > 0:
        cau_hoi = cau_hoi[: args.so_cau]
    if not cau_hoi:
        print("LỖI: bộ dev không có câu KIS/QA hợp lệ.")
        return 2

    nguon_anh = "thumbnail" if args.dung_thumbnail else "keyframe_goc"
    bo = Reranker(
        duong_dan_anh=_duong_dan_thumbnail
        if args.dung_thumbnail
        else None
    )
    cach_gop = args.cach_gop or bo.cach_gop_mac_dinh
    so_lay_tho = max(args.so_dau, so_ung_vien_moi_nguon())

    print(f"Nguon anh     : {nguon_anh}")
    print(f"Dev           : {args.tep} ({len(cau_hoi)} câu KIS/QA)")
    print(f"Tầng thô      : CLIP B/32, lấy {so_lay_tho} ứng viên")
    print(f"Rerank        : {bo.ten_day_du}, top-{args.so_dau}, {cach_gop}, batch={bo.kich_thuoc_lo}")
    print(f"Câu cho CLIP  : {args.nguon_mo_rong or 'nguyên bản'}")
    print("Hai lượt rerank đều mã hoá ảnh lại từ đầu.\n")

    # Tra tầng thô đúng MỘT lần rồi đóng băng danh sách. Hai lượt rerank nhờ
    # vậy chỉ khác ở lần mã hoá lại ảnh, không lẫn dao động từ tầng tìm kiếm.
    cac_muc = []
    print("Bước 1/3 — lấy và chấm ứng viên thô")
    for i, q in enumerate(cau_hoi, 1):
        qid, _, _ = build_gt(q)
        try:
            cau_clip = _cau_cho_clip(str(q["cau_hoi"]), args.nguon_mo_rong)
            hits_goc = list(tim_ung_vien_clip(cau_clip, so_lay_tho))
            d0 = _cham(q, hits_goc)
            cac_muc.append({
                "q": q,
                "query_id": qid,
                "cau_clip": cau_clip,
                "hits_goc": hits_goc,
                "baseline": d0,
            })
            print(f"  [{i:>3}/{len(cau_hoi)}] {qid:<12} baseline={d0:.4f}")
        except Exception as exc:
            print(f"  [{i:>3}/{len(cau_hoi)}] {qid}: LỖI {type(exc).__name__}: {exc}")
            return 2

    def _chay_luot(so_luot: int):
        # Xoá MỘT lần ở đầu lượt: hai lượt độc lập, nhưng trong mỗi lượt cache
        # vẫn dùng lại ảnh trùng giữa các câu hỏi để không lãng phí thời gian.
        bo.xoa_nho_anh()
        ket_qua_luot = []
        print(f"\nBước {so_luot + 1}/3 — rerank lượt {so_luot}")
        for i, muc in enumerate(cac_muc, 1):
            try:
                hits, bc = bo.xep_lai(
                    muc["cau_clip"],
                    muc["hits_goc"],
                    so_dau=args.so_dau,
                    cach_gop=cach_gop,
                )
                diem = _cham(muc["q"], hits)
                ket_qua_luot.append({
                    "hits": hits,
                    "chu_ky": _chu_ky(hits),
                    "diem": diem,
                    "bao_cao": bc,
                })
                print(
                    f"  [{i:>3}/{len(cac_muc)}] {muc['query_id']:<12} "
                    f"điểm={diem:.4f} | thiếu={bc.so_thieu_anh} | "
                    f"{bc.thoi_gian_ms:.0f} ms"
                )
            except Exception as exc:
                print(
                    f"  [{i:>3}/{len(cac_muc)}] {muc['query_id']}: "
                    f"LỖI {type(exc).__name__}: {exc}"
                )
                raise
        return ket_qua_luot

    try:
        lan_1 = _chay_luot(1)
        lan_2 = _chay_luot(2)
    except Exception:
        return 2

    diem_goc = [float(m["baseline"]) for m in cac_muc]
    diem_lan_1 = [float(r["diem"]) for r in lan_1]
    diem_lan_2 = [float(r["diem"]) for r in lan_2]
    chu_ky_1 = [r["chu_ky"] for r in lan_1]
    chu_ky_2 = [r["chu_ky"] for r in lan_2]
    tong_thieu = sum(r["bao_cao"].so_thieu_anh for r in lan_1)
    tat_dinh = chu_ky_1 == chu_ky_2
    chi_tiet = []

    print("\nĐối chiếu hai lượt")
    for muc, r1, r2 in zip(cac_muc, lan_1, lan_2):
        cau_tat_dinh = r1["chu_ky"] == r2["chu_ky"]
        chi_tiet.append({
            "query_id": muc["query_id"],
            "baseline": muc["baseline"],
            "rerank_lan_1": r1["diem"],
            "rerank_lan_2": r2["diem"],
            "delta": r1["diem"] - muc["baseline"],
            "tat_dinh": cau_tat_dinh,
            "bao_cao_lan_1": asdict(r1["bao_cao"]),
            "bao_cao_lan_2": asdict(r2["bao_cao"]),
        })
        print(
            f"  {muc['query_id']:<12} {muc['baseline']:.4f} -> "
            f"{r1['diem']:.4f} | lần 2 {r2['diem']:.4f} | "
            f"{'GIỐNG' if cau_tat_dinh else 'KHÁC'}"
        )

    fp_1, fp_2 = _bam(chu_ky_1), _bam(chu_ky_2)
    diem_0 = _trung_binh(diem_goc)
    diem_1 = _trung_binh(diem_lan_1)
    diem_2 = _trung_binh(diem_lan_2)
    du_co_mau = len(cau_hoi) >= CO_MAU_KET_LUAN_TOI_THIEU and args.so_cau == 0
    hop_le_du_lieu = tong_thieu == 0
    dat_nghiem_thu = tat_dinh and hop_le_du_lieu and du_co_mau and diem_1 > diem_0

    bao_cao = {
        "task": 4,
        "thoi_diem_utc": datetime.now(timezone.utc).isoformat(),
        "tep_dev": str(Path(args.tep).resolve()),
        "so_cau_kis_qa": len(cau_hoi),
        "du_co_mau_ket_luan": du_co_mau,
        "co_mau_ket_luan_toi_thieu": CO_MAU_KET_LUAN_TOI_THIEU,
        "mo_hinh": bo.ten_day_du,
        "nguon_anh": nguon_anh,
        "so_dau": args.so_dau,
        "so_lay_tho": so_lay_tho,
        "kich_thuoc_lo": bo.kich_thuoc_lo,
        "cach_gop": cach_gop,
        "nguon_mo_rong": args.nguon_mo_rong,
        "so_thieu_anh": tong_thieu,
        "tat_dinh": tat_dinh,
        "fingerprint_lan_1": fp_1,
        "fingerprint_lan_2": fp_2,
        "diem_baseline": diem_0,
        "diem_rerank_lan_1": diem_1,
        "diem_rerank_lan_2": diem_2,
        "delta": diem_1 - diem_0,
        "dat_nghiem_thu": dat_nghiem_thu,
        "chi_tiet": chi_tiet,
    }

    if args.ra:
        tep_ra = Path(args.ra)
    else:
        nhan = datetime.now().strftime("%Y%m%d_%H%M%S")
        tep_ra = RUNS_DIR / f"{nhan}_task4_rerank.json"
    tep_ra.parent.mkdir(parents=True, exist_ok=True)
    tep_ra.write_text(json.dumps(bao_cao, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Baseline       : {diem_0:.4f}")
    print(f"Rerank lần 1  : {diem_1:.4f}  (delta {diem_1 - diem_0:+.4f})")
    print(f"Rerank lần 2  : {diem_2:.4f}")
    print(f"Hai lượt      : {'GIỐNG NHAU' if tat_dinh else 'KHÁC NHAU'}")
    print(f"Ảnh thiếu      : {tong_thieu}")
    print(f"Đủ cỡ mẫu      : {'CÓ' if du_co_mau else 'CHƯA'}")
    print(f"Nghiệm thu     : {'ĐẠT' if dat_nghiem_thu else 'CHƯA ĐẠT'}")
    print(f"Báo cáo        : {tep_ra}")
    print("=" * 72)

    if not tat_dinh:
        return 3
    if tong_thieu and not args.cho_phep_thieu_anh:
        print("Không công nhận điểm: thiếu ảnh top-N. Dùng đủ keyframe rồi chạy lại.")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
