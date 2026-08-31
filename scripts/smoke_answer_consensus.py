"""
QA-V4 smoke test — chạy candidate generation thật rồi đưa qua answer consensus.

Mặc định chỉ chạy 1 câu QA để kiểm tra integration.
Không dùng GT để chọn consensus. GT chỉ được in ở cuối để người chạy tự đối chiếu.

Ví dụ:
    python -u -m scripts.smoke_answer_consensus
    python -u -m scripts.smoke_answer_consensus --limit 2
    python -u -m scripts.smoke_answer_consensus --khong-vlm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Giữ import order an toàn trên Windows của repo.
try:
    import torch  # noqa: F401
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.aic2026.paths import DATA_ROOT  # noqa: E402
from src.aic2026.qa_answer import (
    BoDocAnh,
    HitQALanCan,
    tra_loi_bang_ocr,
    tra_loi_bang_asr,
    tra_loi_bang_vlm,
)
from scripts.answer_consensus import choose_consensus  # noqa: E402


DEFAULT_DEV = DATA_ROOT / "dev" / "dev_questions_baseline_clean.jsonl"


def doc_cau_qa(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON lỗi tại {path}:{line_no}: {e}"
                ) from e

            if row.get("loai_truy_van") == "hoi_dap":
                rows.append(row)

    return rows


def _fmt_conf(x) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "-"


def parse_args():
    p = argparse.ArgumentParser(
        description="QA-V4 smoke — candidate generation thật -> consensus"
    )
    p.add_argument(
        "--tep",
        type=Path,
        default=DEFAULT_DEV,
        help="JSONL dev; mặc định dev_questions_baseline_clean.jsonl",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=1,
        help="số câu QA chạy smoke; mặc định 1",
    )
    p.add_argument(
        "--khong-vlm",
        action="store_true",
        help="không nạp BLIP; chỉ dùng OCR/ASR/fallback",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.limit <= 0:
        raise ValueError("--limit phải > 0")

    questions = doc_cau_qa(args.tep)
    if not questions:
        print(f"Không có câu hỏi-đáp trong {args.tep}")
        return 1

    questions = questions[: args.limit]

    print("=" * 92)
    print("QA-V4 — REAL-DATA CONSENSUS SMOKE")
    print("=" * 92)
    print(f"Dev file        : {args.tep}")
    print(f"Số câu          : {len(questions)}")
    print(f"VLM             : {'TẮT' if args.khong_vlm else 'BẬT'}")
    print("Nguồn candidate : OCR + ASR + VLM" if not args.khong_vlm else "Nguồn candidate : OCR + ASR")
    print()

    bo_doc = None

    # Windows: nếu dùng BLIP, phải nạp model trước FrameMap/pandas
    # để tránh xung đột native runtime.
    if not args.khong_vlm:
        bo_doc = BoDocAnh()
        bo_doc._nap()

    from src.aic2026.frame_map import FrameMap
    from src.aic2026.qa_answer import HitQALanCan

    ok = 0

    for idx, q in enumerate(questions, 1):
        qid = str(q.get("id", ""))
        question = str(q.get("cau_hoi", ""))
        gt = str(q.get("cau_tra_loi", ""))
        gt_video = str(q.get("video_id", ""))

        print()
        print("-" * 92)
        print(f"[{idx}/{len(questions)}] Câu {qid}")
        print(f"Question : {question}")

        t0 = time.perf_counter()

        fm = FrameMap.load(gt_video)

        bat_dau = int(q["frame_start"])
        ket_thuc = int(q["frame_end"])
        pts = float(q["pts_time"])

        row = fm.nearest_by_time(pts)

        if not (bat_dau <= int(row.frame_idx) <= ket_thuc):
            fps = float(row.fps)
            if fps > 0:
                pts_giua = ((bat_dau + ket_thuc) / 2.0) / fps
                row = fm.nearest_by_time(pts_giua)

        if not (bat_dau <= int(row.frame_idx) <= ket_thuc):
            print("Không tìm được GT keyframe nằm trong range.")
            continue

        hits = [
            HitQALanCan(
                video_id=gt_video,
                n=int(row.n),
                score=1.0,
                frame_idx=int(row.frame_idx),
                pts_time=float(row.pts_time),
                source="oracle_gt",
            )
        ]

        oracle_ms = (time.perf_counter() - t0) * 1000.0

        # Smoke QA-V4 chỉ cần vài candidate độc lập.
        # Tắt mở rộng lân cận để dễ nhìn đúng số candidate.
        t1 = time.perf_counter()
        
        hit = hits[0]
        candidates = []

        # OCR
        d_ocr = tra_loi_bang_ocr(
            question,
            str(hit.video_id),
            int(hit.n),
        )
        if d_ocr is not None:
            d_ocr.frame_idx = int(hit.frame_idx)
            candidates.append(d_ocr)

        # ASR
        d_asr = tra_loi_bang_asr(
            question,
            str(hit.video_id),
            float(hit.pts_time),
        )
        if d_asr is not None:
            d_asr.frame_idx = int(hit.frame_idx)
            candidates.append(d_asr)

        # VLM
        if not args.khong_vlm:
            d_vlm = tra_loi_bang_vlm(
                question,
                str(hit.video_id),
                int(hit.n),
                bo_doc,
            )
            if (
                d_vlm is not None
                and str(d_vlm.nguon) != "du_phong"
            ):
                d_vlm.frame_idx = int(hit.frame_idx)
                candidates.append(d_vlm)

        reader_ms = (time.perf_counter() - t1) * 1000.0

        print(
            f"Oracle frame : {int(hit.frame_idx)} | {oracle_ms:.0f} ms"
        )
        print(
            f"Reader    : {len(candidates)} candidates | {reader_ms:.0f} ms"
        )
        print()
        print(
            f"{'#':<3} {'video':<12} {'frame':>8} "
            f"{'source':<12} {'conf':>7}  answer"
        )
        print("-" * 92)

        for j, d in enumerate(candidates, 1):
            print(
                f"{j:<3} {str(hit.video_id):<12} "
                f"{int(hit.frame_idx):>8} "
                f"{str(d.nguon):<12} "
                f"{_fmt_conf(d.do_tin):>7}  "
                f"{d.van_ban!r}"
            )

        # QUAN TRỌNG: consensus chỉ nhận question + candidate.
        # Không truyền gt_answer.
        result = choose_consensus(question, candidates)

        print()
        print("CONSENSUS")
        print(f"  final_answer     : {result.final_answer!r}")
        print(f"  canonical_answer : {result.canonical_answer!r}")
        print(f"  question_type    : {result.question_type}")
        print(f"  support          : {result.support}")
        print(f"  sources          : {list(result.sources)}")
        print(f"  confidence       : {result.confidence:.3f}")
        print(f"  used_fallback    : {result.used_fallback}")

        if result.alternatives:
            print("  alternatives:")
            for alt in result.alternatives:
                print(
                    f"    - {alt['answer']!r} "
                    f"(canonical={alt['canonical_answer']!r}, "
                    f"support={alt['support']}, "
                    f"sources={alt['sources']})"
                )

        # GT chỉ để người chạy nhìn sau khi consensus đã xong.
        print()
        print("ĐỐI CHIẾU THỦ CÔNG — KHÔNG THAM GIA CHỌN WINNER")
        print(f"  GT video  : {gt_video}")
        print(f"  GT answer : {gt!r}")

        if result.final_answer.strip():
            ok += 1

    print()
    print("=" * 92)
    print(
        f"Smoke hoàn tất: {ok}/{len(questions)} câu có final_answer không rỗng."
    )
    print(
        "Nếu candidate/source/confidence và consensus nhìn hợp lý, "
        "QA-V4 đã qua smoke integration."
    )
    print("=" * 92)

    return 0 if ok == len(questions) else 2


if __name__ == "__main__":
    raise SystemExit(main())
