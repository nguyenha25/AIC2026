"""
QA-V2 — Qwen3-VL reader oracle trên đúng frame GT.

Mục tiêu
--------
Đo riêng năng lực reader Qwen3-VL khi retrieval được loại khỏi phép đo:
    câu hỏi QA -> frame GT -> Qwen3-VL -> đáp án -> exact/semantic match

KHÔNG dùng retrieval / CLIP / OCR / ASR / tra_loi() / tra_loi_theo_hang().

Khuyến nghị chạy trong môi trường riêng .venv-qwen vì QA-V2 cần phiên bản
Transformers mới hơn môi trường chính của dự án.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Chỉ import paths ở mức module. Các module có pandas / sentence-transformers
# được import sau khi Qwen đã nạp để giảm nguy cơ xung đột native library trên Windows.
from src.aic2026.paths import DEV_DIR, DEV_QUERIES_PATH, KEYFRAMES_DIR, RUNS_DIR  # noqa: E402

SCHEMA_VERSION = "1.1"
MODEL_MAC_DINH = "Qwen/Qwen3-VL-2B-Instruct"
PROMPT_MODE_MAC_DINH = "direct_vi_short"
SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def doc_qa(duong_dan: Path) -> list[dict[str, Any]]:
    if not duong_dan.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dev: {duong_dan}")

    ra: list[dict[str, Any]] = []
    with duong_dan.open("r", encoding="utf-8-sig") as f:
        for so_dong, dong in enumerate(f, start=1):
            if not dong.strip():
                continue
            try:
                q = json.loads(dong)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON lỗi ở dòng {so_dong}: {e}") from e
            if q.get("loai_truy_van") == "hoi_dap":
                ra.append(q)
    return ra


def chuan_hoa_exact(van_ban: str) -> str:
    s = unicodedata.normalize("NFKC", str(van_ban)).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(gt: str, pred: str) -> bool:
    return bool(chuan_hoa_exact(gt)) and chuan_hoa_exact(gt) == chuan_hoa_exact(pred)


def tao_prompt(cau_hoi: str, prompt_mode: str = PROMPT_MODE_MAC_DINH) -> str:
    if prompt_mode != PROMPT_MODE_MAC_DINH:
        raise ValueError(f"prompt_mode chưa hỗ trợ: {prompt_mode}")
    return f"{cau_hoi.strip()} Trả lời ngắn gọn bằng tiếng Việt, chỉ nêu đáp án."


def chon_frame_oracle(q: dict[str, Any], fm: Any) -> Any | None:
    """Chọn keyframe gần pts_time nhất nhưng bắt buộc nằm trong GT range."""
    bat_dau = int(q["frame_start"])
    ket_thuc = int(q["frame_end"])
    pts = float(q["pts_time"])

    gan = fm.nearest_by_time(pts)
    if bat_dau <= int(gan.frame_idx) <= ket_thuc:
        return gan

    fps = float(gan.fps)
    if fps > 0:
        pts_giua = ((bat_dau + ket_thuc) / 2.0) / fps
        giua = fm.nearest_by_time(pts_giua)
        if bat_dau <= int(giua.frame_idx) <= ket_thuc:
            return giua
    return None


def tim_thu_muc_video(video_id: str, root: Path = KEYFRAMES_DIR) -> Path | None:
    truc_tiep = root / video_id
    if truc_tiep.is_dir():
        return truc_tiep
    for p in root.rglob(video_id):
        if p.is_dir() and p.name == video_id:
            return p
    return None


def tim_anh_theo_n(video_id: str, n: int, root: Path = KEYFRAMES_DIR) -> Path | None:
    thu_muc = tim_thu_muc_video(video_id, root)
    if thu_muc is None:
        return None

    for digits in (3, 4, 5, 6):
        p = thu_muc / f"{int(n):0{digits}d}.jpg"
        if p.exists():
            return p

    for p in thu_muc.glob("*.jpg"):
        try:
            if int(p.stem) == int(n):
                return p
        except ValueError:
            continue
    return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def tong_hop(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok"]
    theo_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        theo_intent[str(r.get("intent", "unknown"))].append(r)

    def nhom(rs: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rs)
        if n == 0:
            return {
                "n": 0,
                "exact_match": 0,
                "semantic_match": 0,
                "exact_rate": 0.0,
                "semantic_rate": 0.0,
                "avg_latency_ms": 0.0,
            }
        exact_n = sum(bool(r.get("exact_match")) for r in rs)
        semantic_n = sum(bool(r.get("semantic_match")) for r in rs)
        return {
            "n": n,
            "exact_match": exact_n,
            "semantic_match": semantic_n,
            "exact_rate": exact_n / n,
            "semantic_rate": semantic_n / n,
            "avg_latency_ms": sum(float(r.get("latency_ms", 0.0)) for r in rs) / n,
        }

    ly_do: dict[str, int] = defaultdict(int)
    for r in records:
        if r.get("status") != "ok":
            ly_do[str(r.get("reason", "unknown"))] += 1

    return {
        "total_records": len(records),
        "ok": len(ok),
        "missing_or_error": len(records) - len(ok),
        "overall": nhom(ok),
        "by_intent": {k: nhom(v) for k, v in sorted(theo_intent.items())},
        "non_ok_reasons": dict(sorted(ly_do.items())),
    }


def ghi_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class QwenReader:
    def __init__(self, model_name: str, max_new_tokens: int = 64) -> None:
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.processor: Any = None
        self.model: Any = None
        self.device = "cpu"

    def nap(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = "cpu"

    def hoi(self, image_path: Path, prompt: str) -> str:
        if self.processor is None or self.model is None:
            raise RuntimeError("QwenReader chưa được nạp model")

        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def main() -> int:
    clean_mac_dinh = DEV_DIR / "dev_questions_baseline_clean.jsonl"
    tep_mac_dinh = clean_mac_dinh if clean_mac_dinh.exists() else DEV_QUERIES_PATH

    p = argparse.ArgumentParser(description="QA-V2 — đo Qwen3-VL reader oracle trên đúng frame GT")
    p.add_argument("--tep", default=str(tep_mac_dinh))
    p.add_argument("--limit", type=int, default=0, help="0 = chạy toàn bộ QA")
    p.add_argument("--model", default=MODEL_MAC_DINH)
    p.add_argument("--prompt-mode", default=PROMPT_MODE_MAC_DINH, choices=[PROMPT_MODE_MAC_DINH])
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--cho-phep-tai-model",
        action="store_true",
        help="cho phép Hugging Face tải model; mặc định ép offline/local cache",
    )
    args = p.parse_args()

    if not args.cho_phep_tai_model:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tep = Path(args.tep)
    cau = doc_qa(tep)
    if args.limit > 0:
        cau = cau[: args.limit]
    if not cau:
        print(f"Không có câu hoi_dap trong {tep}")
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M_QA-V2_qwen_oracle")
    run_dir = RUNS_DIR / run_id
    contracts_dir = run_dir / "contracts"
    report_dir = run_dir / "report"
    out_jsonl = contracts_dir / "qa_v2_qwen_oracle.jsonl"
    out_summary = report_dir / "qa_v2_summary.json"
    out_manifest = run_dir / "manifest.json"

    print("=" * 88)
    print("QA-V2 — QWEN3-VL READER ORACLE")
    print("=" * 88)
    print(f"Tệp dev       : {tep}")
    print(f"Số câu QA     : {len(cau)}")
    print(f"Model         : {args.model}")
    print(f"Prompt mode   : {args.prompt_mode}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Offline       : {not args.cho_phep_tai_model}")
    print(f"Run dir       : {run_dir}")
    print()

    reader = QwenReader(args.model, max_new_tokens=args.max_new_tokens)
    t0 = time.perf_counter()
    try:
        reader.nap()
    except Exception as e:
        print(f"LỖI nạp Qwen3-VL: {e}")
        print("Nếu model chưa có trong cache, chạy lại với --cho-phep-tai-model khi có mạng.")
        return 2
    model_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Qwen đã nạp: device={reader.device} | {model_load_ms:.0f} ms\n")

    # Import sau khi Qwen đã nạp. Cách này cũng giữ QA-V2 tách biệt với lỗi
    # pandas/native runtime từng gặp ở môi trường QA-V1 trên Windows.
    try:
        from aic2026.frame_map import FrameMap
        from aic2026.eval.answer_match import is_semantic_match
        from aic2026.qa_answer import loai_cau_hoi
    except Exception as e:
        print(f"LỖI import module dự án sau khi nạp Qwen: {e}")
        return 3

    records: list[dict[str, Any]] = []
    print(f"{'id':<6}{'frame':>8}{'n':>6}{'ms':>10}  {'sem':<4}  prediction")
    print("-" * 88)

    for q in cau:
        qid = str(q.get("id", ""))
        vid = str(q.get("video_id", ""))
        base = {
            "schema_version": SCHEMA_VERSION,
            "query_id": qid,
            "video_id": vid,
            "question_vi": q.get("cau_hoi", ""),
            "gt_answer": q.get("cau_tra_loi", ""),
            "gt_frame_start": int(q["frame_start"]),
            "gt_frame_end": int(q["frame_end"]),
            "gt_pts_time": float(q["pts_time"]),
            "model": args.model,
            "prompt_mode": args.prompt_mode,
            "max_new_tokens": int(args.max_new_tokens),
        }

        try:
            fm = FrameMap.load(vid)
        except Exception as e:
            records.append({**base, "status": "missing", "reason": "frame_map", "error": str(e)})
            print(f"{qid:<6}{'-':>8}{'-':>6}{'-':>10}  ----  MISSING frame_map")
            continue

        row = chon_frame_oracle(q, fm)
        if row is None:
            records.append({**base, "status": "missing", "reason": "gt_keyframe_in_range"})
            print(f"{qid:<6}{'-':>8}{'-':>6}{'-':>10}  ----  NO GT KEYFRAME")
            continue

        anh = tim_anh_theo_n(vid, int(row.n))
        if anh is None:
            records.append(
                {
                    **base,
                    "oracle_frame_idx": int(row.frame_idx),
                    "oracle_n": int(row.n),
                    "oracle_pts_time": float(row.pts_time),
                    "status": "missing",
                    "reason": "keyframe_image",
                }
            )
            print(f"{qid:<6}{int(row.frame_idx):>8}{int(row.n):>6}{'-':>10}  ----  MISSING IMAGE")
            continue

        prompt = tao_prompt(str(q["cau_hoi"]), args.prompt_mode)
        try:
            bat_dau = time.perf_counter()
            pred = reader.hoi(anh, prompt)
            latency_ms = (time.perf_counter() - bat_dau) * 1000.0
        except Exception as e:
            records.append(
                {
                    **base,
                    "oracle_frame_idx": int(row.frame_idx),
                    "oracle_n": int(row.n),
                    "oracle_pts_time": float(row.pts_time),
                    "image_path_rel": str(anh.relative_to(KEYFRAMES_DIR)),
                    "prompt": prompt,
                    "status": "error",
                    "reason": "qwen_inference",
                    "error": repr(e),
                }
            )
            print(f"{qid:<6}{int(row.frame_idx):>8}{int(row.n):>6}{'-':>10}  ERR   {type(e).__name__}")
            continue

        gt = str(q["cau_tra_loi"])
        ex = exact_match(gt, pred)
        try:
            sem = bool(is_semantic_match(gt, pred))
        except Exception as e:
            records.append(
                {
                    **base,
                    "oracle_frame_idx": int(row.frame_idx),
                    "oracle_n": int(row.n),
                    "oracle_pts_time": float(row.pts_time),
                    "image_path_rel": str(anh.relative_to(KEYFRAMES_DIR)),
                    "prompt": prompt,
                    "pred_answer": pred,
                    "exact_match": ex,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "reason": "semantic_match",
                    "error": repr(e),
                }
            )
            print(f"{qid:<6}{int(row.frame_idx):>8}{int(row.n):>6}{latency_ms:>10.0f}  ERR   semantic_match")
            continue

        records.append(
            {
                **base,
                "oracle_frame_idx": int(row.frame_idx),
                "oracle_n": int(row.n),
                "oracle_pts_time": float(row.pts_time),
                "image_path_rel": str(anh.relative_to(KEYFRAMES_DIR)),
                "prompt": prompt,
                "pred_answer": pred,
                "exact_match": ex,
                "semantic_match": sem,
                "latency_ms": latency_ms,
                "intent": loai_cau_hoi(str(q["cau_hoi"])),
                "status": "ok",
            }
        )
        print(
            f"{qid:<6}{int(row.frame_idx):>8}{int(row.n):>6}{latency_ms:>10.0f}  "
            f"{'OK' if sem else '--':<4}  {pred[:44]!r}"
        )

    summary = tong_hop(records)
    summary.update(
        {
            "schema_version": SCHEMA_VERSION,
            "task": "QA-V2",
            "measurement": "Qwen3-VL reader oracle on GT keyframe",
            "input_file": str(tep),
            "model": args.model,
            "prompt_mode": args.prompt_mode,
            "max_new_tokens": int(args.max_new_tokens),
            "model_load_ms": model_load_ms,
            "device": reader.device,
            "semantic_model": SEMANTIC_MODEL,
        }
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "QA-V2",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "input_file": str(tep),
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "max_new_tokens": int(args.max_new_tokens),
        "offline": not args.cho_phep_tai_model,
        "device": reader.device,
        "records": len(records),
        "ok": summary["ok"],
        "outputs": {"records": str(out_jsonl), "summary": str(out_summary)},
    }

    ghi_jsonl(out_jsonl, records)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ov = summary["overall"]
    print("\n" + "=" * 88)
    print("KẾT QUẢ QA-V2")
    print("=" * 88)
    print(f"OK                  : {summary['ok']}/{summary['total_records']}")
    print(f"Exact match         : {ov['exact_match']}/{ov['n']} ({ov['exact_rate']:.1%})")
    print(f"Semantic match      : {ov['semantic_match']}/{ov['n']} ({ov['semantic_rate']:.1%})")
    print(f"Latency TB          : {ov['avg_latency_ms']:.0f} ms/câu")
    print(f"Model load          : {model_load_ms:.0f} ms")
    if summary["non_ok_reasons"]:
        print(f"Không đo được       : {summary['non_ok_reasons']}")
    print(f"\nRecords             : {out_jsonl}")
    print(f"Summary             : {out_summary}")
    print(f"Manifest            : {out_manifest}")

    if summary["ok"] < 10:
        print("\nCẢNH BÁO: cỡ mẫu reader oracle < 10 câu; chỉ dùng để đánh giá feasibility/pipeline, chưa nên chốt model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
