import json
import statistics
import argparse
import time
import os
import datetime
import platform
import subprocess
import hashlib
from collections import defaultdict, Counter
from aic2026.semantic.parser import RuleBasedParser
from aic2026.rank.hop_nhat import dict_sang_hit, gop_nguon, no_khoang_asr, tim_ung_vien_gop, NGUON_OCR_FTS
from aic2026.index.fts_index import TextSearchIndex
from aic2026.rank.search import tim_ung_vien_clip

_parser = RuleBasedParser()
kho_chu = TextSearchIndex()

a3_routing_stats = {"clip": 0, "ocr": 0, "asr": 0, "ocr_zero_hits": 0, "asr_zero_hits": 0}

def _is_trake(plan) -> bool:
    return getattr(plan, "task", "") == "trake" or plan.__class__.__name__ == "QueryPlanTRAKE"

hook_a0_baseline = tim_ung_vien_gop(dung_clip=True, dung_ocr=False, dung_asr=False)
hook_a1_parser = tim_ung_vien_gop(dung_clip=True, dung_ocr=False, dung_asr=False)

def hook_a2_expansion(cau_hoi: str, so_ung_vien: int) -> list:
    plan = _parser.parse_qa("eval_01", cau_hoi)
    if _is_trake(plan):
        return hook_a0_baseline(cau_hoi, so_ung_vien)

    k_search = getattr(plan, "semantic_k_hint", so_ung_vien)
    variants = getattr(plan, "queries", {}).get("clip_l", [cau_hoi]) or [cau_hoi]

    cac_nguon = {}
    trong_so = {}
    for i, v in enumerate(variants):
        ten = f"clip_v{i}"
        cac_nguon[ten] = list(tim_ung_vien_clip(v, k_search))
        trong_so[ten] = 1.0

    return gop_nguon(cac_nguon, trong_so, k_rrf=60)

def hook_a3_routing(cau_hoi: str, so_ung_vien: int) -> list:
    global a3_routing_stats
    
    plan = _parser.parse_qa("eval_01", cau_hoi)
    if _is_trake(plan):
        a3_routing_stats["clip"] += 1
        return hook_a0_baseline(cau_hoi, so_ung_vien)

    k_search = getattr(plan, "semantic_k_hint", so_ung_vien)
    variants = getattr(plan, "queries", {}).get("clip_l", [cau_hoi]) or [cau_hoi]

    cac_nguon = {}
    active_modalities = {}

    a3_routing_stats["clip"] += 1
    for i, v in enumerate(variants):
        ten = f"clip_v{i}"
        cac_nguon[ten] = list(tim_ung_vien_clip(v, k_search))
        active_modalities[ten] = 1.0
    
    preferred_mods = getattr(plan, "preferred_modalities", {})
    plan_queries = getattr(plan, "queries", {})

    for modality, weight in preferred_mods.items():
        if weight <= 0 or modality not in ["ocr", "asr"]:
            continue
            
        hits_nhanh = []
        for sub_query in plan_queries.get(modality, []):
            if modality == "ocr":
                hits_nhanh.extend([dict_sang_hit(d, NGUON_OCR_FTS) for d in kho_chu.search_text(sub_query, top_k=k_search) if d])
            elif modality == "asr":
                hits_nhanh.extend(no_khoang_asr(kho_chu.search_asr(sub_query, top_k=k_search)))
                
        if hits_nhanh:
            cac_nguon[modality] = [h for h in hits_nhanh if h is not None]
            active_modalities[modality] = weight
            a3_routing_stats[modality] += 1
        else:
            a3_routing_stats[f"{modality}_zero_hits"] += 1

    return gop_nguon(cac_nguon, active_modalities, k_rrf=60)

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def evaluate_ablation(dev_file_path: str):
    queries = []
    gt_mapping = {}
    intent_by_qid = {}

    with open(dev_file_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("loai_truy_van") == "chuoi_su_kien":
                continue
            qid = str(data.get("id"))
            text = data.get("cau_hoi")
            queries.append({"query_id": qid, "text": text})
            gt_mapping[qid] = data.get("video_id") or data.get("dap_an")

            plan = _parser.parse_qa(qid, text)
            intent_by_qid[qid] = getattr(plan, "intent", "unknown")

    if not queries:
        return

    qid_counts = Counter(q["query_id"] for q in queries)
    duplicates = {qid: n for qid, n in qid_counts.items() if n > 1}
    if duplicates:
        raise ValueError(f"Duplicate query_id detected: {duplicates}")

    hooks = {
        "A0_baseline": hook_a0_baseline,
        "A1_parser": hook_a1_parser,
        "A2_expansion": hook_a2_expansion,
        "A3_routing": hook_a3_routing,
    }
    K_LEVELS = [50, 100, 300, 500]

    results_hit_count = {stage: {k: 0 for k in K_LEVELS} for stage in hooks}
    per_intent_hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    retrieval_history = {stage: {} for stage in hooks}

    print(f"=== EVAL-01: ABLATION & INCREMENTAL CONTRIBUTION (n={len(queries)}) ===\n")
    print(f"{'Stage':<15} | {'R@50':<7} | {'R@100':<7} | {'R@300':<7} | {'R@500':<7} | {'MRR':<7} | {'p50(ms)':<7} | {'p95(ms)':<7}")
    print("-" * 90)

    for stage_name, hook in hooks.items():
        latencies = []
        mrr_sum = 0

        for q in queries:
            qid = q["query_id"]
            target = gt_mapping.get(qid)
            if not target: continue

            t0 = time.perf_counter()
            hits = hook(q["text"], 500)
            latencies.append((time.perf_counter() - t0) * 1000)

            retrieved = []
            for h in hits:
                if h.video_id not in retrieved:
                    retrieved.append(h.video_id)
            
            try:
                gt_rank = retrieved.index(target) + 1
                mrr_sum += 1.0 / gt_rank
            except ValueError:
                gt_rank = None

            retrieval_history[stage_name][qid] = {
                "rank_gt": gt_rank,
                "retrieved": retrieved
            }
            
            hit_100 = (gt_rank is not None) and (gt_rank <= 100)
            intent = intent_by_qid.get(qid, "unknown")
            per_intent_hits[stage_name][intent][1] += 1
            if hit_100:
                per_intent_hits[stage_name][intent][0] += 1

            for k in K_LEVELS:
                if (gt_rank is not None) and (gt_rank <= k):
                    results_hit_count[stage_name][k] += 1

        n = len(queries)
        
        if len(latencies) >= 2:
            quantiles = statistics.quantiles(latencies, n=100, method="inclusive")
            p50 = quantiles[49]
            p95 = quantiles[94]
        else:
            p50 = p95 = latencies[0] if latencies else 0
            
        mrr = mrr_sum / n

        print(f"{stage_name:<15} | {results_hit_count[stage_name][50]/n:.4f}  | {results_hit_count[stage_name][100]/n:.4f}  | "
              f"{results_hit_count[stage_name][300]/n:.4f}  | {results_hit_count[stage_name][500]/n:.4f}  | {mrr:.4f}  | {p50:<7.1f} | {p95:<7.1f}")

    # --- Breakdown Intent ---
    print("\n=== BREAKDOWN INTENT (R@100) ===")
    all_intents = sorted({i for stage_map in per_intent_hits.values() for i in stage_map})
    header = f"{'Intent':<25} | " + " | ".join(f"{s:<12}" for s in hooks)
    print(header)
    print("-" * len(header))
    for intent in all_intents:
        row = [f"{intent:<25}"]
        for stage_name in hooks:
            hits, total = per_intent_hits[stage_name].get(intent, [0, 0])
            cell = f"{hits}/{total}" if total else "n/a"
            row.append(f"{cell:<12}")
        print(" | ".join(row))

    # --- A3 Routing Statistics ---
    print("\n=== A3 ROUTING STATISTICS ===")
    print(f"- CLIP activated: {a3_routing_stats['clip']}/{len(queries)}")
    print(f"- OCR successfully retrieved hits: {a3_routing_stats['ocr']} queries")
    print(f"- OCR triggered but 0 hits: {a3_routing_stats['ocr_zero_hits']} queries")
    print(f"- ASR successfully retrieved hits: {a3_routing_stats['asr']} queries")
    print(f"- ASR triggered but 0 hits: {a3_routing_stats['asr_zero_hits']} queries")

    # --- Diagnostic ---
    print("\n=== DIAGNOSTIC: A2 vs A3 (R@100) ===")
    a2_hits = {qid for qid, data in retrieval_history["A2_expansion"].items() if data["rank_gt"] and data["rank_gt"] <= 100}
    a3_hits = {qid for qid, data in retrieval_history["A3_routing"].items() if data["rank_gt"] and data["rank_gt"] <= 100}
    
    lost_to_a3 = a2_hits - a3_hits
    gained_by_a3 = a3_hits - a2_hits
    
    assert len(lost_to_a3) - len(gained_by_a3) == results_hit_count["A2_expansion"][100] - results_hit_count["A3_routing"][100], "Hit count mismatch in diagnostics!"

    rank_shifts = 0
    for q in queries:
        qid = q["query_id"]
        r2 = retrieval_history["A2_expansion"][qid]["rank_gt"]
        r3 = retrieval_history["A3_routing"][qid]["rank_gt"]
        if r2 != r3:
            rank_shifts += 1
    print(f"- Tổng số câu bị thay đổi thứ hạng GT (Rank Shifts): {rank_shifts}/{len(queries)}")

    # --- Save Enhanced Manifest with Full Contract (P1) ---
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/eval_01_{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    
    config_payload = {
        "stages": list(hooks.keys()),
        "A0_baseline": "CLIP-only via tim_ung_vien_gop",
        "A1_parser": "CLIP-only + Parser intent metadata",
        "A2_expansion": "CLIP + Query Expansion",
        "A3_routing": "A2 Expansion + OCR/ASR Routing",
        "k_search": 500,
        "k_rrf": 60
    }
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "task": "EVAL-01_Ablation",
        "git_commit": get_git_commit(),
        "dataset_version": os.path.basename(dev_file_path),
        "split": "dev_holdout (QA only)",
        "n_queries": len(queries),
        "query_ids": [q["query_id"] for q in queries],
        "config_hash": config_hash,
        "model_id": "ViT-L/14", # Cập nhật chuẩn theo Handbook
        "precision": "fp16",    # Hoặc fp32 tùy vào model thực tế bạn đang chạy
        "prompt_version": "v1.0-rule-parser",
        "seed": None,
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version()
        },
        "config": config_payload,
        "missing_assets": [],
        "outputs": {
            "report": "REPORT_EVAL_01.md",
            "diagnostic": f"{run_dir}/diagnostic.json"
        }
    }
    
    with open(os.path.join(run_dir, "diagnostic.json"), "w", encoding="utf-8") as f:
        json.dump(retrieval_history, f, indent=4, ensure_ascii=False)
        
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    print(f"\n[INFO] Đã lưu manifest chuẩn contract tại: {run_dir}/manifest.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-file", type=str, default=r"D:\aic-data\dev\dev_questions.jsonl")
    args = parser.parse_args()
    evaluate_ablation(args.dev_file)