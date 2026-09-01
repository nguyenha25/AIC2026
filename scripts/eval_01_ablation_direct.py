import json
import statistics
import argparse
import time
from aic2026.semantic.parser import RuleBasedParser
from aic2026.rank.hop_nhat import dict_sang_hit, gop_nguon, no_khoang_asr, tim_ung_vien_gop, NGUON_OCR_FTS
from aic2026.index.fts_index import TextSearchIndex
from aic2026.rank.search import tim_ung_vien_clip

_parser = RuleBasedParser()
kho_chu = TextSearchIndex()

# --- HOOK A0_baseline ---
hook_a0_baseline = tim_ung_vien_gop(dung_clip=True, dung_ocr=False, dung_asr=False)

# --- HOOK A3_routing ---
def hook_a3_routing(cau_hoi: str, so_ung_vien: int) -> list:
    plan = _parser.parse_qa("eval_01", cau_hoi)
    
    if getattr(plan, "task", "") == "trake" or plan.__class__.__name__ == "QueryPlanTRAKE":
        return tim_ung_vien_clip(cau_hoi, so_ung_vien)
        
    k_search = getattr(plan, "semantic_k_hint", so_ung_vien)
    
    cac_nguon = {
        "clip": list(tim_ung_vien_clip(cau_hoi, k_search))
    }
    
    preferred_mods = getattr(plan, "preferred_modalities", {})
    active_modalities = {
        "clip": preferred_mods.get("clip_l", 1.0)
    }
    
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

    return gop_nguon(cac_nguon, active_modalities, k_rrf=60)

def evaluate_ablation(dev_file_path: str):
    queries = []
    gt_mapping = {}
    with open(dev_file_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("loai_truy_van") == "chuoi_su_kien":
                continue
            qid = str(data.get("id"))
            queries.append({"query_id": qid, "text": data.get("cau_hoi")})
            gt_mapping[qid] = data.get("video_id") or data.get("dap_an")

    hooks = {"A0_baseline": hook_a0_baseline, "A3_routing": hook_a3_routing}
    K_LEVELS = [50, 100, 300, 500]
    per_query_hits = {}

    print(f"=== EVAL-01 (QA ONLY: n={len(queries)}) ===")
    print(f"{'Stage':<15} | {'R@50':<7} | {'R@100':<7} | {'R@300':<7} | {'R@500':<7} | {'p50 (ms)':<8} | {'Avg Pool':<8}")
    print("-" * 75)

    for stage_name, hook in hooks.items():
        recalls = {k: 0 for k in K_LEVELS}
        latencies = []
        pool_sizes = []
        per_query_hits[stage_name] = {}

        for q in queries:
            target = gt_mapping.get(q["query_id"])
            if not target:
                continue

            t0 = time.perf_counter()
            hits = hook(q["text"], 500)          
            latencies.append((time.perf_counter() - t0) * 1000)

            retrieved = []
            for h in hits:
                if h.video_id not in retrieved:
                    retrieved.append(h.video_id)
            
            pool_sizes.append(len(retrieved))

            hit_100 = target in retrieved[:100]
            per_query_hits[stage_name][q["query_id"]] = hit_100
            for k in K_LEVELS:
                if target in retrieved[:k]:
                    recalls[k] += 1

        n = len(queries)
        p50 = statistics.median(latencies) if latencies else 0
        avg_pool = sum(pool_sizes) / len(pool_sizes) if pool_sizes else 0
        
        print(f"{stage_name:<15} | {recalls[50]/n:.4f}  | {recalls[100]/n:.4f}  | "
              f"{recalls[300]/n:.4f}  | {recalls[500]/n:.4f}  | {p50:<8.1f} | {avg_pool:.1f}")

    print("\n[KIỂM TRA THỦ CÔNG] Các ID câu hỏi A3 cứu thành công (A0 trượt):")
    a0_hits = per_query_hits.get("A0_baseline", {})
    a3_hits = per_query_hits.get("A3_routing", {})
    
    flipped = [qid for qid, hit in a3_hits.items() if hit and not a0_hits.get(qid, False)]
    if not flipped:
        print(" - Không có câu nào lật kèo.")
    else:
        for qid in flipped:
            print(f" - ID: {qid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-file", type=str, default=r"D:\aic-data\dev\dev_questions.jsonl", help="Đường dẫn tới file dev jsonl")
    args = parser.parse_args()
    evaluate_ablation(args.dev_file)