import json
from pathlib import Path
from src.aic2026.semantic.parser import RuleBasedParser

def main():
    dev_path = Path("D:/aic-data/dev/dev_questions.jsonl")
    output_path = Path("D:/aic-data/runs/n01_semantic_parsing.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not dev_path.exists():
        print(f"Lỗi: Không tìm thấy tệp {dev_path}")
        return

    parser = RuleBasedParser()
    parsed_results = []
    
    with open(dev_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            query_id = data.get("id")
            loai = data.get("loai_truy_van")
            
            if loai == "hoi_dap":
                query_text = data.get("cau_hoi", "")
                plan = parser.parse_qa(query_id=query_id, query_text=query_text)
                parsed_results.append(plan.model_dump())
                
            elif loai == "chuoi_su_kien":
                events = [gd.get("su_kien", "") for gd in data.get("cac_giai_doan", [])]
                plan = parser.parse_trake(query_id=query_id, events=events)
                parsed_results.append(plan.model_dump())
    
    with open(output_path, "w", encoding="utf-8") as f:
        for res in parsed_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f"======================================================")
    print(f"N-01 — ĐÃ PHÂN LOẠI CÂU HỎI Q&A VÀ TRAKE")
    print(f"Tổng số câu đã xử lý: {len(parsed_results)}")
    print(f"File kết quả được lưu tại: {output_path}")
    print(f"======================================================")
    print(f"Vui lòng mở file trên, chọn ra ít nhất 30 câu để kiểm tra")
    print(f"tay (đảm bảo độ chính xác intent/modality >= 90%).")

if __name__ == "__main__":
    main()