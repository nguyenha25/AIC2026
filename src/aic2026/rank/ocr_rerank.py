"""Module OCR Retrieval & Reranking cho Task 10."""
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from src.aic2026.paths import DERIVED_DIR

def remove_accents(input_str: str) -> str:
    if not input_str:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()

def extract_ocr_keywords(query: str) -> list[str]:
    keywords = []
    # Trích xuất từ trong ngoặc kép
    quoted = re.findall(r'["\'](.*?)["\']', query)
    keywords.extend(quoted)
    
    # Trích xuất từ viết hoa (nghi ngờ là tên riêng/biển hiệu)
    uppers = re.findall(r"\b[A-Z0-9\-]{3,}\b", query)
    keywords.extend(uppers)
    
    # Trích xuất theo pattern ngữ cảnh chữ
    patterns = [
        r"(?:chữ|bảng ghi|mang tên|tên là|hiệu|thương hiệu|bảng tên|in chữ|tên|băng rôn|chào mừng)\s+([A-ZÀ-Ỹa-zà-ỹ0-9\s\-]+?)(?:,|\.|\bđặt\b|\bđứng\b|\bphía\b|\bđang\b|$)",
    ]
    for p in patterns:
        for m in re.findall(p, query, flags=re.IGNORECASE):
            if len(m.strip()) > 2:
                keywords.append(m.strip())

    # Loại bỏ stop words và chuẩn hóa
    stop_words = {"trong", "video", "tren", "duoi", "hinh", "anh", "cua", "mot", "cac", "nhung"}
    norm_keywords = set()
    for kw in keywords:
        norm = remove_accents(kw).strip()
        if len(norm) >= 3 and norm not in stop_words:
            norm_keywords.add(norm)
    return list(norm_keywords)

class OCRReranker:
    def __init__(self, ocr_dir: Path | None = None):
        self.ocr_dir = ocr_dir or (DERIVED_DIR / "ocr")
        self._index: list[dict] = []
        self._build_in_memory_index()

    def _build_in_memory_index(self):
        if not self.ocr_dir.exists():
            return
        for file_path in self.ocr_dir.glob("*.jsonl"):
            v_id = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    raw_text = item.get("text", "")
                    if raw_text:
                        self._index.append({
                            "video_id": v_id,
                            "frame_idx": item.get("frame_idx", 0),
                            "norm_text": remove_accents(raw_text),
                        })

    def search_ocr(self, query: str, top_k: int = 100) -> list[dict]:
        keywords = extract_ocr_keywords(query)
        if not keywords or not self._index:
            return []

        matched_frames = []
        for doc in self._index:
            text = doc["norm_text"]
            match_count = sum(1 for kw in keywords if kw in text)
            if match_count > 0:
                v_id = doc["video_id"]
                f_idx = doc["frame_idx"]
                score = match_count * 2.0
                
                # CHỈ LƯU ĐÚNG FRAME CHỨA CHỮ (Đã xóa vòng lặp sinh frame rác delta)
                matched_frames.append({"video_id": v_id, "frame_idx": f_idx, "score": score})

        matched_frames.sort(key=lambda x: x["score"], reverse=True)
        return matched_frames[:top_k]

    # (Lưu ý: Nếu trong code cũ bạn còn các hàm khác như def rerank(...) ở bên dưới, 
    # hãy giữ nguyên chúng nhé, chỉ cần đảm bảo hàm search_ocr() ở trên giống y hệt là được).