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
    # 1. Bắt chuỗi trong ngoặc kép
    quoted = re.findall(r'["\'](.*?)["\']', query)
    keywords.extend(quoted)

    # 2. Bắt cụm viết hoa / số năm (1655-1735, BURBERRY...)
    uppers = re.findall(r"\b[A-Z0-9\-]{3,}\b", query)
    keywords.extend(uppers)

    # 3. Bắt các cụm danh từ sau từ chỉ dẫn
    patterns = [
        r"(?:chữ|bảng ghi|mang tên|tên là|hiệu|thương hiệu|bảng tên|in chữ|tên|băng rôn|chào mừng)\s+([A-ZÀ-Ỹa-zà-ỹ0-9\s\-]+?)(?:,|\.|\bđặt\b|\bđứng\b|\bphía\b|\bđang\b|$)",
    ]
    for p in patterns:
        for m in re.findall(p, query, flags=re.IGNORECASE):
            if len(m.strip()) > 2:
                keywords.append(m.strip())

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
        matched_videos = defaultdict(float)

        for doc in self._index:
            text = doc["norm_text"]
            match_count = sum(1 for kw in keywords if kw in text)
            if match_count > 0:
                v_id = doc["video_id"]
                f_idx = doc["frame_idx"]
                score = match_count * 2.0
                matched_frames.append({
                    "video_id": v_id,
                    "frame_idx": f_idx,
                    "score": score,
                })
                matched_videos[v_id] = max(matched_videos[v_id], score)

                # Mở rộng các frame lân cận trong bán kính +- 1000 frames (bước nhảy 50 frames)
                for delta in [-600, -400, -200, -100, -50, 50, 100, 200, 400, 600]:
                    nearby_idx = max(0, f_idx + delta)
                    matched_frames.append({
                        "video_id": v_id,
                        "frame_idx": nearby_idx,
                        "score": score * 0.7,
                    })

        matched_frames.sort(key=lambda x: x["score"], reverse=True)
        return matched_frames[:top_k]

    def rerank(self, candidates: list[dict], query: str, alpha: float = 0.6) -> list[dict]:
        ocr_hits = self.search_ocr(query, top_k=100)
        if not ocr_hits:
            return candidates

        seen = set()
        final_list = []
        for hit in ocr_hits:
            key = (hit["video_id"], hit["frame_idx"])
            if key not in seen:
                seen.add(key)
                final_list.append(hit)

        for cand in candidates:
            key = (cand["video_id"], cand["frame_idx"])
            if key not in seen:
                seen.add(key)
                final_list.append(cand)

        return final_list