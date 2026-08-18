"""Hệ thống Reciprocal Rank Fusion kết hợp Visual và Text."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from aic2026.frame_map import FrameMap
from aic2026.rank.config import RankConfig
from aic2026.rank.dedupe import deduplicate_temporal


def reciprocal_rank_fusion(
    modality_results: Mapping[str, Sequence[dict[str, Any]]],
    weights: Mapping[str, float] | None = None,
    k_rrf: int = 60,
) -> list[dict[str, Any]]:
    """RRF Score(d) = sum_{m in M} (w_m / (k_rrf + rank_m(d)))."""
    weights = weights or {}
    rrf_scores: dict[tuple[str, int], float] = defaultdict(float)
    provenance: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)

    for mod_name, results in modality_results.items():
        w = float(weights.get(mod_name, 1.0))
        for rank_idx, item in enumerate(results, start=1):
            key = (str(item["video_id"]), int(item["frame_idx"]))
            rrf_scores[key] += w / (k_rrf + rank_idx)
            provenance[key][mod_name] = rank_idx

    fused: list[dict[str, Any]] = []
    for (vid, fidx), score in rrf_scores.items():
        fused.append(
            {
                "video_id": vid,
                "frame_idx": fidx,
                "score": float(score),
                "ranks": provenance[(vid, fidx)],
            }
        )

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


class MultimediaSearchEngine:
    def __init__(
        self,
        config: RankConfig | None = None,
        frame_map: FrameMap | None = None,
        text_index: Any = None,
        faiss_index: Any = None,
        clip_encoder: Any = None,
    ) -> None:
        self.config = config or RankConfig()
        self.frame_map = frame_map
        self.text_index = text_index
        self.faiss_index = faiss_index
        self.clip_encoder = clip_encoder

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        limit = top_k or self.config.top_k_candidates
        modality_candidates: dict[str, list[dict[str, Any]]] = {}

        # 1. Visual FAISS (Task 2 / Task 12)
        if self.faiss_index is not None and self.clip_encoder is not None:
            q_emb = self.clip_encoder.encode_text(query)
            modality_candidates["visual"] = self.faiss_index.search(q_emb, top_k=limit * 2)

        # 2. Text (Hỗ trợ linh hoạt cả FTSIndex và OCRReranker)
        if self.text_index is not None:
            if hasattr(self.text_index, "search_ocr"):
                modality_candidates["ocr"] = self.text_index.search_ocr(query, top_k=limit * 2)
            elif hasattr(self.text_index, "search_text"):
                modality_candidates["ocr"] = self.text_index.search_text(query, top_k=limit * 2)

        # 3. RRF Fusion
        # VÔ HIỆU HÓA ĐỘC TÀI w=1000.0 TỪ RANKCONFIG
        # Ép trọng số ở mức dân chủ: OCR chỉ nhỉnh hơn 1 chút (1.2) để break-tie
        safe_w_visual = 1.0
        safe_w_ocr = 0.45 
        safe_w_asr = 1.0

        weights = {
            "visual": safe_w_visual,
            "ocr": safe_w_ocr,
            "asr": safe_w_asr,
        }
        
        fused = reciprocal_rank_fusion(
            modality_results=modality_candidates,
            weights=weights,
            k_rrf=self.config.k_rrf,
        )

        # 4. Deduplicate
        return deduplicate_temporal(
            ranked_items=fused,
            frame_map=self.frame_map,
            window_seconds=self.config.dedupe_window_seconds,
            limit=limit,
        )