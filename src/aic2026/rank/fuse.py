"""
Gộp nhiều bảng xếp hạng bằng Reciprocal Rank Fusion — Task 10.

VÌ SAO LÀ TỆP RIÊNG, KHÔNG NỐI VÀO search.py
--------------------------------------------
`rank/search.py` là mạch đầu-cuối của Ngân (Việc 4). Task 10 chỉ cần cắm vào
tham số `tim_ung_vien` mà Ngân đã chừa sẵn ở `run_query()`, nên KHÔNG có lý do
gì phải sửa tệp của Ngân. Đặt riêng ở đây thì hai người làm song song không
đụng nhau, và việc gộp có thể thay/tắt mà mạch chính không phải đổi một dòng.

(Trước đây bản Task 10 ghi đè `search.py`, làm mất ~340 dòng mạch đầu-cuối của
Ngân — gồm cả `_gom_trake` khiến TRAKE sai định dạng và luôn 0 điểm. Tách tệp
là để chuyện đó không lặp lại.)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


def reciprocal_rank_fusion(
    modality_results: Mapping[str, Sequence[dict[str, Any]]],
    weights: Mapping[str, float] | None = None,
    k_rrf: int = 60,
) -> list[dict[str, Any]]:
    """
    RRF Score(d) = sum_{m in M} (w_m / (k_rrf + rank_m(d))).

    Mỗi phần tử vào cần ít nhất `video_id` và `frame_idx`. Phần tử ra có thêm
    `ranks` — hạng của nó trong từng nhánh, dùng để biết frame nào lên nhờ nguồn nào.

    KHỬ TRÙNG KHOÁ TRONG TỪNG NHÁNH
    -------------------------------
    Cộng thẳng `rrf_scores[key] += ...` là sai khi MỘT nhánh trả cùng
    (video_id, frame_idx) nhiều lần — chuyện có thật trong bộ dữ liệu này:
    các keyframe kề nhau làm tròn về cùng frame_idx, ~1.228 dòng trên 192 video.
    Frame đó được cộng điểm nhiều lần cho cùng một nhánh rồi nhảy lên trên frame
    đơn có hạng cao hơn, tức RRF tự đảo thứ tự nhánh hình ảnh dù nhánh chữ không
    đóng góp gì. Đo trên bộ dev 32 câu: 6 câu bị đảo.

    Công thức RRF chuẩn tính MỘT hạng cho mỗi tài liệu trong mỗi nhánh, nên ở đây
    khử trùng trước, giữ hạng nhỏ nhất, rồi mới cộng.
    """
    weights = weights or {}
    rrf_scores: dict[tuple[str, int], float] = defaultdict(float)
    provenance: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)

    for mod_name, results in modality_results.items():
        w = float(weights.get(mod_name, 1.0))

        hang_tot_nhat: dict[tuple[str, int], int] = {}
        for rank_idx, item in enumerate(results, start=1):
            key = (str(item["video_id"]), int(item["frame_idx"]))
            if key not in hang_tot_nhat:
                hang_tot_nhat[key] = rank_idx

        for key, rank_idx in hang_tot_nhat.items():
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
