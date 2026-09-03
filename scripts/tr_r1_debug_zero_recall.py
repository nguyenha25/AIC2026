"""
tr_r1_debug_zero_recall.py — Tìm nguyên nhân Covered=0.

Covered=0/50 TUYỆT ĐỐI trên cả 12 query gần như chắc chắn là lỗi hệ thống
(sai định dạng/đơn vị), KHÔNG PHẢI model CLIP-L yếu — model yếu vẫn trúng
vài câu do may rủi.

Script này đọc THẲNG file tr_r1_benchmark.json đã lưu (không chạy lại
CLIP-L, không cần torch/FAISS) và in ra 2 nhóm chẩn đoán:

  A. So khớp video_id: đếm bao nhiêu event có ít nhất 1 region TRÙNG
     video_id với GT (bất kể thời gian) — nếu con số này CŨNG = 0, gần như
     chắc chắn là lỗi ĐỊNH DẠNG video_id (vd "L23_V024" vs "l23_v024" vs
     "L23_V024.mp4"), không liên quan gì đến thời gian.

  B. Nếu video_id có khớp nhưng IoU vẫn = 0: in ra magnitude chênh lệch
     giữa region start/end và GT start/end — nếu chênh lệch có tỉ lệ cố
     định (vd ~25x, ~1000x), gần như chắc chắn là lỗi ĐƠN VỊ pts_time
     (giây vs mili-giây, hoặc giây vs frame_idx chưa chia fps).

Chạy:
    python scripts/tr_r1_debug_zero_recall.py
"""
from __future__ import annotations
import json
from aic2026.paths import RUNS_DIR


def temporal_iou(a_start, a_end, b_start, b_end) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def main() -> None:
    path = RUNS_DIR / "tr_r1_benchmark.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_events = 0
    events_with_video_match = 0
    events_with_iou_gt_0 = 0
    events_with_gt_frame_hit = 0
    singleton_regions = 0

    video_id_mismatch_samples = []
    time_gap_samples = []

    for q in data["queries"]:
        for ev in q["events"]:
            total_events += 1
            gt = ev["gt"]
            regions = ev["result"]["regions"]

            if not regions:
                continue

            singleton_regions += sum(
                1
                for region in regions
                if float(region["end_time"]) <= float(region["start_time"])
            )

            video_match = any(r["video_id"] == gt["video_id"] for r in regions)
            if video_match:
                events_with_video_match += 1
            elif len(video_id_mismatch_samples) < 5:
                video_id_mismatch_samples.append({
                    "query_id": q["query_id"],
                    "event_id": ev["event_id"],
                    "gt_video_id": repr(gt["video_id"]),
                    "top_region_video_ids": [repr(r["video_id"]) for r in regions[:3]],
                })

            best_iou = max(
                (temporal_iou(r["start_time"], r["end_time"], gt["start_time"], gt["end_time"])
                 for r in regions if r["video_id"] == gt["video_id"]),
                default=0.0,
            )
            if best_iou > 0:
                events_with_iou_gt_0 += 1
            elif video_match and len(time_gap_samples) < 5:
                same_video_regions = [r for r in regions if r["video_id"] == gt["video_id"]]
                closest = min(same_video_regions,
                              key=lambda r: abs(r["start_time"] - gt["start_time"]))
                time_gap_samples.append({
                    "query_id": q["query_id"],
                    "event_id": ev["event_id"],
                    "gt_start": gt["start_time"], "gt_end": gt["end_time"],
                    "region_start": closest["start_time"], "region_end": closest["end_time"],
                    "ratio_start": (closest["start_time"] / gt["start_time"])
                                   if gt["start_time"] else None,
                })

            frame_start = int(gt.get("frame_start", 1))
            frame_end = int(gt.get("frame_end", 0))
            if frame_start > frame_end:
                frame_start, frame_end = frame_end, frame_start

            has_gt_frame = any(
                r["video_id"] == gt["video_id"]
                and any(
                    hit.get("frame_idx") is not None
                    and frame_start <= int(hit["frame_idx"]) <= frame_end
                    for hit in r.get("hits", [])
                )
                for r in regions
            )
            if has_gt_frame:
                events_with_gt_frame_hit += 1

    print("=" * 72)
    print("TR-R1 — CHẨN ĐOÁN COVERED=0")
    print("=" * 72)
    print(f"Tổng số event                              : {total_events}")
    print(f"Event có ÍT NHẤT 1 region trùng video_id   : {events_with_video_match}")
    print(f"Event có IoU > 0 (bất kể ngưỡng)            : {events_with_iou_gt_0}")
    print(f"Event có candidate frame nằm trong GT        : {events_with_gt_frame_hit}")
    print(f"Region suy biến start_time == end_time        : {singleton_regions}")
    print()

    if events_with_video_match == 0:
        print("=> KẾT LUẬN: video_id KHÔNG BAO GIỜ khớp. Đây gần như chắc chắn")
        print("   là lỗi ĐỊNH DẠNG video_id giữa FAISS index và dev_questions.jsonl.")
        print("   Ví dụ lệch:")
        for s in video_id_mismatch_samples:
            print(f"   - query={s['query_id']} event={s['event_id']}: "
                  f"GT={s['gt_video_id']}  vs  top_regions={s['top_region_video_ids']}")
    elif events_with_iou_gt_0 == 0 and singleton_regions:
        print("=> Có region suy biến [t, t]. Temporal IoU của region này luôn bằng 0,")
        print("   kể cả t nằm trong GT. Chạy benchmark bằng bản retrieval mới có")
        print("   min_region_duration_seconds > 0 rồi mới đánh giá IoU.")
    elif events_with_iou_gt_0 == 0:
        print("=> KẾT LUẬN: video_id CÓ khớp nhưng thời gian không bao giờ giao nhau.")
        print("   Nhiều khả năng lỗi ĐƠN VỊ thời gian (giây vs ms, hoặc pts_time")
        print("   chưa được chia cho fps). Xem tỉ lệ start_time dưới đây — nếu")
        print("   ratio_start ~ hằng số (vd ~25, ~1000, ~30) thì đúng là lỗi đơn vị/fps:")
        for s in time_gap_samples:
            print(f"   - query={s['query_id']} event={s['event_id']}: "
                  f"GT=[{s['gt_start']:.3f},{s['gt_end']:.3f}]  "
                  f"region_gần_nhất=[{s['region_start']:.3f},{s['region_end']:.3f}]  "
                  f"ratio_start={s['ratio_start']}")
    else:
        print("=> video_id và thời gian đều CÓ giao nhau ở một số event, nhưng")
        print("   IoU >= 0.30 (ngưỡng acceptance nội bộ) vẫn không đạt đủ.")
        print("   Đây có thể là vấn đề chất lượng model/config thật (region quá ngắn/dài,")
        print("   max_region_duration_seconds/region_merge_gap_seconds chưa hợp lý),")
        print("   không phải bug định dạng.")


if __name__ == "__main__":
    main()
