"""
tr_r1_acceptance.py — Chấm điểm PASS/FAIL cho TR-R1.

Tiêu chí acceptance nội bộ của task:
    "TR-R1 PASS khi Recall@Region@10 >= 0.75 trên ít nhất 10/12 query TRAKE
    sạch, với hit = đúng video và temporal IoU >= 0.30."

Ba điểm khác biệt so với benchmark cũ (tr_r1_benchmark.py hiện tại):
    1. Hit = IoU >= 0.30 (không phải "có giao nhau" là đủ).
    2. Chỉ xét top-10 region mỗi event (Recall@Region@10), không phải
       max_regions_per_event=5 như default hiện tại của TRR1Config.
    3. Recall tính THEO TỪNG QUERY rồi đếm số query đạt ngưỡng — không phải
       gộp covered_events/total_events trên toàn bộ tập.

Đây không phải công thức chấm chính thức của cuộc thi. Theo tài liệu AIC,
TRAKE được chấm bằng ``video_id`` và từng ``frame_id`` nằm trong đoạn GT;
IoU region chỉ là proxy để đánh giá tầng coarse retrieval.

Module này KHÔNG đụng vào phần retrieval/CLIP-L — chỉ nhận input là
TRR1Result (hoặc dict tương đương) + GT, để có thể unit test độc lập,
không cần torch/FAISS/model thật.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path


RECALL_TARGET = 0.75
IOU_THRESHOLD = 0.30
REGION_TOP_K = 10
MIN_PASSING_QUERIES = 10
EXPECTED_TOTAL_QUERIES = 12


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """
    IoU của 2 khoảng thời gian [a_start,a_end] và [b_start,b_end].
    Trả 0.0 nếu không giao nhau hoặc một trong hai khoảng có độ dài <= 0
    (tránh chia 0 khi region/GT suy biến thành 1 điểm thời gian).
    """
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    intersection = max(0.0, inter_end - inter_start)

    union_start = min(a_start, b_start)
    union_end = max(a_end, b_end)
    union = union_end - union_start

    if union <= 0:
        return 0.0
    return intersection / union


def region_matches_gt(region_video_id: str, region_start: float, region_end: float,
                       gt_video_id: str, gt_start: float, gt_end: float,
                       iou_threshold: float = IOU_THRESHOLD) -> bool:
    """Hit = cùng video AND IoU >= ngưỡng. Đây là định nghĩa MỚI theo chốt của team."""
    if region_video_id != gt_video_id:
        return False
    return temporal_iou(region_start, region_end, gt_start, gt_end) >= iou_threshold


def event_is_covered_at_k(regions: list[dict], gt_video_id: str, gt_start: float,
                           gt_end: float, region_top_k: int = REGION_TOP_K,
                           iou_threshold: float = IOU_THRESHOLD) -> bool:
    """
    Event được coi là "covered" nếu MỘT TRONG top-`region_top_k` region (theo
    thứ tự đã sort giảm dần theo score — giả định input đã sort sẵn, đúng
    như _gom_vung() đã làm) khớp GT theo region_matches_gt().
    """
    top_regions = regions[:region_top_k]
    return any(
        region_matches_gt(r["video_id"], r["start_time"], r["end_time"],
                           gt_video_id, gt_start, gt_end, iou_threshold)
        for r in top_regions
    )


@dataclass
class QueryRecall:
    query_id: str
    total_events: int
    covered_events: int
    candidate_frame_covered_events: int = 0

    @property
    def recall(self) -> float:
        if self.total_events == 0:
            return 0.0
        return self.covered_events / self.total_events

    @property
    def passes_target(self) -> bool:
        return self.recall >= RECALL_TARGET

    @property
    def candidate_frame_recall(self) -> float:
        if self.total_events == 0:
            return 0.0
        return self.candidate_frame_covered_events / self.total_events


def event_has_gt_frame(
    regions: list[dict],
    gt: dict,
    region_top_k: int = REGION_TOP_K,
) -> bool:
    """Proxy sát luật thi: có frame provenance nằm trong đoạn GT hay không."""

    if "frame_start" not in gt or "frame_end" not in gt:
        return False

    frame_start = int(gt["frame_start"])
    frame_end = int(gt["frame_end"])
    if frame_start > frame_end:
        frame_start, frame_end = frame_end, frame_start

    for region in regions[:region_top_k]:
        if str(region.get("video_id", "")) != str(gt["video_id"]):
            continue
        for hit in region.get("hits", []):
            frame_idx = hit.get("frame_idx")
            if frame_idx is None:
                continue
            if frame_start <= int(frame_idx) <= frame_end:
                return True

    return False


def compute_query_recall(query_id: str, events: list[dict],
                          region_top_k: int = REGION_TOP_K,
                          iou_threshold: float = IOU_THRESHOLD) -> QueryRecall:
    """
    events: list các dict, mỗi dict có "regions" (list region dict đã sort
    theo score giảm dần) và "gt" (dict có video_id/start_time/end_time).
    """
    total = len(events)
    covered = sum(
        1 for e in events
        if event_is_covered_at_k(
            e["regions"], e["gt"]["video_id"], e["gt"]["start_time"], e["gt"]["end_time"],
            region_top_k, iou_threshold,
        )
    )
    frame_covered = sum(
        1
        for event in events
        if event_has_gt_frame(
            event["regions"],
            event["gt"],
            region_top_k,
        )
    )
    return QueryRecall(
        query_id=query_id,
        total_events=total,
        covered_events=covered,
        candidate_frame_covered_events=frame_covered,
    )


@dataclass
class AcceptanceResult:
    total_queries: int
    passing_queries: int
    min_passing_required: int
    overall_pass: bool
    per_query: list[QueryRecall]

    def to_dict(self) -> dict:
        return {
            "recall_target": RECALL_TARGET,
            "iou_threshold": IOU_THRESHOLD,
            "region_top_k": REGION_TOP_K,
            "total_queries": self.total_queries,
            "min_passing_required": self.min_passing_required,
            "passing_queries": self.passing_queries,
            "overall_pass": self.overall_pass,
            "per_query": [
                {"query_id": q.query_id, "total_events": q.total_events,
                 "covered_events": q.covered_events, "recall": round(q.recall, 4),
                 "candidate_frame_covered_events": q.candidate_frame_covered_events,
                 "candidate_frame_recall": round(q.candidate_frame_recall, 4),
                 "passes_target": q.passes_target}
                for q in self.per_query
            ],
        }


def compute_acceptance(per_query_events: dict[str, list[dict]],
                        expected_total_queries: int = EXPECTED_TOTAL_QUERIES,
                        min_passing_queries: int = MIN_PASSING_QUERIES) -> AcceptanceResult:
    """
    per_query_events: {query_id: [event_dict, ...]} — CHỈ chứa các query
    "TRAKE sạch" đã lọc sẵn (việc lọc "sạch" thuộc trách nhiệm nơi gọi hàm
    này, vì phụ thuộc dữ liệu thật — xem ghi chú trong tr_r1_benchmark.py).

    Fail loud nếu số query không đúng 12 — không tự cắt/lặp cho khớp,
    đúng tinh thần "Không tự merge/truncate ở benchmark" đã có trong code.
    """
    if len(per_query_events) != expected_total_queries:
        raise ValueError(
            f"Kỳ vọng đúng {expected_total_queries} query TRAKE sạch, "
            f"nhận được {len(per_query_events)}. Kiểm tra lại bộ lọc 'sạch' "
            f"ở tầng gọi hàm này — không tự động cắt/lặp cho khớp số."
        )

    per_query = [
        compute_query_recall(qid, events)
        for qid, events in per_query_events.items()
    ]
    passing = sum(1 for q in per_query if q.passes_target)

    return AcceptanceResult(
        total_queries=len(per_query),
        passing_queries=passing,
        min_passing_required=min_passing_queries,
        overall_pass=passing >= min_passing_queries,
        per_query=per_query,
    )


# ---------------------------------------------------------------------------
# CLI — chấm PASS/FAIL từ file benchmark ĐÃ LƯU (không chạy lại CLIP-L)
# ---------------------------------------------------------------------------


def _build_per_query_events(benchmark_data: dict) -> dict[str, list[dict]]:
    """
    Đọc đúng schema tr_r1_benchmark.json thật (key "queries" -> mỗi query có
    "events" -> mỗi event có "result"."regions" và "gt").
    """
    per_query_events: dict[str, list[dict]] = {}
    for q in benchmark_data["queries"]:
        qid = q["query_id"]
        per_query_events[qid] = [
            {"regions": ev["result"]["regions"], "gt": ev["gt"]}
            for ev in q["events"]
        ]
    return per_query_events


def _ghi_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def main() -> None:
    from aic2026.paths import RUNS_DIR

    benchmark_path = RUNS_DIR / "tr_r1_benchmark.json"
    output_path = RUNS_DIR / "tr_r1_acceptance.json"

    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Chưa có {benchmark_path}. Chạy scripts/benchmark_tr_r1.py trước."
        )

    with benchmark_path.open("r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    per_query_events = _build_per_query_events(benchmark_data)

    # Không tự "lọc sạch" ở đây — nếu benchmark chỉ xử lý đúng 12 query
    # (như thực tế hiện tại của bạn: "TRAKE queries : 12"), toàn bộ tập đó
    # CHÍNH LÀ tập 12 query để chấm, không cần file lọc riêng. Nếu số khác
    # 12, compute_acceptance() sẽ raise lỗi rõ ràng ngay dưới đây.
    result = compute_acceptance(per_query_events)
    result_dict = result.to_dict()

    _ghi_json(output_path, result_dict)

    print("=" * 72)
    print("TR-R1 ACCEPTANCE")
    print("=" * 72)
    print(f"Recall target      : {result_dict['recall_target']}")
    print(f"IoU threshold      : {result_dict['iou_threshold']}")
    print(f"Region top-K       : {result_dict['region_top_k']}")
    print(f"Passing queries    : {result_dict['passing_queries']}/{result_dict['total_queries']} "
          f"(cần >= {result_dict['min_passing_required']})")
    print(f"OVERALL PASS       : {result_dict['overall_pass']}")
    print("Lưu ý              : gate nội bộ, không phải điểm TRAKE chính thức")
    print()
    print(
        f"{'query_id':<10} {'events':<8} {'covered':<8} "
        f"{'recall':<8} {'frame_hit':<10} pass"
    )
    for q in result_dict["per_query"]:
        print(f"{q['query_id']:<10} {q['total_events']:<8} {q['covered_events']:<8} "
              f"{q['recall']:<8} {q['candidate_frame_recall']:<10} "
              f"{q['passes_target']}")
    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
