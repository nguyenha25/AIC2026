"""
check_index_coverage.py — Kiểm tra FAISS index CLIP-L có phủ GT TRAKE chưa.

Chạy:
    python scripts/check_index_coverage.py

Không cần chạy retrieval — chỉ đọc bảng metadata (ids) mà clip_l_index đã
nạp sẵn trong bộ nhớ, cực nhanh.
"""
import json

from aic2026.index import clip_l_index
from aic2026.paths import DEV_DIR


def main() -> None:
    index, ids = clip_l_index.nap_chi_muc()

    if index.ntotal != len(ids):
        raise ValueError(
            f"FAISS có {index.ntotal:,} vector nhưng ids có {len(ids):,} dòng. "
            "Phải rebuild đồng thời index và ids trước khi benchmark."
        )

    required = {"video_id", "frame_idx", "pts_time"}
    missing = required - set(ids.columns)
    if missing:
        raise ValueError(
            "clip_l_ids thiếu cột: " + ", ".join(sorted(missing))
        )

    video_ids = ids["video_id"].astype(str)
    unique_videos = sorted(video_ids.unique())

    batches = sorted(set(v.split("_")[0] for v in unique_videos if "_" in v))

    print("=" * 72)
    print("KIỂM TRA ĐỘ PHỦ FAISS INDEX (CLIP-L)")
    print("=" * 72)
    print(f"Tổng số vector trong index : {len(ids)}")
    print(f"Tổng số video duy nhất     : {len(unique_videos)}")
    print(f"Các batch có trong index   : {batches}")
    print()

    batch_counts = {
        batch: sum(1 for video_id in unique_videos if video_id.startswith(batch + "_"))
        for batch in batches
    }
    print(f"Số video theo batch         : {batch_counts}")

    dev_path = DEV_DIR / "dev_questions.jsonl"
    gt_videos = set()
    with dev_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("loai_truy_van") == "chuoi_su_kien":
                gt_videos.add(str(row["video_id"]).strip())

    index_video_set = set(unique_videos)
    missing_gt = sorted(gt_videos - index_video_set)
    print()
    print(f"GT video TRAKE             : {len(gt_videos)}")
    print(f"GT video có trong index    : {len(gt_videos) - len(missing_gt)}")
    print(f"GT video thiếu khỏi index  : {len(missing_gt)}")

    if missing_gt:
        print(f"Danh sách thiếu             : {missing_gt}")
        print()
        print("=> FAIL DATA COVERAGE: retrieval không thể tìm các GT video trên.")
    else:
        print()
        print("=> PASS DATA COVERAGE: toàn bộ GT video TRAKE có trong CLIP-L index.")
        print("   Nếu recall vẫn thấp, tiếp tục soi raw rank/query translation và")
        print("   query-level video consensus; không kết luận do thiếu batch.")


if __name__ == "__main__":
    main()
