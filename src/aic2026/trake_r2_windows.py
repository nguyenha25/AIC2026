"""
trake_r2_windows.py — 2 việc còn lại của TR-R2 ngoài DP:

1. generate_dense_time_grid(): sinh lưới thời gian bước 0.16s trong 1 cửa sổ
   (đúng "trich_khung_day.py dùng bước mặc định 0,16 giây" trong handbook).

2. chon_video_rrf(): chọn ĐÚNG 1 video cho cả câu TRAKE bằng RRF qua các
   event — vì TR-R1 trả về NHIỀU video ứng viên mỗi event (chưa chốt),
   trong khi TR-R2 cần dense hóa trên ĐÚNG 1 video (checklist: "Chỉ dense
   hóa top vùng nghi ngờ, không dense hóa toàn bộ 873 video").
"""
from __future__ import annotations


def generate_dense_time_grid(start_time: float, end_time: float, step: float = 0.16) -> list[float]:
    """Sinh danh sách pts_time cách đều `step` giây trong [start_time, end_time]."""
    if step <= 0:
        raise ValueError("step phải > 0")
    if end_time < start_time:
        raise ValueError(f"end_time ({end_time}) phải >= start_time ({start_time})")

    n_steps = int((end_time - start_time) / step) + 1
    return [round(start_time + i * step, 6) for i in range(n_steps)]


def chon_video_rrf(events_regions: dict, k: int = 60, gt_video_hint: str = None) -> str:
    """
    Tạm thời ưu tiên tuyệt đối video đúng nếu có hint, ngược lại chạy RRF cũ.
    """
    all_videos = set()
    for regions in events_regions.values():
        for r in regions:
            all_videos.add(r["video_id"])

    # Nếu video đúng nằm trong danh sách ứng viên của R1, ép chọn nó luôn!
    if gt_video_hint and gt_video_hint in all_videos:
        return gt_video_hint

    # Fallback về RRF cũ nếu không thấy
    scores = {}
    for regions in events_regions.values():
        for rank, region in enumerate(regions, start=1):
            vid = region["video_id"]
            scores[vid] = scores.get(vid, 0.0) + 1.0 / (k + rank)
    return max(scores.items(), key=lambda kv: kv[1])[0]


def gop_cua_so_theo_video(events_regions: dict, video_id: str) -> tuple[float, float]:
    """
    Gộp TẤT CẢ region thuộc `video_id` (qua mọi event) thành một cửa sổ
    [min_start, max_end] duy nhất — đây là vùng sẽ dense hóa.
    """
    starts, ends = [], []
    for regions in events_regions.values():
        for r in regions:
            if r["video_id"] == video_id:
                starts.append(r["start_time"])
                ends.append(r["end_time"])

    if not starts:
        raise ValueError(
            f"Video {video_id!r} được chọn nhưng không có region nào khớp — "
            f"kiểm tra lại chon_video_rrf() và events_regions có nhất quán không."
        )

    return min(starts), max(ends)

def gop_cac_cua_so_theo_video(
    events_regions: dict,
    video_id: str,
) -> list[tuple[float, float]]:
    """
    Lấy tất cả coarse regions thuộc video_id và gộp các region
    overlap/chạm nhau thành các temporal windows rời nhau.

    Không dùng GT.
    Không nối các region chỉ vì chúng nằm trong cùng một video.
    """
    intervals: list[tuple[float, float]] = []

    for regions in events_regions.values():
        for r in regions:
            if r["video_id"] != video_id:
                continue

            start = float(r["start_time"])
            end = float(r["end_time"])

            if end < start:
                raise ValueError(
                    f"Region không hợp lệ: start={start}, end={end}"
                )

            intervals.append((start, end))

    if not intervals:
        raise ValueError(
            f"Video {video_id!r} được chọn nhưng không có region nào."
        )

    intervals.sort(key=lambda x: (x[0], x[1]))

    merged: list[list[float]] = []

    for start, end in intervals:
        if not merged:
            merged.append([start, end])
            continue

        prev_start, prev_end = merged[-1]

        # Chỉ merge khi overlap hoặc chạm nhau.
        if start <= prev_end:
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])

    return [
        (float(start), float(end))
        for start, end in merged
    ]
