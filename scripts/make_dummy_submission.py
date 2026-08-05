"""
TASK 8 — tạo tệp nộp giả 100 dòng đúng định dạng BTC.

Chạy:  python -m scripts.make_dummy_submission

Sinh ba tệp trong <gốc dữ liệu>/submissions/:
    query-demo-kis.csv      100 dòng: video_id, frame_id
    query-demo-qa.csv       100 dòng: video_id, frame_id, answer
    query-demo-trake.csv    100 dòng: video_id, frame_id_1..frame_id_4

frame_id lấy từ CỘT frame_idx của map-keyframes thật, không bịa số.
Đây chính là chỗ chứng minh nhóm hiểu đúng yêu cầu nộp bài ngay từ đầu,
chứ không phải đến sát hạn mới phát hiện sai định dạng.
"""

import csv

from src.aic2026.frame_map import FrameMap, available_video_ids
from src.aic2026.paths import SUBMISSIONS_DIR
from src.aic2026.submit import (
    KIS,
    QA,
    TRAKE,
    Answer,
    SubmissionBudget,
    submission_filename,
)

TARGET_ROWS = 100
TRAKE_STEPS = 4


def collect_rows(max_videos: int = 40):
    """Gom (video_id, [frame_idx...]) từ dữ liệu thật."""
    ids = available_video_ids()[:max_videos]
    out = []
    for vid in ids:
        try:
            fm = FrameMap.load(vid)
        except Exception:
            continue
        if len(fm) >= TRAKE_STEPS:
            out.append(fm)
    return out


def main() -> int:
    maps = collect_rows()
    if not maps:
        print("Chưa có map-keyframes trên máy. Làm Task 4 trước.")
        return 1

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print("TẠO TỆP NỘP GIẢ (Task 8)")
    print("=" * 66)
    print(f"Nguồn : {len(maps)} video thật")
    print(f"Đích  : {SUBMISSIONS_DIR}")
    print()

    # --- KIS -------------------------------------------------------------
    kis = SubmissionBudget(task=KIS)
    step = 0
    while not kis.is_full() and step < 2000:
        fm = maps[step % len(maps)]
        row = fm.rows[(step // len(maps)) % len(fm)]
        kis.add(Answer(video_id=fm.video_id, frame_ids=[row.frame_idx]))
        step += 1

    # --- Q&A -------------------------------------------------------------
    qa = SubmissionBudget(task=QA)
    step = 0
    while not qa.is_full() and step < 2000:
        fm = maps[step % len(maps)]
        row = fm.rows[(step // len(maps)) % len(fm)]
        qa.add(Answer(
            video_id=fm.video_id,
            frame_ids=[row.frame_idx],
            answer=f"đáp án thử {step + 1}",   # có dấu, để kiểm bảng mã tệp nộp
        ))
        step += 1

    # --- TRAKE -----------------------------------------------------------
    trake = SubmissionBudget(task=TRAKE)
    step = 0
    while not trake.is_full() and step < 2000:
        fm = maps[step % len(maps)]
        base = (step // len(maps)) % max(1, len(fm) - TRAKE_STEPS)
        seq = [fm.rows[base + k].frame_idx for k in range(TRAKE_STEPS)]
        trake.add(Answer(video_id=fm.video_id, frame_ids=seq))
        step += 1

    # --- Ghi ra tệp ------------------------------------------------------
    results = []
    for budget in (kis, qa, trake):
        path = SUBMISSIONS_DIR / submission_filename("demo", budget.task)
        budget.write(path)
        results.append((budget, path))
        print(f"[ghi] {path.name:<24} {budget.report()}")

    # --- Tự soát lại tệp vừa ghi -----------------------------------------
    print()
    print("-" * 66)
    print("TỰ SOÁT LẠI TỆP VỪA GHI")
    print("-" * 66)
    ok = True
    expected_cols = {KIS: 2, QA: 3, TRAKE: 1 + TRAKE_STEPS}

    for budget, path in results:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        n_rows = len(rows)
        widths = {len(r) for r in rows}
        dup = n_rows - len({tuple(r) for r in rows})
        want = expected_cols[budget.task]

        line = f"{path.name:<24} {n_rows:>3} dòng | số cột {sorted(widths)} | trùng {dup}"
        if n_rows != TARGET_ROWS or widths != {want} or dup:
            ok = False
            print("[LỖI]    " + line + f"  (cần {TARGET_ROWS} dòng, {want} cột, 0 trùng)")
        else:
            print("[ĐẠT]    " + line)

        print(f"         dòng đầu: {','.join(rows[0])}")

    print()
    if not ok:
        return 1
    print("[ĐẠT] Ba tệp đúng số dòng, đúng số cột, không dòng nào trùng.")
    print()
    print("Lưu ý: quy ước ĐẶT TÊN TỆP (query-<id>-<dạng>.csv) là nhóm tự chọn.")
    print("Khi BTC công bố tên chính thức, chỉ sửa hàm submission_filename()")
    print("trong src/aic2026/submit/formatter.py, chỗ khác không phải đụng.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
