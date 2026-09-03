"""
scripts/diagnose_tr_r1.py — Chẩn đoán recall gần 0 của TR-R1
================================================================

Bối cảnh: benchmark_tr_r1.py chạy xong không lỗi (crash sentencepiece
đã fix), nhưng overlap_recall (diagnostic, định nghĩa lỏng nhất -
CHỈ CẦN IoU > 0) chỉ đạt 0.02 (1/50 event). Con số này THẤP BẤT
THƯỜNG — một retrieval CLIP-L tầm thường vẫn thường tìm đúng VIDEO
cho phần lớn truy vấn ở top-500 candidate (định vị đúng thời điểm
mới khó, còn đúng video thường không quá khó). Recall gần 0 tuyệt
đối gợi ý bug ở tầng DỮ LIỆU/PIPELINE (index và frame_map không khớp
nhau, hoặc query text bị hỏng trước khi tới CLIP), KHÔNG PHẢI do bản
thân CLIP-L "chưa đủ tốt".

Script này kiểm tra theo thứ tự ưu tiên — nghi data trước, nghi model
sau:

    Bước 1: FAISS index và frame_map.parquet có phủ CÙNG một tập
            video hay không.
    Bước 2: Số frame/video giữa 2 nguồn có tương đồng không (lệch
            quá xa -> nghi 2 lần extract khác nhau).
    Bước 3: pts_time range mỗi video trong index có hợp lý so với
            frame_map không (index bị cắt/sample quá thưa?).
    Bước 4: Với toàn bộ GT event, gọi THẲNG tim_hit_clip_l() (bỏ qua
            _gom_vung()/scoring) — video_id đúng của GT có xuất hiện
            ở BẤT KỲ đâu trong top-500 mỗi query variant không, rank bao nhiêu.
            Đây tách bạch: lỗi nằm ở CLIP/index, hay ở bước
            grouping/scoring của TR-R1.

Chạy:
    python -X faulthandler -u scripts\\diagnose_tr_r1.py

Đọc kết quả:
    - Bước 1 coverage < 90%  -> gần như chắc chắn đây là root cause.
      Sửa lại việc build FAISS index (hoặc frame_map) cho khớp đúng
      MỘT nguồn dữ liệu, rồi benchmark lại — đừng tune config trước.
    - Bước 1 OK nhưng Bước 4 vẫn cho thấy GT video không lọt top-500
      ở nhiều event -> vấn đề nằm ở chất lượng query text (su_kien
      tiếng Việt / bản dịch Marian), không phải dữ liệu index.
    - Cả 2 đều OK nhưng recall vẫn thấp -> mới đến lượt nghi ngờ
      CLIP-L / cấu hình region grouping thật sự cần tune.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Khi chạy ``python scripts/diagnose_tr_r1.py``, Python đặt ``scripts/`` ở
# sys.path[0] nên package ``scripts`` không còn import được theo tên đầy đủ.
# Thêm repo root để cả hai cách chạy trực tiếp và ``python -m`` đều hoạt động.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Phải set TRƯỚC mọi import khác, và import open_clip sớm — đúng
# workaround đã dùng để fix crash sentencepiece trong benchmark_tr_r1.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import open_clip  # noqa: E402  (phải import sớm, trước semantic/transformers stack)

from aic2026.semantic.parser import RuleBasedParser  # noqa: E402
from aic2026.trake_retrieval import (  # noqa: E402
    _get_trr1_clip_l_runtime,
    _query_variants,
    _video_consensus_scores,
    tim_hit_clip_l,
)

from scripts.benchmark_tr_r1 import (  # noqa: E402
    DEV_QUESTIONS,
    build_queryplan,
    doc_jsonl,
    get_gt_stages,
    get_gt_video_id,
    load_frame_map,
)


# ============================================================================
# HELPERS
# ============================================================================


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================================
# BƯỚC 1 — VIDEO ID ALIGNMENT
# ============================================================================


def check_video_id_alignment() -> float:
    """
    Nghi ngờ đầu tiên: FAISS index và frame_map.parquet có THỰC SỰ
    trỏ tới cùng một tập video hay không.

    Nếu index chỉ chứa một phần nhỏ video_id có trong frame_map (hoặc
    ngược lại), gần như chắc chắn đây là root cause của recall gần 0
    — CLIP không "kém", nó đơn giản KHÔNG CÓ video đúng trong index
    để mà tìm ra.

    Returns:
        coverage — tỉ lệ video của frame_map thực sự có mặt trong
        index, dùng để in kết luận cuối script.
    """

    _section("BƯỚC 1 — Video ID alignment: FAISS index vs frame_map")

    frame_map = load_frame_map()
    frame_map_videos = set(frame_map["video_id"].astype(str))

    _model, _tokenizer, _device, index, ids = _get_trr1_clip_l_runtime()

    index_videos = set(ids["video_id"].astype(str))

    only_in_frame_map = frame_map_videos - index_videos
    only_in_index = index_videos - frame_map_videos
    common = frame_map_videos & index_videos

    print(f"frame_map   : {len(frame_map_videos)} video, {len(frame_map)} frame")
    print(
        f"FAISS index : {len(index_videos)} video, "
        f"{index.ntotal} vector, len(ids)={len(ids)}"
    )
    print(f"Video CHUNG giữa 2 nguồn                : {len(common)}")
    print(f"Video CHỈ có trong frame_map (thiếu index): {len(only_in_frame_map)}")
    print(f"Video CHỈ có trong index (thiếu frame_map): {len(only_in_index)}")

    if only_in_frame_map:
        sample = sorted(only_in_frame_map)[:10]
        print(f"  Ví dụ video thiếu trong index: {sample}")

    if only_in_index:
        sample = sorted(only_in_index)[:10]
        print(f"  Ví dụ video thừa trong index (không có trong frame_map): {sample}")

    coverage = len(common) / len(frame_map_videos) if frame_map_videos else 0.0
    print(f"\n=> Tỉ lệ video của frame_map CÓ trong index: {coverage:.1%}")

    if coverage < 0.9:
        print(
            "\n!!! CẢNH BÁO: index KHÔNG phủ đủ video của frame_map.\n"
            "    Đây rất có thể là nguyên nhân chính khiến recall gần 0\n"
            "    — không phải do CLIP-L kém, mà do index thiếu/sai video."
        )
    else:
        print("\nOK — index phủ đủ video, đi tiếp Bước 2/3/4.")

    if index.ntotal != len(ids):
        print(
            f"\n!!! CẢNH BÁO: index.ntotal ({index.ntotal}) != len(ids) "
            f"({len(ids)}).\n"
            "    Positions trả về từ FAISS search() có thể map SAI dòng\n"
            "    trong `ids` -> video_id/pts_time trả về đều có thể sai\n"
            "    lệch hàng loạt (root cause rất mạnh cho recall gần 0)."
        )

    return coverage


# ============================================================================
# BƯỚC 2 — SỐ FRAME MỖI VIDEO
# ============================================================================


def check_frame_count_alignment() -> None:
    """
    Kiểm tra thô số frame/video giữa 2 nguồn có gần tương đồng không
    (không cần khớp tuyệt đối — cách sample có thể khác nhau — nhưng
    lệch quá xa (vd 10x) thì nghi ngờ 2 lần extract khác tham số
    sampling, hoặc thậm chí 2 phiên bản dữ liệu khác nhau).
    """

    _section("BƯỚC 2 — Số frame mỗi video: frame_map vs index")

    frame_map = load_frame_map()
    _model, _tokenizer, _device, _index, ids = _get_trr1_clip_l_runtime()

    fm_counts = frame_map["video_id"].astype(str).value_counts()
    idx_counts = ids["video_id"].astype(str).value_counts()

    common_videos = sorted(set(fm_counts.index) & set(idx_counts.index))[:5]

    if not common_videos:
        print("Không có video nào chung để so sánh (xem lại Bước 1).")
        return

    for video_id in common_videos:
        fm_n = int(fm_counts.get(video_id, 0))
        idx_n = int(idx_counts.get(video_id, 0))
        ratio = (idx_n / fm_n) if fm_n else float("inf")

        flag = ""
        if ratio < 0.05 or ratio > 20.0:
            flag = "  <-- lệch rất xa, đáng ngờ"

        print(
            f"  {video_id}: frame_map={fm_n} frame, "
            f"index={idx_n} vector, tỉ lệ={ratio:.3f}{flag}"
        )


# ============================================================================
# BƯỚC 3 — PTS_TIME SANITY
# ============================================================================


def check_video_pts_time_sanity() -> None:
    """
    Với vài video, in min/max pts_time trong index vs frame_map. Nếu
    range lệch nhau nhiều (vd index chỉ có pts_time trong khoảng
    0-5s trong khi video dài 300s theo frame_map) -> nghi ngờ index
    được build từ một lần sample keyframe rất thưa, hoặc bị cắt/lệch
    thời gian.
    """

    _section("BƯỚC 3 — Sanity pts_time range mỗi video")

    frame_map = load_frame_map()
    _model, _tokenizer, _device, _index, ids = _get_trr1_clip_l_runtime()

    sample_videos = sorted(set(frame_map["video_id"].astype(str)))[:5]

    for video_id in sample_videos:
        fm_rows = frame_map[frame_map["video_id"].astype(str) == video_id]
        idx_rows = ids[ids["video_id"].astype(str) == video_id]

        fm_range = (
            (float(fm_rows["pts_time"].min()), float(fm_rows["pts_time"].max()))
            if len(fm_rows)
            else None
        )
        idx_range = (
            (float(idx_rows["pts_time"].min()), float(idx_rows["pts_time"].max()))
            if len(idx_rows)
            else None
        )

        print(
            f"  {video_id}: frame_map range={fm_range}, "
            f"index range={idx_range}"
        )


# ============================================================================
# BƯỚC 4 — RAW RETRIEVAL CHO GT EVENT THẬT
# ============================================================================


def check_raw_retrieval() -> dict[str, float]:
    """
    Cầm TRỰC TIẾP toàn bộ event GT thật, gọi tim_hit_clip_l() với
    top_k=500 mỗi query variant (KHÔNG qua _gom_vung()/scoring của TR-R1),
    rồi kiểm tra video/frame đúng có xuất hiện trong raw hits hay không.

    Đây tách bạch rõ: nếu ngay ở bước retrieval thô video đúng đã
    KHÔNG xuất hiện -> lỗi nằm ở CLIP/index/query text, không phải ở
    _gom_vung()/scoring (những phần đã có test pass đầy đủ).
    """

    _section("BƯỚC 4 — Raw CLIP-L retrieval (top_k=500 mỗi variant)")

    rows = doc_jsonl(DEV_QUESTIONS)
    trake_rows = [r for r in rows if r.get("loai_truy_van") == "chuoi_su_kien"]

    parser = RuleBasedParser()

    total_events = 0
    raw_video_hits = 0
    raw_frame_hits = 0
    consensus_top_10 = 0

    for row in trake_rows:

        query_id = str(row["id"])
        gt_video_id = get_gt_video_id(row)
        gt_stages = get_gt_stages(row)
        plan = build_queryplan(gt_stages, parser)

        print(f"\n--- query={query_id}  GT video_id={gt_video_id} ---")

        hits_by_event = []

        for event, stage in zip(plan.events, gt_stages):

            total_events += 1

            variants = _query_variants(
                event.text,
                use_query_expansion=True,
                max_query_variants=2,
            )

            print(f"  event.text = {event.text!r}")
            print(f"    variants = {variants!r}")

            hits = tim_hit_clip_l(event.text, top_k=500)
            hits_by_event.append(hits)

            gt_hit_ranks = [
                rank
                for rank, hit in enumerate(hits, start=1)
                if str(hit.get("video_id")) == gt_video_id
            ]

            if gt_hit_ranks:
                raw_video_hits += 1
                shown = gt_hit_ranks[:5]
                more = " ..." if len(gt_hit_ranks) > 5 else ""
                print(
                    f"    -> GT video XUẤT HIỆN trong raw hits, rank "
                    f"{shown}{more} ({len(gt_hit_ranks)} hit tổng)"
                )
            else:
                print(
                    "    -> GT video KHÔNG xuất hiện trong bất kỳ hit thô "
                    "nào."
                )

            frame_start = int(stage["frame_start"])
            frame_end = int(stage["frame_end"])
            if frame_start > frame_end:
                frame_start, frame_end = frame_end, frame_start

            matching_frames = [
                hit
                for hit in hits
                if str(hit.get("video_id")) == gt_video_id
                and hit.get("frame_idx") is not None
                and frame_start <= int(hit["frame_idx"]) <= frame_end
            ]

            if matching_frames:
                raw_frame_hits += 1
                best = min(
                    matching_frames,
                    key=lambda hit: int(hit.get("rank", 10**9)),
                )
                print(
                    "    -> HIT frame chính thức trong GT: "
                    f"frame_idx={best['frame_idx']} rank={best.get('rank')}"
                )
            else:
                print(
                    "    -> Không có raw frame nào nằm trực tiếp trong "
                    "[frame_start, frame_end]."
                )

            if hits:
                top1 = hits[0]
                print(
                    f"    top-1 thực tế trả về : "
                    f"video_id={top1.get('video_id')} "
                    f"pts_time={top1.get('pts_time')} "
                    f"score={float(top1.get('score', 0.0)):.4f}"
                )
            else:
                print("    -> tim_hit_clip_l() trả về RỖNG.")

        consensus = _video_consensus_scores(hits_by_event)
        consensus_order = sorted(
            consensus,
            key=lambda video_id: (-consensus[video_id], video_id),
        )
        consensus_rank = (
            consensus_order.index(gt_video_id) + 1
            if gt_video_id in consensus_order
            else None
        )

        if consensus_rank is not None and consensus_rank <= 10:
            consensus_top_10 += 1

        print(
            f"  query-level video consensus rank của GT: {consensus_rank}"
        )

    summary = {
        "total_events": total_events,
        "raw_video_hits": raw_video_hits,
        "raw_frame_hits": raw_frame_hits,
        "consensus_top_10_queries": consensus_top_10,
        "total_queries": len(trake_rows),
    }

    print()
    print(
        "Raw GT-video recall  : "
        f"{raw_video_hits}/{total_events} "
        f"({raw_video_hits / total_events if total_events else 0.0:.1%})"
    )
    print(
        "Raw GT-frame recall  : "
        f"{raw_frame_hits}/{total_events} "
        f"({raw_frame_hits / total_events if total_events else 0.0:.1%})"
    )
    print(
        "GT video consensus@10: "
        f"{consensus_top_10}/{len(trake_rows)} query"
    )

    return summary


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    coverage = check_video_id_alignment()
    check_frame_count_alignment()
    check_video_pts_time_sanity()
    raw_summary = check_raw_retrieval()

    _section("KẾT LUẬN")

    if coverage < 0.9:
        print(
            "Bước 1 cho coverage thấp — xử lý đó TRƯỚC TIÊN: sửa lại việc\n"
            "build FAISS index (hoặc frame_map) cho khớp đúng MỘT nguồn dữ\n"
            "liệu, rồi mới chạy lại benchmark. Đừng tune TRR1Config trước\n"
            "khi bước này OK — tune trên dữ liệu lệch sẽ không có ý nghĩa."
        )
    elif raw_summary["raw_video_hits"] == 0:
        print(
            "Index phủ video nhưng GT video không xuất hiện trong raw top-K.\n"
            "Ưu tiên kiểm tra query variants/bản dịch và encoder/index model\n"
            "identity; tune region grouping không thể sửa được tầng này."
        )
    else:
        print(
            "Index có coverage và raw retrieval đã tìm thấy GT video ở một\n"
            "số event. So Raw GT-video recall với GT video consensus@10:\n"
            "  - consensus tốt nhưng benchmark thấp -> lỗi/ranking region;\n"
            "  - raw video tốt nhưng raw frame thấp -> cần dense/local R2;\n"
            "  - raw video thấp -> cải thiện query variants hoặc retrieval."
        )


if __name__ == "__main__":
    main()
    
