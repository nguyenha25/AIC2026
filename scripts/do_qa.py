"""
Việc 11 — ĐO ĐƯỜNG SINH ĐÁP ÁN Q&A, TÁCH RIÊNG HAI TẦNG.

CÂU HỎI SCRIPT NÀY TRẢ LỜI
--------------------------
Q&A đang 0,2167 trên bộ dev. Nhưng con số đó đo bằng ĐÁP ÁN ĐÚNG ĐIỀN SẴN —
tức chỉ đo tầng TÌM ẢNH. Tầng SINH ĐÁP ÁN chưa từng chạy lần nào.

Script chấm hai lượt trên cùng một tập ứng viên:

    A. đáp án đúng điền sẵn  -> TRẦN của cả câu. Tìm ảnh sai thì hỏng ở đây.
    B. đáp án tự sinh        -> điểm THẬT.

Chênh lệch A - B là phần mất riêng của tầng sinh đáp án. Không tách ra thì
không biết nên sửa chỗ nào: mạch tìm ảnh, hay mạch đọc đáp án.

VÌ SAO KHÔNG NHÉT VÀO run_scoring.py
------------------------------------
`run_scoring` chấm TỆP NỘP đã có sẵn trên đĩa — nó không sinh gì cả, và đó là
đúng việc của nó. Sinh đáp án rồi chấm là một phép đo khác, nên để riêng.

CẦN ẢNH GỐC
-----------
Tầng sinh đáp án đọc chữ trên khung hình. Máy không có ảnh của video đáp án
thì mọi câu đều rơi về đáp án dự phòng, và con số nói về SHARD chứ không nói
về thuật toán. Script đếm và cảnh báo.

CÁCH CHẠY
    python -u -m scripts.do_qa
    python -u -m scripts.do_qa --tep D:\\aic-data\\dev\\dev_co_anh.jsonl
    python -u -m scripts.do_qa --khong-vlm      # chỉ dùng OCR/ASR, nhanh hơn
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# NẠP torch TRƯỚC pandas — xem đầu scripts/chay_trake.py.
try:
    import torch  # noqa: F401
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.eval import compute_final_score  # noqa: E402
from aic2026.eval.answer_match import is_semantic_match  # noqa: E402
from aic2026.paths import (  # noqa: E402
    DEV_QUERIES_PATH,
    KEYFRAMES_DIR,
    RUNS_DIR,
    list_video_ids,
)
from aic2026.submit import QA  # noqa: E402

SO_DONG = 100


def doc_cau_qa(duong_dan: Path) -> list[dict]:
    return [
        q
        for d in duong_dan.open("r", encoding="utf-8")
        if d.strip()
        for q in [json.loads(d)]
        if q.get("loai_truy_van") == "hoi_dap"
    ]


def cham(q: dict, dong: list[dict]) -> float:
    gt = {
        "gt_video_id": q["video_id"],
        "gt_frame_range": [int(q["frame_start"]), int(q["frame_end"])],
        "gt_answer": q["cau_tra_loi"],
    }
    return compute_final_score(gt, dong, QA)["final_score"]


def dung_tim_qa_hien_tai(
    nguon_mo_rong: str = "nguyen_ban",
    nguon_clip_l: str = "marian",
    bo_sung_chu: bool = False,
):
    """Dựng đúng nguồn tìm kiếm QA mà bộ sinh bài nộp đang dùng.

    Không ghi cứng CLIP B/32 + OCR + ASR như bản cũ. Trọng số và nhánh bật
    được đọc từ ``config/rrf_weights.yaml`` theo dạng ``qa``; nhờ vậy phép đo
    Task 11 và ``scripts.tao_bo_nop`` không âm thầm chạy hai pipeline khác nhau.
    """
    from aic2026.index.fts_index import TextSearchIndex
    from aic2026.paths import FTS_DIR
    from aic2026.rank.config import trong_so_theo_dang
    from aic2026.rank.hop_nhat import tim_ung_vien_gop

    trong_so = dict(trong_so_theo_dang("qa"))
    if bo_sung_chu:
        # Chế độ thử nghiệm Task 11. Chưa ghi đè YAML production khi máy này
        # mới đo được 5/18 câu QA.
        trong_so["ocr_fts"] = 1.0
        trong_so["asr"] = 1.0
    dung_clip = float(trong_so.get("clip", 0.0)) > 0
    dung_ocr_fts = float(trong_so.get("ocr_fts", 0.0)) > 0
    dung_asr = float(trong_so.get("asr", 0.0)) > 0
    dung_clip_l = float(trong_so.get("clip_l", 0.0)) > 0

    # Chỉ mở SQLite khi một nhánh chữ thật sự được bật. Cấu hình QA hiện tại
    # chỉ dùng CLIP-L nên không bắt máy phải có FTS chỉ để chạy nghiệm thu.
    kho_chu = None
    if dung_ocr_fts or dung_asr:
        kho_chu = TextSearchIndex(FTS_DIR / "text.sqlite")

    mo_rong = nguon_mo_rong != "nguyen_ban"
    tim = tim_ung_vien_gop(
        kho_chu=kho_chu,
        dung_clip=dung_clip,
        dung_ocr_fts=dung_ocr_fts,
        dung_asr=dung_asr,
        dung_clip_l=dung_clip_l,
        mo_rong_truy_van=mo_rong,
        nguon_mo_rong=nguon_mo_rong if mo_rong else None,
        # Đây là cấu hình production hiện được tao_bo_nop dùng cho CLIP-L.
        nguon_clip_l=nguon_clip_l if dung_clip_l else None,
        dang_cau="qa",
        trong_so=trong_so,
    )
    return tim, trong_so


def main() -> int:
    p = argparse.ArgumentParser(description="Việc 11 — đo đường sinh đáp án Q&A")
    p.add_argument("--tep", default=str(DEV_QUERIES_PATH))
    p.add_argument("--so-ung-vien", type=int, default=500)
    p.add_argument("--so-doc", type=int, default=5,
                   help="đọc bao nhiêu ứng viên đầu để rút đáp án")
    p.add_argument("--khong-vlm", action="store_true",
                   help="chỉ dùng OCR/ASR, không nạp mô hình đọc ảnh")
    p.add_argument(
        "--bo-sung-chu", action="store_true",
        help="thử OCR FTS + ASR trọng số 1.0; không sửa YAML production",
    )
    p.add_argument(
        "--nguon-mo-rong",
        default="nguyen_ban",
        choices=["nguyen_ban", "tu_dien", "marian", "llm"],
        help="mở rộng riêng CLIP B/32; mặc định giữ tiếng Việt theo Task 5",
    )
    p.add_argument(
        "--nguon-clip-l",
        default="marian",
        choices=["tu_dien", "marian", "llm"],
        help="nguồn tiếng Anh cho CLIP-L; mặc định khớp bộ sinh bài nộp",
    )
    args = p.parse_args()

    cau = doc_cau_qa(Path(args.tep))
    if not cau:
        print(f"Không có câu hỏi-đáp nào trong {args.tep}")
        return 1

    co_anh = set(list_video_ids(KEYFRAMES_DIR))
    thieu = [q for q in cau if str(q["video_id"]) not in co_anh]

    print(f"{len(cau)} câu hỏi-đáp | shard có ảnh: "
          f"{sorted({v.split('_')[0] for v in co_anh})}")
    if thieu:
        print(
            f"\n  {len(thieu)}/{len(cau)} câu có video đáp án KHÔNG có ảnh trên máy "
            "này.\n  Những câu đó chắc chắn rơi về đáp án dự phòng — con số nói về\n"
            "  SHARD, không nói về thuật toán. Lọc bớt bằng:\n"
            "    python -m scripts.loc_dev_co_anh"
        )
    print()

    from aic2026.qa_answer import BoDocAnh, tra_loi_theo_hang

    tim, trong_so = dung_tim_qa_hien_tai(
        nguon_mo_rong=args.nguon_mo_rong,
        nguon_clip_l=args.nguon_clip_l,
        bo_sung_chu=args.bo_sung_chu,
    )
    # Chỉ in những nguồn THỰC SỰ được dựng ở dung_tim_qa_hien_tai(). Các khóa
    # ocr/caption có thể còn trọng số kế thừa dương nhưng không có engine được
    # truyền vào; in chúng là bật sẽ làm chẩn đoán sai.
    nhanh_bat = []
    for ten in ("clip", "ocr_fts", "asr", "clip_l"):
        if float(trong_so.get(ten, 0.0)) > 0:
            nhanh_bat.append(ten)
    print(f"Pipeline QA: {nhanh_bat} | trọng số: {trong_so}")
    print(
        f"CLIP B/32: {args.nguon_mo_rong} | "
        f"CLIP-L: {args.nguon_clip_l if 'clip_l' in nhanh_bat else 'tắt'}\n"
    )
    bo_doc = None if args.khong_vlm else BoDocAnh()

    print(f"{'câu':<5}{'trần':>7}{'thật':>7}{'mất':>7}  {'nguồn':<10}đáp án sinh ra")
    print("-" * 78)

    bao_cao = []
    dem_nguon: dict[str, int] = {}

    for q in cau:
        hits = list(tim(q["cau_hoi"], args.so_ung_vien))[:SO_DONG]
        if not hits:
            bao_cao.append({"id": q["id"], "tran": 0.0, "that": 0.0,
                            "dap_an": "", "nguon": "khong-co-ung-vien",
                            "khop": False})
            print(f"{q['id']:<5}{0.0:>7.3f}{0.0:>7.3f}{0.0:>7.3f}  mạch không trả gì")
            continue

        # Sinh đáp án RIÊNG cho từng khung. Bản cũ lấy một đáp án từ top-5
        # rồi gán nó cho cả 100 khung.
        bat_dau = time.perf_counter()
        cap = tra_loi_theo_hang(
            q["cau_hoi"], hits, so_dong=SO_DONG,
            so_hang_vlm=args.so_doc, bo_doc_anh=bo_doc,
            dung_vlm=not args.khong_vlm,
        )
        giay = time.perf_counter() - bat_dau

        # Trần trên chính dãy khung đã nở lân cận.
        tran = cham(q, [
            {"video_id": str(h.video_id), "frame_id": int(h.frame_idx),
             "answer": q["cau_tra_loi"]}
            for h, _ in cap
        ])

        that = cham(q, [
            {"video_id": str(h.video_id), "frame_id": int(h.frame_idx),
             "answer": d_hang.van_ban}
            for h, d_hang in cap
        ])

        cap_gt = [
            (h, d_hang) for h, d_hang in cap
            if str(h.video_id) == str(q["video_id"])
            and int(q["frame_start"]) <= int(h.frame_idx) <= int(q["frame_end"])
        ]
        d = next(
            (d_hang for _, d_hang in cap_gt
             if is_semantic_match(q["cau_tra_loi"], d_hang.van_ban)),
            cap[0][1],
        )
        khop = any(
            is_semantic_match(q["cau_tra_loi"], d_hang.van_ban)
            for _, d_hang in cap_gt
        )
        dem_nguon[d.nguon] = dem_nguon.get(d.nguon, 0) + 1

        bao_cao.append({
            "id": q["id"], "video_id": q["video_id"],
            "co_anh": str(q["video_id"]) in co_anh,
            "tran": tran, "that": that, "mat": tran - that,
            "dap_an_dung": q["cau_tra_loi"], "dap_an_sinh": d.van_ban,
            "nguon": d.nguon, "do_tin": d.do_tin, "khop": khop, "giay": giay,
        })

        dau = "OK" if khop else "  "
        print(
            f"{q['id']:<5}{tran:>7.3f}{that:>7.3f}{tran - that:>7.3f}  "
            f"{d.nguon:<10}{dau} {d.van_ban[:32]!r} (đúng: {q['cau_tra_loi'][:22]!r})"
        )

    # -- tổng kết -----------------------------------------------------------
    n = len(bao_cao)
    tb_tran = sum(b["tran"] for b in bao_cao) / n
    tb_that = sum(b.get("that", 0) for b in bao_cao) / n
    so_khop = sum(1 for b in bao_cao if b.get("khop"))

    print("\n" + "=" * 78)
    print(f"\n  Trần (đáp án đúng điền sẵn) : {tb_tran:.4f}   <- chỉ đo TÌM ẢNH")
    print(f"  Thật (đáp án tự sinh)       : {tb_that:.4f}   <- điểm THẬT")
    print(f"  Tầng sinh đáp án làm mất    : {tb_tran - tb_that:.4f}")
    print(f"\n  Đáp án khớp: {so_khop}/{n} câu")
    print(f"  Nguồn đáp án: {dem_nguon}")

    print("\n  ĐỌC THẾ NÀO\n")
    if tb_tran < 0.05:
        print(
            "  Trần gần 0 — mạch TÌM ẢNH mới là chỗ hỏng, chưa đến lượt tầng\n"
            "  sinh đáp án. Sửa truy hồi trước."
        )
    elif tb_tran - tb_that < 0.05:
        print(
            "  Tầng sinh đáp án gần như không làm mất gì. Muốn Q&A cao hơn thì\n"
            "  phải cải thiện TÌM ẢNH, không phải cải thiện cách đọc đáp án."
        )
    else:
        mat = (tb_tran - tb_that) / tb_tran if tb_tran else 0
        print(
            f"  Tầng sinh đáp án làm mất {mat:.0%} số điểm mà mạch tìm ảnh đã\n"
            "  kiếm được. Đây là chỗ đáng sửa nhất cho Q&A."
        )

    du_phong = dem_nguon.get("du_phong", 0)
    if du_phong > n * 0.3:
        print(
            f"\n  {du_phong}/{n} câu rơi về ĐÁP ÁN DỰ PHÒNG — không nguồn nào đọc\n"
            "  được gì. Thường là thiếu ảnh, thiếu OCR, hoặc chưa cài mô hình đọc\n"
            "  ảnh. Kiểm trước khi kết luận đường Q&A yếu."
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    dich = RUNS_DIR / "do_qa.json"
    with dich.open("w", encoding="utf-8") as f:
        json.dump(
            {"so_cau": n, "tran": tb_tran, "that": tb_that,
             "so_khop": so_khop, "nguon": dem_nguon, "chi_tiet": bao_cao},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nĐã ghi {dich}")

    if n < 10:
        print(
            f"\n  CỠ MẪU {n} CÂU — quá nhỏ để chốt bất cứ điều gì. Con số trên chỉ\n"
            "  trả lời được câu hỏi có/không: đường sinh đáp án CHẠY hay CHẾT."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
