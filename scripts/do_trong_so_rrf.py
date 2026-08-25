"""
Việc 6 — CHỈNH TRỌNG SỐ RRF THEO TỪNG DẠNG CÂU.

VẤN ĐỀ
------
Bốn nhánh đang để trọng số gần như ngang nhau. Nhưng trong 25 câu vòng p1,
nhánh ASR quyết định khoảng 10 câu và VÔ DỤNG HOÀN TOÀN với 15 câu còn lại.
Cào bằng trọng số nghĩa là ở 15 câu kia, nhánh ASR vẫn được góp một phiếu
ngang nhánh CLIP — loãng tín hiệu của nhánh đang đúng.

CÁCH LÀM — VÌ SAO KHÔNG CHẠY LẠI MẠCH CHO TỪNG BỘ TRỌNG SỐ
----------------------------------------------------------
Trọng số RRF KHÔNG ảnh hưởng tới việc mỗi nhánh trả về cái gì — nó chỉ ảnh
hưởng tới cách CỘNG các bảng xếp hạng lại. Nên:

    1. Tra kho MỘT LẦN cho mỗi câu, mỗi nhánh. Lưu lại bảng xếp hạng thô.
    2. Quét hàng nghìn bộ trọng số bằng cách gộp lại các bảng ĐÃ LƯU.

Bước 1 mất vài phút (nạp mô hình, tra FAISS). Bước 2 chạy hoàn toàn trong RAM.
Chạy lại mạch cho từng bộ trọng số thì một lần quét mất hàng giờ và không ai
quét nữa.

ĐỌC KỸ — PHÉP ĐO NÀY ĐO CÁI GÌ
------------------------------
Chỉ đo KHẢ NĂNG TÌM ĐÚNG ẢNH. Với câu Q&A, ô đáp án được điền bằng đáp án
ĐÚNG lấy từ bộ dev, giống hệt cách baseline Giai đoạn 1 đã đo. Nghĩa là con
số ở đây KHÔNG bao gồm chất lượng trả lời của Việc 11 — muốn đo cả đường
Q&A thì chạy scripts/run_scoring.py sau khi Việc 11 đã sinh đáp án thật.

TRAKE bị loại khỏi phép quét theo mặc định. Không phải vì lười: TRAKE đang có
trần 0,000 do keyframe thưa hơn cửa sổ đáp án. Quét trọng số trên một số 0
cố định chỉ ra nhiễu. Làm Việc 9 trước.

CÁCH CHẠY
---------
    python -u -m scripts.do_trong_so_rrf                 # quét thô
    python -u -m scripts.do_trong_so_rrf --min           # quét tinh quanh bản tốt
    python -u -m scripts.do_trong_so_rrf --chi-do-rieng  # chỉ đo từng nhánh
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# NẠP torch TRƯỚC MỌI THỨ KHÁC. ĐỪNG DỜI KHỐI NÀY XUỐNG DƯỚI.
#
# Trên Windows, nạp pandas (hoặc faiss, onnxruntime) vào tiến trình TRƯỚC torch
# thì tiến trình chết ngay với 0xC0000005: không traceback, không thông báo,
# màn hình chỉ dừng giữa chừng rồi về dấu nhắc — nhìn y như chạy xong. Mỗi thư
# viện mang một bản OpenMP riêng, bản nạp sau giẫm lên bản nạp trước.
#
# Script này chạm frame_map (pandas) trước khi chạm mô hình (torch), nên nếu
# không ghim thứ tự ở đây thì thứ tự nạp là ngẫu nhiên theo đường mã chạy.
# Cùng một cái bẫy đã ghi ở rank/search.py, hàm tim_ung_vien_clip().
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
except ImportError:
    pass          # máy chưa cài torch: để phần kiểm phụ thuộc báo lỗi tử tế

import yaml   # noqa: E402

from aic2026.eval import compute_final_score           # noqa: E402
from aic2026.paths import CONFIG_DIR, DEV_QUERIES_PATH, RUNS_DIR   # noqa: E402
from aic2026.rank.dedupe import loc_trung              # noqa: E402
from aic2026.rank.fuse import reciprocal_rank_fusion   # noqa: E402
from aic2026.submit import KIS, QA                     # noqa: E402
from scripts.run_scoring import build_gt, doc_dev_questions   # noqa: E402

# CLIP luôn = 1.0. Trọng số RRF chỉ có ý nghĩa TƯƠNG ĐỐI — nhân cả bốn số với
# 2 thì thứ hạng không đổi một chút nào. Ghim một nhánh lại thì lưới quét nhỏ
# đi bốn lần mà không mất bộ nào.
LUOI_THO = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BUOC_TINH = 0.1

NHANH = ["ocr_fts", "asr", "object", "caption", "clip_l"]

TEP_TRONG_SO = CONFIG_DIR / "rrf_weights.yaml"

_THAM_CHIEU: dict = {}

# Đếm ứng viên KHÔNG có ảnh khi rerank. Đây là con số quyết định phép đo có
# đọc được hay không — xem cảnh báo ở cuối bước 1.
THONG_KE_RERANK = {"da_xep": 0, "thieu_anh": 0}


# ---------------------------------------------------------------------------
# Bước 1 — tra kho một lần, lưu bảng xếp hạng thô
# ---------------------------------------------------------------------------

def dung_cac_nguon(
    dung_caption: bool,
    dung_object: bool,
    mo_rong_clip: bool = False,
    loc_vat_the: bool = False,
    loc_token_hiem_ocr: bool = False,
    dung_clip_l: bool = False,
    rerank: bool = False,
    rerank_so_dau: int = 100,
):
    """Trả về {tên nhánh: hàm (câu chữ, số ứng viên) -> list[dict]}.

    mo_rong_clip=False: nhánh CLIP nhận NGUYÊN câu tiếng Việt. Đây là MỐC NỀN.
    mo_rong_clip=True : nhánh CLIP nhận cụm tiếng Anh của Việc 5.

    Bản đầu luôn gọi thẳng tim_ung_vien_clip, nên đặt $env:AIC_NGUON_MO_RONG
    rồi chạy lại chỉ ra SỐ GIỐNG HỆT — Việc 5 chưa từng được đo, mà không có
    dấu hiệu gì báo điều đó.
    """
    from aic2026.index.fts_index import TextSearchIndex
    from aic2026.paths import FTS_DIR
    from aic2026.rank.hop_nhat import hit_sang_dict, no_khoang_asr

    kho_chu = TextSearchIndex(FTS_DIR / "text.sqlite")
    nguon = {}

    if mo_rong_clip:
        from aic2026.query_expand import tim_ung_vien_clip_mo_rong

        _tim = tim_ung_vien_clip_mo_rong()
    else:
        from aic2026.rank.search import tim_ung_vien_clip as _tim

    if rerank:
        # VIỆC 4 — xếp lại top-N bằng mô hình mạnh hơn.
        #
        # Bọc riêng nhánh CLIP: rerank chấm ảnh với câu chữ, nên nó chỉ có
        # nghĩa cho nhánh ảnh. Bọc cả OCR/ASR là đè lên tín hiệu chữ bằng tín
        # hiệu ảnh — hai thứ khác nhau.
        from aic2026.rerank import Reranker

        _bo_xep = Reranker()
        _tim_tho = _tim

        def _tim(cau, k):
            # Tầng thô và tầng rerank phải nhận CÙNG một câu cho mô hình ảnh.
            # Khi Việc 5 bật, _tim_tho đã dùng cụm tiếng Anh; nếu rerank lại
            # chấm bằng câu Việt gốc thì phép đo đang đổi hai biến cùng lúc.
            cau_rerank = cau
            if mo_rong_clip:
                from aic2026.query_expand import mo_rong as _mo_rong

                cau_rerank = _mo_rong(cau).cum_chinh
            da_xep, bao_cao = _bo_xep.xep_lai(
                cau_rerank, _tim_tho(cau, k), so_dau=rerank_so_dau
            )
            THONG_KE_RERANK["da_xep"] += bao_cao.so_da_xep_lai
            THONG_KE_RERANK["thieu_anh"] += bao_cao.so_thieu_anh
            return da_xep

    def _clip(cau, k):
        return [hit_sang_dict(h) for h in _tim(cau, k)]

    nguon["clip"] = _clip
    nguon["ocr_fts"] = lambda cau, k: kho_chu.search_text(
        cau, top_k=k, loc_hiem=loc_token_hiem_ocr
    )
    nguon["asr"] = lambda cau, k: [
        hit_sang_dict(h) for h in no_khoang_asr(kho_chu.search_asr(cau, top_k=k))
    ]

    if dung_object or loc_vat_the:
        from aic2026.index.objects_index import ObjectSearchIndex

        kho_vt = ObjectSearchIndex()
        if dung_object:
            nguon["object"] = lambda cau, k: kho_vt.tra_bang_cau_viet(cau, top_k=k)

    if loc_vat_the:
        # Bọc MỌI nhánh: bộ lọc chạy trước, nhánh nào cũng chỉ thấy phần khung
        # còn lại. Đây là điểm khác căn bản với nhánh xếp hạng — nó tác động
        # lên cả CLIP, OCR và ASR chứ không chỉ góp thêm một bảng.
        def _boc(ham):
            def f(cau, k):
                ra = ham(cau, k)
                tap = kho_vt.khung_chua(cau)
                if tap is None:
                    return ra
                loc = [
                    d for d in ra
                    if (str(d["video_id"]), round(float(d.get("pts_time") or 0), 3))
                    in tap
                ]
                return loc or ra          # lọc sạch thì trả về nguyên
            return f

        nguon = {ten: _boc(h) for ten, h in nguon.items()}

    if dung_clip_l:
        from aic2026.index.clip_l_index import tim as _tim_clip_l

        if mo_rong_clip:
            from aic2026.query_expand import mo_rong as _mr

            def _clip_l(cau, k):
                try:
                    cau = _mr(cau).cum_chinh
                except Exception:
                    pass
                return _tim_clip_l(cau, k)
        else:
            _clip_l = _tim_clip_l

        nguon["clip_l"] = _clip_l

    if dung_caption:
        from aic2026.enrich.caption import CaptionSearchIndex

        kho_cap = CaptionSearchIndex()
        nguon["caption"] = lambda cau, k: kho_cap.tra_cuu(cau, top_k=k)

    return nguon


def tra_kho_mot_lan(cau_hoi_list, nguon, so_ung_vien: int) -> dict:
    """{query_id: {tên nhánh: [dict đã xếp hạng]}}."""
    kho_tam: dict[str, dict[str, list]] = {}

    for thu_tu, q in enumerate(cau_hoi_list, 1):
        qid = str(q["id"])
        cau = q["cau_hoi"]
        kho_tam[qid] = {}
        for ten, ham in nguon.items():
            bat_dau = time.perf_counter()
            try:
                kho_tam[qid][ten] = list(ham(cau, so_ung_vien))
            except Exception as loi:
                print(f"    ! {ten} lỗi ở câu {qid}: {loi}")
                kho_tam[qid][ten] = []
            print(
                f"  [{thu_tu}/{len(cau_hoi_list)}] {qid} {ten:<9} "
                f"{len(kho_tam[qid][ten]):>4} ứng viên "
                f"({(time.perf_counter() - bat_dau) * 1000:.0f} ms)"
            )
    return kho_tam


# ---------------------------------------------------------------------------
# Bước 2 — gộp trong RAM và chấm
# ---------------------------------------------------------------------------

class _MucLoc:
    """Vỏ mỏng để dùng lại loc_trung() — nó chỉ cần video_id và pts_time."""

    __slots__ = ("video_id", "pts_time", "frame_idx", "n", "score")

    def __init__(self, d):
        self.video_id = str(d["video_id"])
        self.pts_time = float(d.get("pts_time") or 0.0)
        self.frame_idx = int(d.get("frame_idx", -1))
        self.n = int(d.get("n", -1))
        self.score = float(d.get("score", 0.0))


def cham_mot_bo(kho_tam, cau_hoi_list, trong_so, cua_so_giay, so_dong, k_rrf):
    """Điểm trung bình của MỘT bộ trọng số, tách theo dạng câu."""
    diem_theo_dang: dict[str, list[float]] = {}

    for q in cau_hoi_list:
        qid, task, gt = build_gt(q)
        if task not in (KIS, QA):
            continue

        gop = reciprocal_rank_fusion(kho_tam[qid], weights=trong_so, k_rrf=k_rrf)
        da_loc, _ = loc_trung(
            [_MucLoc(d) for d in gop],
            cua_so_giay=cua_so_giay,
            so_anh_toi_da_moi_video=None,
        )

        dong = []
        for m in da_loc[:so_dong]:
            r = {"video_id": m.video_id, "frame_id": m.frame_idx}
            if task == QA:
                # Điền đáp án ĐÚNG: phép đo này chỉ đo khả năng TÌM ẢNH.
                r["answer"] = gt["gt_answer"]
            dong.append(r)

        diem_theo_dang.setdefault(task, []).append(
            compute_final_score(gt, dong, task)["final_score"]
        )

    ket_qua = {
        dang: sum(ds) / len(ds) for dang, ds in diem_theo_dang.items() if ds
    }
    tat_ca = [x for ds in diem_theo_dang.values() for x in ds]
    ket_qua["tat_ca"] = sum(tat_ca) / len(tat_ca) if tat_ca else 0.0
    ket_qua["_so_cau"] = {d: len(ds) for d, ds in diem_theo_dang.items()}
    return ket_qua


def sinh_luoi(nhanh_co_that: list[str], quanh: dict | None = None):
    """Sinh các bộ trọng số cần thử."""
    if quanh is None:
        gia_tri = {t: LUOI_THO for t in nhanh_co_that}
    else:
        gia_tri = {}
        for t in nhanh_co_that:
            goc = quanh.get(t, 0.5)
            buoc = {
                round(max(0.0, min(1.5, goc + d)), 2)
                for d in (-2 * BUOC_TINH, -BUOC_TINH, 0.0, BUOC_TINH, 2 * BUOC_TINH)
            }
            # LUÔN thử 0.0. Không có nó thì câu hỏi quan trọng nhất — "nhánh
            # này có đóng góp gì không" — không bao giờ được hỏi. Lưới tinh
            # quanh 0.8 chỉ chạy 0.6..1.0 và bỏ sót hẳn phương án tắt nhánh.
            buoc.add(0.0)
            gia_tri[t] = sorted(buoc)

    ten = list(gia_tri)
    for to_hop in itertools.product(*(gia_tri[t] for t in ten)):
        bo = {"clip": 1.0}
        bo.update(dict(zip(ten, to_hop)))
        yield bo


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tep", default=str(DEV_QUERIES_PATH))
    p.add_argument("--so-ung-vien", type=int, default=500)
    p.add_argument("--so-dong", type=int, default=100)
    p.add_argument("--k-rrf", type=int, default=60)
    p.add_argument("--cua-so-giay", type=float, default=10.0)
    p.add_argument("--khong-object", action="store_true")
    p.add_argument("--loc-vat-the", action="store_true",
                   help="Việc 3 làm BỘ LỌC: thu hẹp ứng viên trước khi gộp")
    p.add_argument("--clip-l", action="store_true",
                   help="Việc 8: bật nhánh chỉ mục ảnh thứ hai (ViT-L-14)")
    p.add_argument("--loc-token-hiem", action="store_true",
                   help="nhánh OCR chỉ tra token HIẾM, bỏ token phổ biến")
    p.add_argument("--rerank", action="store_true",
                   help="Việc 4: xếp lại top-N bằng mô hình mạnh hơn")
    p.add_argument("--rerank-so-dau", type=int, default=100)
    p.add_argument("--khong-caption", action="store_true")
    p.add_argument("--min", action="store_true", help="quét tinh quanh bản đang chốt")
    p.add_argument("--chi-do-rieng", action="store_true")
    p.add_argument("--nguon-mo-rong", choices=["tu_dien", "marian", "llm"],
                   default=None, help="bật Việc 5 cho nhánh CLIP")
    args = p.parse_args()

    cau_hoi = doc_dev_questions(Path(args.tep))
    so_kis_qa = sum(
        1 for q in cau_hoi if q.get("loai_truy_van") in ("mo_ta", "hoi_dap")
    )
    print(f"Bộ dev: {len(cau_hoi)} câu, trong đó {so_kis_qa} câu KIS/QA được chấm.")
    if so_kis_qa < 40:
        print(
            "  CẢNH BÁO: dưới 40 câu thì sai số cỡ ±9 điểm phần trăm. Chênh lệch "
            "nhỏ hơn mức đó KHÔNG kết luận được gì. Làm Việc 1 (dev v2) trước."
        )
    print(
        "  TRAKE bị loại khỏi phép quét — đang có trần 0,000 do keyframe thưa "
        "hơn cửa sổ đáp án. Làm Việc 9 trước.\n"
    )

    mo_rong = bool(args.nguon_mo_rong)
    if mo_rong:
        import os

        os.environ["AIC_NGUON_MO_RONG"] = args.nguon_mo_rong

        # Dịch thử MỘT câu trước: nhánh hỏng thì phải biết ngay, không phải
        # sau khi in ra một bảng số trông như đã đo xong.
        from aic2026.query_expand import LuiNhanhKhongMongMuon, mo_rong as _mr

        try:
            thu = _mr("một người đàn ông cầm ô", bat_buoc=True)
        except LuiNhanhKhongMongMuon as loi:
            print(f"NHÁNH DỊCH {args.nguon_mo_rong.upper()} KHÔNG DÙNG ĐƯỢC\n  {loi}")
            return 1
        print(f"Thử dịch: 'một người đàn ông cầm ô' -> {thu.cum_chinh!r}")

    if args.rerank:
        # Thử xếp lại MỘT lượt trước khi chạy cả mẻ. Rerank hỏng thì phải biết
        # ngay, không phải sau vài phút và một bảng số trông như đã đo xong.
        try:
            from aic2026.rerank import Reranker

            _thu = Reranker()
            _thu.ma_hoa_cau("a man holding an umbrella")
            print(f"Rerank: {_thu.ten_day_du} — nạp được")
        except Exception as loi:
            print(
                f"RERANK KHÔNG DÙNG ĐƯỢC: {type(loi).__name__}: {loi}\n"
                "  Cần: pip install open-clip-torch\n"
                "  Dừng ở đây thay vì chạy tiếp rồi in một bảng số giống hệt\n"
                "  lượt không rerank — đó là phép đo giả."
            )
            return 1

    nguon = dung_cac_nguon(
        dung_caption=not args.khong_caption,
        dung_object=not args.khong_object,
        mo_rong_clip=mo_rong,
        loc_vat_the=args.loc_vat_the,
        loc_token_hiem_ocr=args.loc_token_hiem,
        dung_clip_l=args.clip_l,
        rerank=args.rerank,
        rerank_so_dau=args.rerank_so_dau,
    )
    print(
        f"Nhánh bật: {', '.join(nguon)} | nhánh CLIP: "
        + (f"mở rộng qua {args.nguon_mo_rong}" if mo_rong
           else "NGUYÊN câu tiếng Việt (mốc nền)")
        + (" | LỌC theo vật thể hiếm" if args.loc_vat_the else "")
        + (" | OCR chỉ token hiếm" if args.loc_token_hiem else "")
        + (" | + nhánh clip_l" if args.clip_l else "")
        + (f" | RERANK top-{args.rerank_so_dau}" if args.rerank else "")
        + "\n"
    )

    print("Bước 1 — tra kho một lần cho mỗi câu, mỗi nhánh")
    bat_dau = time.perf_counter()
    kho_tam = tra_kho_mot_lan(cau_hoi, nguon, args.so_ung_vien)
    print(f"  xong sau {time.perf_counter() - bat_dau:.0f} giây\n")

    if args.rerank:
        tong = THONG_KE_RERANK["da_xep"] + THONG_KE_RERANK["thieu_anh"]
        ti_le = THONG_KE_RERANK["thieu_anh"] / tong if tong else 0.0
        print(
            f"  Rerank: xếp lại {THONG_KE_RERANK['da_xep']:,} ứng viên, "
            f"{THONG_KE_RERANK['thieu_anh']:,} thiếu ảnh ({ti_le:.0%})\n"
        )
        if ti_le > 0.20:
            print(
                "  " + "!" * 68 + "\n"
                f"  {ti_le:.0%} ỨNG VIÊN KHÔNG CÓ ẢNH TRÊN MÁY NÀY.\n\n"
                "  Rerank đẩy ứng viên thiếu ảnh xuống CUỐI, bất kể nó đúng hay sai.\n"
                "  Câu nào có video đáp án không nằm trong shard của máy này thì ứng\n"
                "  viên đúng bị đẩy xuống một cách máy móc.\n\n"
                "  Con số dưới đây ĐO 'MÁY NÀY CÓ ẢNH KHÔNG', KHÔNG đo rerank tốt hay\n"
                "  xấu. ĐỪNG ghi vào sổ điểm.\n\n"
                "  Muốn đo đúng: chạy trên máy giữ shard chứa phần lớn câu dev, hoặc\n"
                "  lọc bộ dev xuống những câu có ảnh rồi truyền qua --tep.\n"
                "  " + "!" * 68 + "\n"
            )

    # -- đo riêng từng nhánh ------------------------------------------------
    print("Bước 2 — điểm của TỪNG nhánh chạy một mình")
    rieng = {}
    for ten in nguon:
        bo = {t: (1.0 if t == ten else 0.0) for t in nguon}
        rieng[ten] = cham_mot_bo(
            kho_tam, cau_hoi, bo, args.cua_so_giay, args.so_dong, args.k_rrf
        )
        print(
            f"  {ten:<9} tất cả {rieng[ten]['tat_ca']:.4f}"
            + "".join(
                f" | {d} {rieng[ten].get(d, 0):.4f}" for d in (KIS, QA)
                if d in rieng[ten]
            )
        )
    print()

    if args.chi_do_rieng:
        return 0

    # -- quét lưới ----------------------------------------------------------
    nhanh_co_that = [t for t in NHANH if t in nguon]
    quanh = None
    if args.min and TEP_TRONG_SO.exists():
        with TEP_TRONG_SO.open("r", encoding="utf-8") as f:
            quanh = (yaml.safe_load(f) or {}).get("mac_dinh")

    cac_bo = list(sinh_luoi(nhanh_co_that, quanh))
    print(f"Bước 3 — quét {len(cac_bo):,} bộ trọng số")

    bang: list[tuple[dict, dict]] = []
    bat_dau = time.perf_counter()
    for i, bo in enumerate(cac_bo, 1):
        bang.append((bo, cham_mot_bo(
            kho_tam, cau_hoi, bo, args.cua_so_giay, args.so_dong, args.k_rrf
        )))
        if i % 100 == 0 or i == len(cac_bo):
            print(f"  {i:,}/{len(cac_bo):,}", end="\r", flush=True)
    print(f"\n  xong sau {time.perf_counter() - bat_dau:.0f} giây\n")

    # -- bộ tham chiếu: chống tinh chỉnh nhiễu -------------------------------
    #
    # Mặt điểm rất phẳng: đã đo thấy KIS 0,4333 với một bộ và 0,4167 với bộ
    # khác hẳn — chênh 0,017, tức DƯỚI một câu trên bộ 12 câu. Chọn bộ cao
    # nhất trong hàng trăm bộ là đang khớp nhiễu.
    #
    # Nên luôn chấm vài bộ ĐƠN GIẢN, có cơ chế giải thích được, rồi so. Hơn
    # nhau dưới sai số thì chọn bộ đơn giản — nó tổng quát tốt hơn trên đề thi.
    print("Bước 4 — so với các bộ đơn giản, giải thích được\n")

    def _bo(**kw):
        return {t: float(kw.get(t, 0.0)) for t in ["clip"] + nhanh_co_that}

    tham_chieu = {
        "chỉ clip": _bo(clip=1),
        "clip + ocr": _bo(clip=1, ocr_fts=1),
        "clip + ocr + asr": _bo(clip=1, ocr_fts=1, asr=1),
        "ba nhánh, bỏ vật thể": _bo(clip=1, ocr_fts=1, asr=0.6),
        "mặc định settings.yaml": _bo(clip=1, ocr_fts=0.6, asr=0.6, object=0.4),
    }

    global _THAM_CHIEU
    _THAM_CHIEU = tham_chieu
    diem_tham_chieu = {}
    for ten, bo in tham_chieu.items():
        d = cham_mot_bo(
            kho_tam, cau_hoi, bo, args.cua_so_giay, args.so_dong, args.k_rrf
        )
        diem_tham_chieu[ten] = d
        print(
            f"  {ten:<24} tất cả {d['tat_ca']:.4f}"
            + "".join(
                f" | {dang} {d.get(dang, 0):.4f}" for dang in (KIS, QA) if dang in d
            )
        )
    print()

    # -- chốt ---------------------------------------------------------------
    chot: dict[str, dict] = {}
    for dang in ("tat_ca", KIS, QA):
        co = [(b, d) for b, d in bang if dang in d]
        if not co:
            continue
        tot = max(co, key=lambda t: t[1][dang])
        chot[dang] = {
            "trong_so": tot[0],
            "diem": tot[1][dang],
            "so_cau": tot[1].get("_so_cau", {}).get(dang),
        }
        print(f"  {dang:<8} {tot[1][dang]:.4f}  {tot[0]}")

    # So THEO TỪNG DẠNG, không so trên cột "tất cả".
    #
    # Bản đầu chỉ so cột "tất cả" và khuyên bộ 'clip + ocr + asr' — bộ KÉM HƠN
    # cách tách theo dạng ở CẢ HAI dạng. Sai vì so nhầm chiều: hệ thống chạy
    # trọng số RIÊNG cho từng dạng (rank/config.trong_so_theo_dang), nên ép
    # một vector chung là tự trói tay.
    print("\n  KHUYẾN NGHỊ THEO TỪNG DẠNG\n")

    khuyen_nghi, _DIEM_CHON = {}, {}
    for dang in (KIS, QA):
        so_cau_dang = (
            chot.get(dang, {}).get("so_cau")
            or diem_tham_chieu[next(iter(diem_tham_chieu))]["_so_cau"].get(dang, 1)
        )
        nguong_dang = 1.0 / max(so_cau_dang, 1)

        quet = chot.get(dang, {})
        diem_quet = quet.get("diem", 0.0)

        # Gom MỌI ứng viên: các bộ tham chiếu VÀ bộ quét được. "Đơn giản" đo
        # bằng SỐ NHÁNH BẬT, không phải bằng việc có tên trong danh sách —
        # bộ quét được cho Q&A ({clip, asr}) vừa ít nhánh hơn 'clip+ocr+asr'
        # vừa điểm cao hơn, mà bản đầu vẫn bỏ qua nó.
        ung_vien = [
            (d.get(dang, 0.0), ten, tham_chieu[ten])
            for ten, d in diem_tham_chieu.items()
        ]
        if quet.get("trong_so"):
            ung_vien.append((diem_quet, "quét được", quet["trong_so"]))

        cao_nhat = max(d for d, _, _ in ung_vien)

        def _so_nhanh(bo):
            return sum(1 for k, v in bo.items() if float(v) > 0)

        # Trong số các bộ KHÔNG thua bộ cao nhất quá một câu, lấy bộ ít nhánh
        # nhất; hoà thì lấy bộ điểm cao hơn.
        du_tot = [t for t in ung_vien if cao_nhat - t[0] < nguong_dang]
        diem_chon, ten_dg, chon_bo = min(
            du_tot, key=lambda t: (_so_nhanh(t[2]), -t[0])
        )
        chon_ten = f"{ten_dg} ({_so_nhanh(chon_bo)} nhánh)"
        chon_diem = diem_chon

        khuyen_nghi[dang] = chon_bo
        _DIEM_CHON[dang] = chon_diem
        print(
            f"  {dang:<6} chọn {chon_ten}: {chon_diem:.4f}\n"
            f"         (cao nhất {cao_nhat:.4f} | một câu = {nguong_dang:.4f} "
            f"| {len(du_tot)}/{len(ung_vien)} bộ nằm trong khoảng đó)\n"
            f"         { {k: v for k, v in chon_bo.items() if float(v) > 0} }"
        )

    uoc_tinh = sum(_DIEM_CHON.get(d, 0.0) for d in (KIS, QA)) / 2
    print(
        f"\n  Trung bình ước tính khi tách theo dạng: {uoc_tinh:.4f}\n"
        f"  So với bộ chung tốt nhất: {chot.get('tat_ca', {}).get('diem', 0):.4f}"
    )

    nen = next((d for b, d in bang if b == {"clip": 1.0, **{t: 0.6 if t in
              ("ocr_fts", "asr") else 0.4 for t in nhanh_co_that}}), None)

    # Chép riêng từng bộ. Dùng chung một đối tượng thì yaml.safe_dump sinh neo
    # &id001/*id001, tệp vẫn đọc được nhưng người mở ra sửa tay sẽ khó hiểu —
    # mà sửa tay đúng là thứ tệp này sinh ra để phục vụ.
    def _chep(bo):
        return {k: float(v) for k, v in bo.items()} if bo else None

    ghi = {
        "mac_dinh": _chep(khuyen_nghi.get(KIS)) or {"clip": 1.0},
        "theo_dang": {
            "kis": _chep(khuyen_nghi.get(KIS)),
            "qa": _chep(khuyen_nghi.get(QA)),
            # TRAKE giữ nguyên bộ mặc định: chưa đo được, đừng để số bịa ở đây.
            "trake": None,
        },
        "ghi_chu": (
            f"Sinh bởi scripts/do_trong_so_rrf.py trên {so_kis_qa} câu KIS/QA. "
            "Q&A đo bằng đáp án đúng điền sẵn — chỉ đo khả năng TÌM ẢNH. "
            "TRAKE chưa đo được (trần 0,000 do keyframe thưa, xem Việc 9)."
        ),
    }

    TEP_TRONG_SO.parent.mkdir(parents=True, exist_ok=True)
    with TEP_TRONG_SO.open("w", encoding="utf-8") as f:
        yaml.safe_dump(ghi, f, allow_unicode=True, sort_keys=False)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with (RUNS_DIR / "do_trong_so_rrf.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "so_cau": len(cau_hoi),
                "nhanh": list(nguon),
                "diem_tung_nhanh_rieng": rieng,
                "chot": chot,
                "diem_bo_dang_dung": nen,
        "bo_tham_chieu": {
            ten: {k: v for k, v in d.items() if not k.startswith("_")}
            for ten, d in diem_tham_chieu.items()
        },
                "toan_bo": [
                    {"trong_so": b, "diem": {k: v for k, v in d.items()
                                             if not k.startswith("_")}}
                    for b, d in sorted(bang, key=lambda t: -t[1]["tat_ca"])[:200]
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nĐã ghi {TEP_TRONG_SO}")
    print(f"Đã ghi {RUNS_DIR / 'do_trong_so_rrf.json'}")

    print(
        "\nĐỪNG chốt bộ trọng số mới nếu điểm chỉ hơn bộ cũ dưới mức sai số của "
        "bộ dev. Với 60 câu, sai số cỡ ±6 điểm phần trăm."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
