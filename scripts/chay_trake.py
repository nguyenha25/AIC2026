"""
Việc 12 — CHẠY GHÉP THỜI GIAN TRAKE VÀ SO KHUNG THƯA VỚI KHUNG DÀY.

CÂU HỎI SCRIPT NÀY TRẢ LỜI
--------------------------
"Trần 0,000 của TRAKE có đúng là do keyframe thưa không, và trích khung dày
có gỡ được nó không?"

Cách trả lời: chạy ĐÚNG MỘT thuật toán ghép, trên ĐÚNG MỘT video, chỉ đổi
nguồn khung hình — thưa rồi dày. Mọi thứ khác giữ nguyên. Chênh lệch đo được
vì thế là chênh lệch của TẦNG DỮ LIỆU, không lẫn thứ gì khác.

ĐÂY LÀ BẰNG CHỨNG, KHÔNG PHẢI PHÉP ĐO
-------------------------------------
Chỉ hai câu TRAKE có video gốc trên máy này. Hai câu KHÔNG ước lượng được
TRAKE sẽ tăng bao nhiêu trên đề thật. Nhưng một câu nhích khỏi 0,000 là đủ
để kết luận chẩn đoán đúng và hướng đi đúng — đó là câu hỏi có/không, không
phải câu hỏi bao nhiêu.

Ghi vào sổ điểm đúng như vậy. ĐỪNG ghi "TRAKE tăng X%".

--dung-video-dung LÀM GÌ
------------------------
Lấy thẳng video_id từ đáp án thay vì để nhánh tìm kiếm chọn. Cố ý: nó TÁCH
câu hỏi "ghép thời gian có đúng không" khỏi câu hỏi "tìm đúng video không".
Sai video là 0 điểm ngay, nên trộn hai câu hỏi lại thì một lỗi tìm kiếm sẽ
che mất kết quả ghép.

Bỏ cờ này đi thì đo cả đường đầu-cuối — làm sau, khi đã trả lời xong câu hỏi
tầng dữ liệu.

CÁCH CHẠY
---------
    python -u -m scripts.chay_trake --dung-video-dung
    python -u -m scripts.chay_trake --dung-video-dung --bo-ma-hoa lon
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

from aic2026.eval import compute_final_score        # noqa: E402
from aic2026.paths import (                         # noqa: E402
    DEV_QUERIES_PATH,
    FRAMES_DENSE_DIR,
    RUNS_DIR,
    video_file,
)
from aic2026.submit import TRAKE                    # noqa: E402
from aic2026.trake_align import ghep                # noqa: E402


# Mã mà việc này cần, kèm gói bàn giao chứa nó. Kiểm TRƯỚC khi làm gì —
# chết giữa chừng thì đã trích ảnh, đã nạp mô hình, và bảng tổng kết in ra
# trống trơn trông như vừa đo xong.
PHU_THUOC = [
    ("aic2026.trake_align", "03 — Nguyên"),
    ("aic2026.query_expand", "02 — Thi (Việc 5)"),
]


def kiem_phu_thuoc(bo_ma_hoa: str) -> list[str]:
    """Danh sách mã còn THIẾU TỆP. Rỗng nghĩa là đủ tệp để chạy.

    Dùng find_spec chứ không import thật: import thật sẽ kéo theo torch và
    open_clip, nên máy thiếu thư viện cũng bị báo là thiếu tệp — hai chuyện
    khác hẳn nhau, và chữa theo hai cách khác hẳn nhau.
    """
    import importlib.util

    can = list(PHU_THUOC)
    if bo_ma_hoa == "lon":
        can.append(("aic2026.rerank", "02 — Thi (Việc 4)"))

    thieu = []
    for ten, goi in can:
        try:
            co = importlib.util.find_spec(ten) is not None
        except (ImportError, ValueError):
            co = False
        if not co:
            thieu.append(f"  {ten:<38} -> giải nén gói {goi}")
    return thieu


def doc_cau_trake(duong_dan: Path) -> list[dict]:
    ra = []
    for dong in duong_dan.open("r", encoding="utf-8"):
        dong = dong.strip()
        if not dong:
            continue
        q = json.loads(dong)
        if q.get("loai_truy_van") == "chuoi_su_kien":
            ra.append(q)
    return ra


def cham(q: dict, frame_ids: list[int]) -> float:
    """Điểm cuối của MỘT câu TRAKE theo đúng công thức BTC."""
    gt = {
        "gt_video_id": q["video_id"],
        "gt_events": [
            [int(g["frame_start"]), int(g["frame_end"])] for g in q["cac_giai_doan"]
        ],
    }
    dong = [{"video_id": q["video_id"], "frame_ids": frame_ids}]
    return compute_final_score(gt, dong, TRAKE)["final_score"]


def chi_tiet_tung_moc(q: dict, frame_ids: list[int]) -> list[str]:
    """Mốc nào trúng, mốc nào trượt và trượt bao xa."""
    ra = []
    for j, g in enumerate(q["cac_giai_doan"]):
        s, e = int(g["frame_start"]), int(g["frame_end"])
        if j >= len(frame_ids):
            ra.append(f"     mốc {j + 1}: THIẾU (cần khoảng {s}-{e})")
            continue
        f = frame_ids[j]
        if s <= f <= e:
            ra.append(f"     mốc {j + 1}: TRÚNG  {f} trong [{s}, {e}]")
        else:
            xa = s - f if f < s else f - e
            ra.append(f"     mốc {j + 1}: trượt  {f} ngoài [{s}, {e}], lệch {xa} khung")
    return ra


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tep", default=str(DEV_QUERIES_PATH))
    p.add_argument("--dung-video-dung", action="store_true")
    p.add_argument("--bo-ma-hoa", choices=["b32", "lon"], default="b32")
    p.add_argument("--so-moc", type=int, default=None)
    p.add_argument("--nguon-mo-rong", choices=["tu_dien", "marian", "llm"],
                   default=None, help="ghi đè mo_rong_truy_van.nguon")
    p.add_argument("--chan-doan", action="store_true",
                   help="in cụm tiếng Anh và 5 khung điểm cao nhất của từng sự kiện")
    args = p.parse_args()

    if args.nguon_mo_rong:
        import os

        os.environ["AIC_NGUON_MO_RONG"] = args.nguon_mo_rong

    thieu = kiem_phu_thuoc(args.bo_ma_hoa)
    if thieu:
        print("THIẾU MÃ — chưa chạy được:\n")
        print("\n".join(thieu))
        print(
            "\nViệc 12 dùng chung bộ mở rộng truy vấn của Việc 5: nhánh CLIP cần "
            "câu TIẾNG ANH.\nKhông có nó thì CLIP nhận nguyên câu tiếng Việt — "
            "vẫn chạy, nhưng đó là MỐC SÀN,\nvà điểm thấp sẽ bị đọc nhầm thành "
            "'khung dày không giúp gì'. Nên script dừng ở đây\nthay vì lặng lẽ "
            "đo sai."
        )
        return 1

    cau = doc_cau_trake(Path(args.tep))
    lam_duoc = [
        q for q in cau
        if (FRAMES_DENSE_DIR / q["video_id"]).is_dir() or video_file(q["video_id"]).exists()
    ]

    print(f"{len(cau)} câu TRAKE trong bộ dev, {len(lam_duoc)} câu có dữ liệu trên máy này")
    if not lam_duoc:
        print("\nKhông câu nào chạy được. Chạy Việc 9 trước:")
        print("  python -u -m scripts.trich_khung_day --tu-dev")
        return 1

    if not args.dung_video_dung:
        print(
            "\nCHƯA hỗ trợ chọn video bằng nhánh tìm kiếm trong script này.\n"
            "Dùng --dung-video-dung để đo RIÊNG phần ghép thời gian."
        )
        return 1

    from aic2026.query_expand import nguon_mac_dinh

    dang_dung = nguon_mac_dinh()
    if args.nguon_mo_rong and dang_dung != args.nguon_mo_rong:
        print(
            f"Yêu cầu --nguon-mo-rong {args.nguon_mo_rong} nhưng đang chạy "
            f"{dang_dung!r}. Cờ không ăn — dừng thay vì đo nhầm nhánh."
        )
        return 1

    # Thử dịch MỘT câu trước khi làm gì. Nhánh llm/marian hỏng thì phải biết
    # ngay, chứ không phải sau 60 giây chạy và một bảng số nhìn như thật.
    if args.nguon_mo_rong:
        from aic2026.query_expand import LuiNhanhKhongMongMuon, mo_rong

        try:
            thu = mo_rong("một người đàn ông cầm ô", bat_buoc=True)
        except LuiNhanhKhongMongMuon as loi:
            print(f"NHÁNH DỊCH {args.nguon_mo_rong.upper()} KHÔNG DÙNG ĐƯỢC\n")
            print(f"  {loi}\n")
            if args.nguon_mo_rong == "llm":
                # Lý do cụ thể đã nằm trong chính thông báo lỗi ở trên; đừng
                # in thêm danh sách việc chung chung mà người chạy có thể đã
                # làm xong, vì nó dẫn đi sai hướng.
                pass
            else:
                print(
                    "  Nhánh marian chạy LOCAL, không cần khoá API. Lần đầu nó "
                    "tải mô hình\n  VietAI/envit5-translation (~900 MB) — cần "
                    "mạng ở lần chạy đó thôi.\n"
                    "  Thiếu thư viện thì: pip install transformers\n"
                )
            print(
                "  Dừng ở đây thay vì lặng lẽ chạy tu_dien rồi in ra một bảng số\n"
                "  trông như đã đo xong nhánh này."
            )
            return 1

        print(f"Thử dịch: 'một người đàn ông cầm ô' -> {thu.cum_chinh!r}")

    print(f"Bộ mã hoá: {args.bo_ma_hoa} | nguồn mở rộng: {dang_dung}\n")
    print("=" * 78)

    bao_cao = []

    for q in lam_duoc:
        qid, vid = str(q["id"]), str(q["video_id"])
        so_moc = args.so_moc or len(q["cac_giai_doan"])
        print(f"\nCâu {qid} — {vid} — {so_moc} mốc")
        print(f"  {q.get('cau_hoi', '')[:100]}")

        ket, chi_tiet_chay = {}, {}
        for nhan, dung_day in (("khung thưa (BTC)", False), ("khung dày (Việc 9)", True)):
            if dung_day and not (FRAMES_DENSE_DIR / vid).is_dir():
                print(f"\n  {nhan}: chưa trích, bỏ qua")
                continue

            bat_dau = time.perf_counter()
            try:
                r = ghep(
                    q.get("cau_hoi", ""),
                    ung_vien=[],
                    so_moc=so_moc,
                    video_id=vid,
                    uu_tien_khung_day=dung_day,
                    bo_ma_hoa=args.bo_ma_hoa,
                )
            except ImportError as loi:
                # Thiếu mã chứ không phải sai dữ liệu — dừng hẳn, đừng chạy tiếp
                # rồi in bảng tổng kết trống làm người đọc tưởng đã đo xong.
                ten = getattr(loi, "name", "") or ""
                if ten.startswith("aic2026"):
                    print(f"\n  THIẾU TỆP MÃ: {loi}\n  Xem thứ tự trộn gói ở README_BAN_GIAO.md")
                else:
                    print(
                        f"\n  THIẾU THƯ VIỆN: {loi}\n"
                        f"  Cài bằng:  pip install {ten or '<tên gói>'}\n"
                        "  Hoặc:      pip install -r requirements.txt -c constraints.txt"
                    )
                raise SystemExit(1)
            except Exception as loi:
                import traceback

                print(f"\n  {nhan}: lỗi — {type(loi).__name__}: {loi}")
                traceback.print_exc(limit=3)
                continue

            diem = cham(q, r.frame_ids)
            ket[nhan] = diem
            chi_tiet_chay[nhan] = r
            print(
                f"\n  {nhan}: khoảng cách {r.khoang_cach_khung_giay:.3f}s | "
                f"{r.so_khung_ung_vien:,} khung ứng viên | "
                f"độ trải điểm {r.do_trai_diem:.3f} | "
                f"{time.perf_counter() - bat_dau:.0f}s"
            )
            print(f"     điểm: {diem:.4f}")
            for d in chi_tiet_tung_moc(q, r.frame_ids):
                print(d)
            for c in r.canh_bao:
                print(f"     ! {c}")

            if args.chan_doan:
                print("\n     -- chẩn đoán chấm điểm sự kiện --")
                for i, (sk, cum) in enumerate(zip(r.su_kien, r.cum_tieng_anh)):
                    print(f"     sự kiện {i + 1}: {sk}")
                    print(f"        -> tiếng Anh: {cum!r}")
                    dinh = r.dinh_moi_su_kien[i] if i < len(r.dinh_moi_su_kien) else []
                    print(
                        "        -> 5 khung điểm cao nhất: "
                        + ", ".join(f"{f}({d:.3f})" for f, d in dinh)
                    )
                if len(set(r.cum_tieng_anh)) < len(r.cum_tieng_anh):
                    print(
                        "\n     ĐÂY LÀ NGUYÊN NHÂN: hai sự kiện trở lên rút về CÙNG "
                        "một cụm tiếng Anh.\n     Ma trận điểm có các hàng giống hệt "
                        "nhau, nên quy hoạch động chỉ lấy được\n     đỉnh cao nhất "
                        "cùng vài khung kề bên — đúng như thứ đang thấy ở trên."
                    )

        if len(ket) == 2:
            (thua, day), (r_thua, r_day) = list(ket.values()), list(chi_tiet_chay.values())
            bao_cao.append(
                {
                    "id": qid,
                    "video_id": vid,
                    "so_moc": so_moc,
                    "thua": thua,
                    "day": day,
                    "moc_trung_thua": round(thua * so_moc),
                    "moc_trung_day": round(day * so_moc),
                    "khoang_thua": r_thua.khoang_cach_khung_giay,
                    "do_trai_thua": r_thua.do_trai_diem,
                    "do_trai_day": r_day.do_trai_diem,
                }
            )

    print("\n" + "=" * 78)
    print("\nTỔNG KẾT\n")
    print(f"  {'câu':<6}{'video':<12}{'khung thưa':>12}{'khung dày':>12}{'chênh':>10}")
    for b in bao_cao:
        print(
            f"  {b['id']:<6}{b['video_id']:<12}{b['thua']:>12.4f}"
            f"{b['day']:>12.4f}{b['day'] - b['thua']:>+10.4f}"
        )

    if bao_cao:
        tb_thua = sum(b["thua"] for b in bao_cao) / len(bao_cao)
        tb_day = sum(b["day"] for b in bao_cao) / len(bao_cao)
        print(f"\n  trung bình{'':<8}{tb_thua:>12.4f}{tb_day:>12.4f}{tb_day - tb_thua:>+10.4f}")

        so_moc_tong = sum(b["so_moc"] for b in bao_cao)
        moc_thua = sum(b["moc_trung_thua"] for b in bao_cao)
        moc_day = sum(b["moc_trung_day"] for b in bao_cao)

        print(f"\n  Mốc trúng: khung thưa {moc_thua}/{so_moc_tong}, "
              f"khung dày {moc_day}/{so_moc_tong}")
        print(f"  Một mốc đổi chiều = {1 / so_moc_tong:.3f} điểm trung bình.")

        print("\n  ĐỌC KẾT QUẢ NÀY THẾ NÀO\n")
        if moc_day > moc_thua:
            print(
                f"  Khung dày hơn khung thưa {moc_day - moc_thua} mốc "
                f"trên tổng {so_moc_tong}.\n"
                "  KHÔNG kết luận được 'Việc 9 gỡ được trần' từ chừng này. "
                f"Với {so_moc_tong} mốc,\n  một mốc đổi chiều đã làm trung bình "
                f"nhảy {1 / so_moc_tong:.3f} — bằng đúng độ lớn đang thấy.\n"
            )
        elif moc_day == moc_thua:
            print("  Khung dày KHÔNG hơn khung thưa.\n")
        else:
            print("  Khung dày KÉM HƠN khung thưa — bất thường, xem lại phần ghép.\n")

        print(
            "  Điều DUY NHẤT kết luận được chắc chắn: keyframe thưa cách nhau "
            f"{bao_cao[0]['khoang_thua']:.1f}-{bao_cao[-1]['khoang_thua']:.1f} giây\n"
            "  thì về mặt toán học không thể rơi trúng cửa sổ đáp án hẹp. Khung "
            "dày là ĐIỀU KIỆN CẦN.\n  Việc nó có ĐỦ hay không thì bộ dữ liệu này "
            "chưa trả lời được.\n"
        )

        trai = [b["do_trai_day"] for b in bao_cao if b.get("do_trai_day")]
        if trai and max(trai) < 0.08:
            print(
                f"  ĐỘ TRẢI ĐIỂM CHỈ {max(trai):.3f} — đây mới là nút thắt hiện tại.\n"
                "  Điểm CLIP giữa khung tốt nhất và khung tệ nhất chênh nhau chừng đó\n"
                "  nghĩa là mô hình nhìn mọi khung trong cảnh gần như giống hệt nhau,\n"
                "  và thứ hạng phía sau chủ yếu là nhiễu. Trước khi trích dày thêm\n"
                "  video khác, thử --nguon-mo-rong llm rồi --bo-ma-hoa lon."
            )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    dich = RUNS_DIR / "chay_trake.json"
    with dich.open("w", encoding="utf-8") as f:
        json.dump(
            {"bo_ma_hoa": args.bo_ma_hoa, "so_cau": len(bao_cao), "chi_tiet": bao_cao},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nĐã ghi {dich}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
