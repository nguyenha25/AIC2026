"""
Việc 13 — SINH TỆP NỘP CHO MỌI TRUY VẤN BTC RA. Không bỏ sót câu nào.

CHUYỆN ĐÃ XẢY RA Ở VÒNG p1
--------------------------
BTC ra 25 truy vấn. Nhóm nộp 22. Thiếu câu 8, 14, 25 — cả ba đều KIS, mỗi câu
0 điểm CHẮC CHẮN. Đó là 12% tổng điểm mất trắng TRƯỚC KHI thi đấu.

Và câu 8 với câu 14 có nội dung Y HỆT NHAU — BTC ra trùng một đề. Nộp cùng
một đáp án cho cả hai là hoàn toàn hợp lệ, tức nhóm bỏ không hai suất chỉ vì
thấy đề lặp lại.

NGUYÊN TẮC CỦA SCRIPT NÀY
-------------------------
Quét THƯ MỤC ĐỀ của BTC, không quét thư mục kết quả. Mỗi tệp đề phải ra đúng
một tệp nộp — kể cả khi mạch tìm kiếm hỏng, kể cả khi không tìm được gì.

Không tìm được thì nộp 100 dòng đoán, lấy từ frame_map thật. Đoán sai chỉ mất
đúng câu đó; KHÔNG NỘP thì cũng mất đúng câu đó, nhưng bỏ mất luôn cơ hội.
Nộp bừa không bao giờ tệ hơn không nộp.

ĐỀ TRÙNG NHAU
-------------
Hai đề cùng nội dung thì tra kho MỘT LẦN rồi chép kết quả sang cả hai tệp.
Script tự phát hiện và báo, không cần ai để ý.

CÁCH CHẠY
---------
    python -u -m scripts.tao_bo_nop --de D:\\aic-data\\de\\SOTUYEN1 --ra p1
    python -u -m scripts.tao_bo_nop --de ... --ra p1 --thu-nghiem   # không tra kho
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
import zipfile
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

from aic2026.paths import SUBMISSIONS_DIR  # noqa: E402
from aic2026.submit import KIS, QA, TRAKE, Answer, SubmissionBudget  # noqa: E402

SO_DONG = 100

# query-p1-16-trake.txt -> ("p1", 16, "trake")
MAU_TEN_DE = re.compile(
    r"^query-(?P<dot>[^-]+)-(?P<so>\d+)-(?P<dang>kis|qa|trake)$", re.IGNORECASE
)

DANG_BTC = {"kis": KIS, "qa": QA, "trake": TRAKE}


def doc_thu_muc_de(thu_muc: Path) -> list[dict]:
    """Mọi tệp đề trong thư mục, đã sắp theo số truy vấn."""
    de = []
    for tep in sorted(thu_muc.glob("*.txt")):
        khop = MAU_TEN_DE.match(tep.stem)
        if not khop:
            print(f"  bỏ qua {tep.name} — tên không theo khuôn query-<đợt>-<số>-<dạng>")
            continue
        de.append(
            {
                "ten_tep": tep.stem,
                "so": int(khop["so"]),
                "dang": khop["dang"].lower(),
                "cau_hoi": tep.read_text(encoding="utf-8").strip(),
            }
        )
    return sorted(de, key=lambda d: d["so"])


def van_tay(cau_hoi: str) -> str:
    """Dấu vân tay nội dung đề, để bắt hai đề trùng nhau."""
    gon = re.sub(r"\s+", " ", cau_hoi).strip().lower()
    return hashlib.sha1(gon.encode("utf-8")).hexdigest()[:12]


def so_moc_trake(cau_hoi: str) -> int:
    """Đếm mốc từ đề. BTC đánh dấu E1/E2/E3 ở đầu dòng."""
    from aic2026.trake_align import MAU_DANH_DAU_DONG

    n = len(MAU_DANH_DAU_DONG.findall(cau_hoi))
    return n if n >= 2 else 3


# ---------------------------------------------------------------------------
# Dòng đoán — dùng khi mạch không trả về gì
# ---------------------------------------------------------------------------

def dong_doan(so_moc: int = 1) -> list[Answer]:
    """100 dòng lấy từ frame_map THẬT, rải đều khắp kho.

    Không bịa số: mọi frame_idx đều là keyframe có thật, nên tệp vẫn hợp lệ
    và vẫn có cơ hội trúng nếu may.
    """
    from aic2026.frame_map import load_frame_map

    bang = load_frame_map()
    buoc = max(1, len(bang) // (SO_DONG * so_moc))
    mau = bang.iloc[:: buoc].head(SO_DONG * so_moc)

    ra = []
    for i in range(SO_DONG):
        lat = mau.iloc[i * so_moc: (i + 1) * so_moc]
        if len(lat) < so_moc:
            break
        ra.append(
            Answer(
                video_id=str(lat.iloc[0]["video_id"]),
                frame_ids=sorted(int(r) for r in lat["frame_idx"]),
                answer="khong ro" if so_moc == 1 else None,
            )
        )
    return ra


# ---------------------------------------------------------------------------
# Tra kho
# ---------------------------------------------------------------------------

def tra_mot_de(de: dict, tim, kho_chu=None) -> tuple[list[Answer], str]:
    """Trả về (danh sách Answer, ghi chú). Không bao giờ ném lỗi ra ngoài."""
    dang = de["dang"]

    try:
        hits = tim(de["cau_hoi"], SO_DONG * 3)
    except Exception as loi:
        return dong_doan(so_moc_trake(de["cau_hoi"]) if dang == "trake" else 1), (
            f"mạch lỗi ({type(loi).__name__}: {str(loi)[:60]}) -> dòng đoán"
        )

    if not hits:
        return dong_doan(so_moc_trake(de["cau_hoi"]) if dang == "trake" else 1), (
            "mạch không trả về gì -> dòng đoán"
        )

    if dang == "trake":
        so_moc = so_moc_trake(de["cau_hoi"])
        ra = []
        for h in hits[:SO_DONG]:
            # Mốc tăng dần quanh ứng viên. Vòng p1 nộp 36/100 dòng có mốc KHÔNG
            # tăng dần — sự kiện là chuỗi thời gian nên những dòng đó chắc chắn
            # sai, tức chỗ trống lãng phí.
            goc = int(h.frame_idx)
            ra.append(
                Answer(
                    video_id=str(h.video_id),
                    frame_ids=[goc + j * 25 for j in range(so_moc)],
                )
            )
        return ra, f"{len(ra)} dòng, {so_moc} mốc/dòng"

    if dang == "qa":
        from aic2026.qa_answer import doan_du_phong, tra_loi

        try:
            d = tra_loi(de["cau_hoi"], hits[:5])
            dap_an, nguon = d.van_ban, d.nguon
        except Exception as loi:
            d = doan_du_phong(de["cau_hoi"])
            dap_an, nguon = d.van_ban, f"lỗi {type(loi).__name__} -> dự phòng"

        # RẢI BIẾN THỂ THEO HẠNG, không xoay vòng đều.
        #
        # Vòng p1 rải xoay vòng: khung 1 lấy biến thể A, khung 2 lấy B... nên
        # khung ĐÚNG chỉ được ghép với MỘT biến thể — 1/3 cơ hội. Điểm cuối là
        # trung bình R@1..R@100 nên hạng đầu đáng giá hơn hẳn; đặt khung tốt
        # nhất × MỌI biến thể ở hạng 1-3 chỉ tốn 3 suất trong 100.
        from aic2026.qa_answer import rai_theo_hang

        cap = rai_theo_hang(hits, dap_an, de["cau_hoi"], SO_DONG)
        ra = [
            Answer(str(h.video_id), [int(h.frame_idx)], answer=a) for h, a in cap
        ]
        so_bien_the = len({a for _, a in cap})
        return ra, (
            f"{len(ra)} dòng, {so_bien_the} biến thể của {dap_an!r} "
            f"(nguồn {nguon})"
        )

    ra = [Answer(str(h.video_id), [int(h.frame_idx)]) for h in hits[:SO_DONG]]
    return ra, f"{len(ra)} dòng"


def dung_tim():
    from aic2026.index.fts_index import TextSearchIndex
    from aic2026.paths import FTS_DIR
    from aic2026.rank.hop_nhat import tim_ung_vien_gop

    kho_chu = TextSearchIndex(FTS_DIR / "text.sqlite")
    return {
        dang: tim_ung_vien_gop(
            kho_chu=kho_chu,
            dung_clip=True,
            dung_ocr_fts=True,
            dung_asr=True,
            mo_rong_truy_van=True,
            dang_cau=dang,
        )
        for dang in ("kis", "qa", "trake")
    }


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Việc 13 — sinh tệp nộp cho MỌI đề")
    p.add_argument("--de", required=True, help="thư mục chứa các tệp đề .txt")
    p.add_argument("--ra", default="nop", help="tên thư mục kết quả")
    p.add_argument("--thu-nghiem", action="store_true",
                   help="không tra kho, chỉ sinh dòng đoán — để kiểm định dạng")
    args = p.parse_args()

    thu_muc_de = Path(args.de)
    if not thu_muc_de.is_dir():
        print(f"Không thấy thư mục đề: {thu_muc_de}")
        return 1

    de = doc_thu_muc_de(thu_muc_de)
    if not de:
        print(f"Không thấy tệp đề nào trong {thu_muc_de}")
        return 1

    print(f"BTC ra {len(de)} truy vấn: {[d['so'] for d in de]}")

    # Đề trùng nội dung — tra một lần, chép sang cả hai.
    theo_van_tay: dict[str, list[dict]] = {}
    for d in de:
        theo_van_tay.setdefault(van_tay(d["cau_hoi"]), []).append(d)

    trung = {k: v for k, v in theo_van_tay.items() if len(v) > 1}
    if trung:
        print("\nĐỀ TRÙNG NỘI DUNG — tra một lần, nộp cho tất cả:")
        for nhom in trung.values():
            print(f"  câu {[x['so'] for x in nhom]}: {nhom[0]['cau_hoi'][:70]}…")

    goc = SUBMISSIONS_DIR / args.ra
    if goc.exists():
        shutil.rmtree(goc)
    (goc / "submission").mkdir(parents=True)

    tim = None if args.thu_nghiem else dung_tim()

    print(f"\n{'câu':<5}{'dạng':<7}{'tệp':<26}ghi chú")
    nho: dict[str, list[Answer]] = {}
    that_bai = []

    for d in de:
        vt = van_tay(d["cau_hoi"])
        khoa = f"{vt}:{d['dang']}"

        if khoa in nho:
            cau_tra_loi, ghi_chu = nho[khoa], f"chép từ đề trùng ({len(nho[khoa])} dòng)"
        elif args.thu_nghiem:
            cau_tra_loi = dong_doan(
                so_moc_trake(d["cau_hoi"]) if d["dang"] == "trake" else 1
            )
            ghi_chu = "thử nghiệm — dòng đoán"
        else:
            cau_tra_loi, ghi_chu = tra_mot_de(d, tim[d["dang"]])
            nho[khoa] = cau_tra_loi

        so_moc = so_moc_trake(d["cau_hoi"]) if d["dang"] == "trake" else 1

        ngan_sach = SubmissionBudget(task=DANG_BTC[d["dang"]])
        for a in cau_tra_loi:
            ngan_sach.add(a)

        # LUẬT CỨNG: mọi đề phải ra MỘT tệp ĐỦ 100 DÒNG.
        #
        # Đếm theo số ngân sách THẬT SỰ NHẬN, không đếm danh sách đưa vào.
        # Ngân sách khử trùng theo (video, frame, đáp án đã hạ chữ thường), nên
        # một phần dòng có thể bị bỏ mà phía gọi không biết — đã đo: ba tệp chỉ
        # ra 98/100 dòng.
        #
        # BTC chấm theo trung bình R@1..R@100, nên dòng hạng 100 vẫn đáng 1/5
        # điểm câu đó. Không có lý do nào để nộp thiếu dòng.
        if len(ngan_sach) < SO_DONG:
            thieu_truoc = SO_DONG - len(ngan_sach)
            dap_an_qa = (
                cau_tra_loi[0].answer if d["dang"] == "qa" and cau_tra_loi else None
            )
            for a in dong_doan(so_moc):
                if len(ngan_sach) >= SO_DONG:
                    break
                ngan_sach.add(
                    Answer(a.video_id, a.frame_ids, answer=dap_an_qa)
                    if d["dang"] == "qa" else a
                )
            if len(ngan_sach) < SO_DONG:
                ghi_chu += f" | CHỈ {len(ngan_sach)}/{SO_DONG} dòng"
            elif thieu_truoc:
                ghi_chu += f" | lấp {thieu_truoc} dòng đoán"

        dich = goc / "submission" / f"{d['ten_tep']}.csv"
        try:
            ngan_sach.write(dich)
        except Exception as loi:
            that_bai.append((d["ten_tep"], str(loi)[:80]))
            ghi_chu += f" | GHI HỎNG: {str(loi)[:50]}"

        print(f"{d['so']:<5}{d['dang']:<7}{dich.name:<26}{ghi_chu}")

    # --- soát lại: đủ tệp chưa ---------------------------------------------
    print("\n" + "=" * 72)
    da_co = {t.stem for t in (goc / "submission").glob("*.csv")}
    thieu = [d["ten_tep"] for d in de if d["ten_tep"] not in da_co]

    if thieu:
        print(f"THIẾU {len(thieu)} TỆP: {thieu}")
        print("KHÔNG đóng gói. Mỗi tệp thiếu là 0 điểm chắc chắn.")
        for ten, ly_do in that_bai:
            print(f"  {ten}: {ly_do}")
        return 1

    print(f"ĐỦ {len(da_co)}/{len(de)} tệp.")

    thieu_dong = [
        (t.name, n)
        for t in sorted((goc / "submission").glob("*.csv"))
        if (n := sum(1 for _ in t.open("r", encoding="utf-8") if _.strip())) < SO_DONG
    ]
    if thieu_dong:
        print(f"\n{len(thieu_dong)} tệp chưa đủ {SO_DONG} dòng — bỏ phí suất:")
        for ten, n in thieu_dong[:10]:
            print(f"  {ten}: {n} dòng")

    tep_zip = goc.with_suffix(".zip")
    with zipfile.ZipFile(tep_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for t in sorted((goc / "submission").glob("*.csv")):
            # BTC đòi thư mục submission/ NẰM TRONG zip.
            z.write(t, f"submission/{t.name}")

    print(f"\nĐã đóng gói: {tep_zip}  ({tep_zip.stat().st_size / 1024:.0f} KB)")
    print("\nSoát từng tệp trước khi nộp:")
    print(f"  Get-ChildItem '{goc / 'submission'}\\*.csv' | "
          "ForEach-Object { python -m scripts.check_submission $_.FullName }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
