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

def dong_doan(so_moc: int = 1, so_dong: int = SO_DONG) -> list[Answer]:
    """Các dòng lấy từ frame_map THẬT, rải đều khắp kho.

    Không bịa số: mọi frame_idx đều là keyframe có thật, nên tệp vẫn hợp lệ
    và vẫn có cơ hội trúng nếu may.
    """
    from aic2026.frame_map import load_frame_map

    bang = load_frame_map()

    # TRAKE: mọi mốc trên MỘT dòng bắt buộc thuộc CÙNG video. Bản cũ lấy
    # từng lát liên tiếp của toàn frame_map; lát cắt có thể đi qua ranh giới
    # video nhưng lại gắn video_id của dòng đầu cho tất cả frame, khiến các
    # frame còn lại không tồn tại dưới video_id đó.
    if so_moc > 1:
        bang_xep = bang.sort_values(["video_id", "pts_time"])
        theo_video: list[tuple[str, list[int]]] = []
        for video_id, nhom in bang_xep.groupby("video_id", sort=False):
            frame_ids = [int(x) for x in nhom["frame_idx"]]
            if len(frame_ids) >= so_moc:
                theo_video.append((str(video_id), frame_ids))

        ra: list[Answer] = []
        bat_dau = 0
        while len(ra) < so_dong:
            them_duoc = False
            for video_id, frame_ids in theo_video:
                chon = frame_ids[bat_dau: bat_dau + so_moc]
                if len(chon) < so_moc:
                    continue
                ra.append(Answer(video_id=video_id, frame_ids=chon))
                them_duoc = True
                if len(ra) >= so_dong:
                    break
            if not them_duoc:
                break
            bat_dau += 1
        return ra

    buoc = max(1, len(bang) // (so_dong * so_moc))
    mau = bang.iloc[:: buoc].head(so_dong * so_moc)

    ra = []
    for i in range(so_dong):
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


def bang_tra_nguoc_frame_map() -> dict[tuple[str, int], float]:
    """(video_id, frame_idx) -> pts_time cho cổng thoát số 5."""
    from aic2026.frame_map import load_frame_map

    bang = load_frame_map()
    return {
        (str(v), int(f)): float(t)
        for v, f, t in zip(bang["video_id"], bang["frame_idx"], bang["pts_time"])
    }


def loc_cua_so_cau_tra_loi(
    cau_tra_loi: list[Answer],
    dang: str,
    tra_nguoc: dict[tuple[str, int], float] | None = None,
    cua_so: float | None = None,
) -> tuple[list[Answer], int, int]:
    """Giữ hạng tốt nhất và chặn cặp KIS/QA cùng video quá gần.

    Trả về ``(dòng giữ lại, số bỏ vì gần, số bỏ vì frame không tồn tại)``.
    TRAKE không áp dụng cửa sổ 10 giây nhưng vẫn được giữ nguyên ở đây.
    """
    if dang == "trake":
        return list(cau_tra_loi), 0, 0

    from aic2026.rank.config import cua_so_giay
    from aic2026.rank.dedupe import EPSILON

    tra = tra_nguoc if tra_nguoc is not None else bang_tra_nguoc_frame_map()
    nguong = cua_so_giay() if cua_so is None else float(cua_so)
    da_giu: dict[str, list[float]] = {}
    ra: list[Answer] = []
    bo_gan = 0
    bo_khong_tra = 0

    for a in cau_tra_loi:
        pts_time = tra.get((str(a.video_id), int(a.frame_ids[0])))
        if pts_time is None:
            bo_khong_tra += 1
            continue

        moc_cu = da_giu.setdefault(str(a.video_id), [])
        if any(abs(float(pts_time) - t) < nguong - EPSILON for t in moc_cu):
            bo_gan += 1
            continue

        moc_cu.append(float(pts_time))
        ra.append(a)

    return ra, bo_gan, bo_khong_tra


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
        ra: list[Answer] = []
        ghi_chu_ghep = ""

        # Việc 12: chọn video mạnh nhất rồi căn các sự kiện bằng DP trên ảnh
        # thật, ưu tiên frames_dense. Bản cũ chỉ bịa [gốc, gốc+25, ...], các
        # số đó thậm chí không chắc là keyframe có thật.
        try:
            from aic2026.trake_align import ghep

            ket = ghep(
                de["cau_hoi"], hits, so_moc=so_moc,
                uu_tien_khung_day=True, bo_ma_hoa="b32",
            )
            if (
                len(ket.frame_ids) == so_moc
                and ket.frame_ids == sorted(ket.frame_ids)
                and len(set(ket.frame_ids)) == so_moc
            ):
                ra.append(Answer(str(ket.video_id), list(ket.frame_ids)))
                ghi_chu_ghep = (
                    f"ghep={ket.video_id}, "
                    f"{'dense' if ket.du_day else 'thua'}"
                )
        except Exception as loi:
            # Máy không có ảnh của video ứng viên vẫn phải sinh tệp hợp lệ.
            ghi_chu_ghep = f"ghep lỗi {type(loi).__name__} -> keyframe ứng viên"

        # Các dòng còn lại dùng KEYFRAME CÓ THẬT của từng video ứng viên. Lấy
        # các cửa sổ trượt theo hạng rồi sắp thời gian; tuyệt đối không cộng
        # một offset fps giả vào frame_idx.
        theo_video: dict[str, list] = {}
        thu_tu_video: list[str] = []
        for h in hits:
            vid = str(h.video_id)
            if vid not in theo_video:
                theo_video[vid] = []
                thu_tu_video.append(vid)
            if all(int(x.frame_idx) != int(h.frame_idx) for x in theo_video[vid]):
                theo_video[vid].append(h)

        da_co = {
            (a.video_id, tuple(a.frame_ids))
            for a in ra
        }
        for vid in thu_tu_video:
            ds = theo_video[vid]
            for bat_dau in range(max(0, len(ds) - so_moc + 1)):
                chon = sorted(ds[bat_dau: bat_dau + so_moc], key=lambda h: h.pts_time)
                frame_ids = [int(h.frame_idx) for h in chon]
                khoa = (vid, tuple(frame_ids))
                if (
                    len(frame_ids) == so_moc
                    and frame_ids == sorted(frame_ids)
                    and len(set(frame_ids)) == so_moc
                    and khoa not in da_co
                ):
                    da_co.add(khoa)
                    ra.append(Answer(vid, frame_ids))
                if len(ra) >= SO_DONG:
                    break
            if len(ra) >= SO_DONG:
                break

        # Một video có đúng 100 hit và 3 mốc chỉ cho 98 cửa sổ liên tiếp.
        # Bổ sung cửa sổ thưa hơn từ chính các hit đó để đủ 100 dòng; vẫn là
        # frame thật, cùng video và tăng nghiêm ngặt theo thời gian.
        if len(ra) < SO_DONG:
            for buoc in range(2, 11):
                for vid in thu_tu_video:
                    ds = theo_video[vid]
                    do_dai = (so_moc - 1) * buoc + 1
                    for bat_dau in range(max(0, len(ds) - do_dai + 1)):
                        chon = sorted(
                            ds[bat_dau: bat_dau + do_dai: buoc],
                            key=lambda h: h.pts_time,
                        )
                        frame_ids = [int(h.frame_idx) for h in chon]
                        khoa = (vid, tuple(frame_ids))
                        if (
                            len(frame_ids) == so_moc
                            and frame_ids == sorted(frame_ids)
                            and len(set(frame_ids)) == so_moc
                            and khoa not in da_co
                        ):
                            da_co.add(khoa)
                            ra.append(Answer(vid, frame_ids))
                        if len(ra) >= SO_DONG:
                            break
                    if len(ra) >= SO_DONG:
                        break
                if len(ra) >= SO_DONG:
                    break

        return ra[:SO_DONG], (
            f"{len(ra[:SO_DONG])} dòng, {so_moc} mốc/dòng | {ghi_chu_ghep}"
        )

    # KIS/QA: lọc trên toàn bộ 300 hit TRƯỚC khi cắt 100. Bản cũ cắt trước,
    # nên 24/25 tệp production trượt cổng 5 dù mạch tìm kiếm đã trả đủ ứng viên.
    from aic2026.rank.config import cua_so_giay
    from aic2026.rank.dedupe import loc_trung

    hits, bao_cao_loc = loc_trung(hits, cua_so_giay=cua_so_giay())
    ghi_chu_loc = (
        f"lọc {bao_cao_loc.cua_so_giay:.1f}s "
        f"{bao_cao_loc.vao}->{bao_cao_loc.ra}"
    )

    if dang == "qa":
        from collections import Counter

        from aic2026.qa_answer import doan_du_phong, tra_loi_theo_hang

        try:
            cap = tra_loi_theo_hang(
                de["cau_hoi"], hits, so_dong=SO_DONG,
                so_hang_vlm=5, dung_vlm=True, mo_rong_lan_can=False,
            )
        except Exception as loi:
            d = doan_du_phong(de["cau_hoi"])
            cap = [(h, d) for h in hits[:SO_DONG]]

        # Đáp án phải đi CÙNG khung đã sinh nó. Bản cũ lấy một đáp án từ
        # top-5 rồi gán cho cả 100 khung; khung đúng ở hạng sau dù OCR/ASR đọc
        # được đáp án vẫn bị ghi đáp án của khung khác.
        ra = [
            Answer(
                str(h.video_id), [int(h.frame_idx)], answer=d.van_ban
            )
            for h, d in cap
        ]
        nguon = Counter(d.nguon for _, d in cap)
        return ra, (
            f"{len(ra)} dòng, đáp án theo từng khung "
            f"(nguồn {dict(nguon)}) | {ghi_chu_loc}"
        )

    ra = [Answer(str(h.video_id), [int(h.frame_idx)]) for h in hits[:SO_DONG]]
    return ra, f"{len(ra)} dòng | {ghi_chu_loc}"


def dung_tim(dung_caption: bool = False):
    from aic2026.index.fts_index import TextSearchIndex
    from aic2026.paths import FTS_DIR
    from aic2026.rank.config import trong_so_theo_dang
    from aic2026.rank.hop_nhat import tim_ung_vien_gop

    kho_chu = TextSearchIndex(FTS_DIR / "text.sqlite")
    kho_caption = None
    if dung_caption:
        from aic2026.enrich.caption import CaptionSearchIndex

        kho_caption = CaptionSearchIndex(nguon_dich="marian", bat_buoc_dich=True)
        if kho_caption.thong_ke()["so_ban_ghi_caption"] == 0:
            raise RuntimeError(
                "--dung-caption đã bật nhưng caption_fts rỗng; chạy "
                "scripts.run_caption_batch --chi-nap trước."
            )

    def dung_mot_dang(dang):
        trong_so = dict(trong_so_theo_dang(dang))
        dung_caption_dang_nay = dung_caption and dang == "kis"
        if dung_caption_dang_nay and float(trong_so.get("caption", 0.0)) <= 0:
            # Giá trị khởi đầu để chạy thử. Sau phép đo task 10, ghi con số
            # chốt vào rrf_weights.yaml thì giá trị đo được sẽ thắng chỗ này.
            trong_so["caption"] = 0.5

        return tim_ung_vien_gop(
            kho_chu=kho_chu,
            kho_caption=kho_caption,
            dung_clip=float(trong_so.get("clip", 0.0)) > 0,
            dung_ocr_fts=float(trong_so.get("ocr_fts", 0.0)) > 0,
            dung_asr=float(trong_so.get("asr", 0.0)) > 0,
            dung_caption=dung_caption_dang_nay,
            dung_clip_l=float(trong_so.get("clip_l", 0.0)) > 0,
            mo_rong_truy_van=False,
            nguon_clip_l="marian",
            dang_cau=dang,
            trong_so=trong_so,
        )

    return {
        dang: dung_mot_dang(dang)
        for dang in ("kis", "qa", "trake")
    }


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Việc 13 — sinh tệp nộp cho MỌI đề")
    p.add_argument("--de", required=True, help="thư mục chứa các tệp đề .txt")
    p.add_argument("--ra", default="nop", help="tên thư mục kết quả")
    p.add_argument("--thu-nghiem", action="store_true",
                   help="không tra kho, chỉ sinh dòng đoán — để kiểm định dạng")
    p.add_argument(
        "--dung-caption",
        action="store_true",
        help="bật nhánh caption cho câu KIS sau khi đã nghiệm thu/đo task 10",
    )
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

    tim = None if args.thu_nghiem else dung_tim(dung_caption=args.dung_caption)
    tra_nguoc = bang_tra_nguoc_frame_map()

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

        # Chốt độc lập trên đúng các Answer sắp ghi. Lớp này còn bắt được lỗi
        # do QA mở rộng khung lân cận hoặc do một nhánh mới quên gọi loc_trung.
        cau_tra_loi, bo_gan, bo_khong_tra = loc_cua_so_cau_tra_loi(
            cau_tra_loi, d["dang"], tra_nguoc=tra_nguoc,
        )
        if bo_gan:
            ghi_chu += f" | cổng 5 bỏ {bo_gan} dòng gần"
        if bo_khong_tra:
            ghi_chu += f" | bỏ {bo_khong_tra} frame không có trong map"

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
            if d["dang"] == "qa" and not str(dap_an_qa or "").strip():
                dap_an_qa = "khong ro"
            so_du_phong = SO_DONG if d["dang"] == "trake" else SO_DONG * 10
            du_phong = [
                (
                    Answer(a.video_id, a.frame_ids, answer=dap_an_qa)
                    if d["dang"] == "qa" else a
                )
                for a in dong_doan(so_moc, so_dong=so_du_phong)
            ]

            # Lọc lại trên tập kết hợp để dòng lấp cũng không xung đột 10 giây
            # với kết quả thật đã giữ. Sau đó dựng lại ngân sách theo đúng hạng.
            ket_hop, _, _ = loc_cua_so_cau_tra_loi(
                [*ngan_sach.answers, *du_phong],
                d["dang"],
                tra_nguoc=tra_nguoc,
            )
            ngan_sach = SubmissionBudget(task=DANG_BTC[d["dang"]])
            ngan_sach.extend(ket_hop)

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
