"""
Bộ tạo tệp nộp thống nhất — dựng lại ở Giai đoạn 2.

VÌ SAO TỆP NÀY TỒN TẠI
----------------------
Cả nhóm vẫn tin là đã có `tao_tep_nop.py`, nhưng nó KHÔNG có trên nhánh main
(soát ngày 21/08). Đường xuất tệp nộp duy nhất còn lại là hàm TODO
`xuat_csv_gio_nop()` trong giao diện, và hàm đó chỉ ghi được hai cột kiểu KIS.
Nghĩa là hiện tại nhóm KHÔNG có cách nào xuất tệp nộp cho Q&A và TRAKE từ dòng
lệnh. Tệp này lấp đúng chỗ đó.

NÓ KHÔNG TỰ TÌM KIẾM. Nó nhận vào những gì NGƯỜI đã chọn (video + mốc thời
gian) rồi lo phần dễ sai: đổi giây thành frame_idx, dựng đúng định dạng, khử
trùng, cắt 100 dòng, đặt đúng tên tệp. Việc tìm ra đáp án là của giao diện và
của người ngồi máy.

LUẬT ĐỔI GIÂY -> FRAME_IDX (đã xác minh 100% trên 177.321 dòng)

    frame_idx = floor(pts_time * fps)

fps lấy từ cột `fps` của frame_map, KHÔNG đoán 25 — bộ dữ liệu có 25 / 29,97 /
30 / 26,44... Tự nhân tay bằng 25 là cách sai điểm mà không có triệu chứng.

CÁCH DÙNG
---------
1) KIS — một video, một mốc:

    python -m scripts.tao_tep_nop --ma-cau query-1 --loai kis \\
        --muc "L01_V001@12.4" --muc "L02_V015@88.0"

2) Q&A — thêm đáp án (một đáp án dùng chung cho mọi dòng):

    python -m scripts.tao_tep_nop --ma-cau query-7 --loai qa \\
        --dap-an "màu xanh" --muc "L01_V001@12.4"

   Muốn mỗi dòng một đáp án khác nhau thì gắn vào sau dấu `|`:

    --muc "L01_V001@12.4|màu xanh" --muc "L02_V015@88.0|màu đỏ"

3) TRAKE — một dòng là một CHUỖI mốc trong CÙNG một video, ngăn bằng dấu phẩy:

    python -m scripts.tao_tep_nop --ma-cau query-12 --loai trake \\
        --muc "L01_V001@12.4,15.8,19.2,23.0"

4) Nhận giây từ tệp thay vì gõ tay (mỗi dòng một --muc, bỏ dấu nháy):

    python -m scripts.tao_tep_nop --ma-cau query-1 --loai kis --tu-tep cac_moc.txt

CHẤP NHẬN CẢ FRAME_IDX TRỰC TIẾP
Nếu đã có sẵn frame_idx (chứ không phải giây) thì thêm `#` thay cho `@`:

    --muc "L01_V001#310"

MỐC KHÔNG NẰM TRÊN LƯỚI KEYFRAME LÀ HỢP LỆ.
Bài nộp được trỏ vào bất kỳ khung nào trong video, không riêng keyframe — đó
chính là điều quy trình tinh chỉnh bằng PotPlayer sinh ra. Bản nộp BTC đã công
nhận có các mốc 6612, 6613, 6614 liền nhau trong khi lưới keyframe cách 24
khung. Tệp này chỉ CẢNH BÁO theo tỷ lệ: lệch quá nửa số mốc mới là dấu hiệu
nhầm `n` với `frame_idx`.

TÊN TỆP — ĐÃ XÁC NHẬN
Bản nộp BTC công nhận (AIC2026_p1_final.zip) dùng `query-<mã>-<loại>.csv`,
ví dụ `query-p1-16-trake.csv`. Mã câu có cả tiền tố đợt ("p1-16").

    --ma-cau p1-16 --loai trake    ->  query-p1-16-trake.csv

submission_filename() tự thêm "query-", nên `--ma-cau query-p1-16` cũng ra đúng
(tệp này cắt tiền tố thừa). Khi đề cho sẵn tên khác thì ghi đè hoàn toàn:

    --ten-tep "ten-btc-cho-san.csv"

XUỐNG DÒNG: CRLF, theo đúng bản đã được công nhận. formatter.py nay mặc định
như vậy.

MÃ THOÁT
    0   ghi được tệp
    1   sai tham số, hoặc có mốc không tra được trong frame_map
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Windows cp1252 nuốt tiếng Việt khi in ra màn hình hoặc khi bị pipe.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.aic2026.frame_map import load_frame_map
from src.aic2026.submit import KIS, QA, TRAKE
from src.aic2026.submit.formatter import Answer, SubmissionBudget, submission_filename
from src.aic2026.paths import SUBMISSIONS_DIR

CAC_LOAI = (KIS, QA, TRAKE)

# Khung cuối của video nằm sau keyframe cuối, vì keyframe trích thưa.
DU_SAU_KEYFRAME_CUOI_GIAY = 120.0


# ---------------------------------------------------------------- tra cứu fps

def bang_fps() -> dict[str, float]:
    """video_id -> fps, lấy từ frame_map. Không đoán 25."""
    df = load_frame_map()
    return {str(v): float(f) for v, f in zip(df["video_id"], df["fps"])}


def luoi_keyframe() -> set[tuple[str, int]]:
    """
    Tập (video_id, frame_idx) của các KEYFRAME.

    KHÔNG phải danh sách khung hợp lệ. Bài nộp được trỏ vào bất kỳ khung nào
    trong video — bản nộp BTC đã công nhận có các mốc 6612, 6613, 6614 là ba
    khung liền nhau, trong khi lưới keyframe cách nhau 24 khung. Tập này chỉ
    dùng để ĐẾM TỶ LỆ lệch lưới, không dùng để chặn.
    """
    df = load_frame_map()
    return {(str(v), int(f)) for v, f in zip(df["video_id"], df["frame_idx"])}


def khung_cuoi_theo_video() -> dict[str, int]:
    """video_id -> frame_idx của keyframe cuối cùng."""
    df = load_frame_map()
    m: dict[str, int] = {}
    for v, f in zip(df["video_id"], df["frame_idx"]):
        v = str(v)
        f = int(f)
        m[v] = max(m.get(v, f), f)
    return m


def giay_sang_frame(pts_time: float, fps: float) -> int:
    """floor, KHÔNG phải round. round khớp 87% số dòng, floor khớp 100%."""
    return math.floor(pts_time * fps)


# ------------------------------------------------------------------ phân tích

def tach_muc(chuoi: str) -> tuple[str, list[float] | list[int], bool, str | None]:
    """
    "L01_V001@12.4,15.8|màu xanh"
        -> ("L01_V001", [12.4, 15.8], True, "màu xanh")

    Trả về (video_id, các mốc, la_giay, đáp án).
    la_giay=True nghĩa là các mốc đang là GIÂY và cần đổi sang frame_idx.
    """
    dap_an = None
    if "|" in chuoi:
        chuoi, dap_an = chuoi.split("|", 1)
        dap_an = dap_an.strip() or None

    chuoi = chuoi.strip()

    if "@" in chuoi:
        vid, phan = chuoi.split("@", 1)
        la_giay = True
    elif "#" in chuoi:
        vid, phan = chuoi.split("#", 1)
        la_giay = False
    else:
        raise ValueError(
            f"Mục {chuoi!r} thiếu @ hoặc #. Dùng 'L01_V001@12.4' (giây) "
            f"hoặc 'L01_V001#310' (frame_idx)."
        )

    vid = vid.strip()
    if not vid:
        raise ValueError(f"Mục {chuoi!r} thiếu video_id")

    moc_raw = [x.strip() for x in phan.split(",") if x.strip()]
    if not moc_raw:
        raise ValueError(f"Mục {chuoi!r} không có mốc nào")

    try:
        moc = [float(x) for x in moc_raw] if la_giay else [int(x) for x in moc_raw]
    except ValueError as loi:
        raise ValueError(f"Mục {chuoi!r} có mốc không phải số: {loi}") from None

    return vid, moc, la_giay, dap_an


# ---------------------------------------------------------------------- chính

def main() -> int:
    args = sys.argv[1:]

    def lay(ten: str, mac_dinh=None):
        return args[args.index(ten) + 1] if ten in args and args.index(ten) + 1 < len(args) else mac_dinh

    ma_cau = lay("--ma-cau")
    loai = (lay("--loai") or "").strip().lower()
    dap_an_chung = lay("--dap-an")
    tu_tep = lay("--tu-tep")
    thu_muc = lay("--thu-muc")
    ten_tep_ep = lay("--ten-tep")

    cac_muc = [args[i + 1] for i, a in enumerate(args) if a == "--muc" and i + 1 < len(args)]

    if tu_tep:
        p = Path(tu_tep)
        if not p.exists():
            print(f"Không thấy tệp: {p}")
            return 1
        cac_muc += [d.strip() for d in p.read_text(encoding="utf-8-sig").splitlines()
                    if d.strip() and not d.strip().startswith("#")]

    if not ma_cau or loai not in CAC_LOAI or not cac_muc:
        print(__doc__)
        print("THIẾU THAM SỐ. Cần --ma-cau, --loai (kis|qa|trake) và ít nhất một --muc.")
        return 1

    if loai == QA and not dap_an_chung and not any("|" in m for m in cac_muc):
        print("Q&A bắt buộc phải có đáp án: dùng --dap-an, hoặc gắn sau dấu | trong --muc.")
        return 1

    # --- nạp bảng đối chiếu -------------------------------------------------
    try:
        fps_theo_video = bang_fps()
        luoi = luoi_keyframe()
        khung_cuoi = khung_cuoi_theo_video()
    except Exception as loi:
        print(f"Không đọc được frame_map: {type(loi).__name__}: {loi}")
        return 1

    print("=" * 72)
    print(f"TẠO TỆP NỘP: {ma_cau}  ·  loại {loai.upper()}  ·  {len(cac_muc)} mục")
    print("=" * 72)

    ngan_sach = SubmissionBudget(task=loai)
    loi: list[str] = []
    tong_lech_luoi: list[str] = []
    tong_moc_da_xet = [0]

    for so_thu_tu, chuoi in enumerate(cac_muc, start=1):
        try:
            vid, moc, la_giay, dap_an_rieng = tach_muc(chuoi)
        except ValueError as e:
            loi.append(f"mục {so_thu_tu}: {e}")
            continue

        if vid not in fps_theo_video:
            loi.append(f"mục {so_thu_tu}: video {vid} không có trong frame_map")
            continue

        fps = fps_theo_video[vid]

        if la_giay:
            frame_ids = [giay_sang_frame(float(t), fps) for t in moc]
            mo_ta = " ".join(f"{t}s->{f}" for t, f in zip(moc, frame_ids))
        else:
            frame_ids = [int(x) for x in moc]
            mo_ta = " ".join(str(f) for f in frame_ids)

        # Nằm ngoài video thì chắc chắn sai — chặn.
        tran = khung_cuoi.get(vid, 0) + DU_SAU_KEYFRAME_CUOI_GIAY * fps
        ngoai = [f for f in frame_ids if f < 0 or f > tran]
        if ngoai:
            loi.append(
                f"mục {so_thu_tu} ({vid}): frame {ngoai} nằm ngoài video "
                f"(keyframe cuối {khung_cuoi.get(vid, 0)})"
            )
            continue

        # Lệch lưới keyframe KHÔNG phải lỗi — đó chính là kết quả của quy trình
        # tinh chỉnh tay bằng PotPlayer. Chỉ đếm để cảnh báo ở cuối, theo tỷ lệ.
        lech = [f for f in frame_ids if (vid, f) not in luoi]
        tong_moc_da_xet[0] += len(frame_ids)
        tong_lech_luoi.extend(f"{vid}#{f}" for f in lech)

        if loai == TRAKE and len(frame_ids) < 2:
            loi.append(f"mục {so_thu_tu}: TRAKE cần ít nhất 2 mốc, chỉ có {len(frame_ids)}")
            continue

        if loai in (KIS, QA) and len(frame_ids) > 1:
            loi.append(
                f"mục {so_thu_tu}: {loai.upper()} chỉ nhận MỘT mốc, đang có {len(frame_ids)}"
            )
            continue

        cau_tra_loi = None
        if loai == QA:
            cau_tra_loi = dap_an_rieng or dap_an_chung

        nhan = ngan_sach.add(
            Answer(video_id=vid, frame_ids=frame_ids, answer=cau_tra_loi)
        )

        trang_thai = "nhận" if nhan else "BỎ (trùng hoặc quá 100)"
        print(f"  {so_thu_tu:>3}. {vid:<12} fps {fps:>6.2f}  {mo_ta}   [{trang_thai}]")

    print()

    # Nhầm `n` với `frame_idx` là lỗi HỆ THỐNG: sai gần hết số mốc. Vài mốc
    # lệch lưới là tinh chỉnh tay, và bản BTC công nhận cũng có.
    if tong_moc_da_xet[0] and len(tong_lech_luoi) / tong_moc_da_xet[0] >= 0.5:
        loi.append(
            f"{len(tong_lech_luoi)}/{tong_moc_da_xet[0]} mốc không nằm trên lưới "
            f"keyframe — tỷ lệ quá cao để là tinh chỉnh tay, nhiều khả năng đang "
            f"dùng n thay vì frame_idx"
        )
    elif tong_lech_luoi:
        print(f"Lưu ý: {len(tong_lech_luoi)}/{tong_moc_da_xet[0]} mốc lệch lưới keyframe "
              f"({tong_lech_luoi[:3]}).")
        print("       HỢP LỆ — bài nộp trỏ được vào bất kỳ khung nào trong video.")
        print()

    if loi:
        print("KHÔNG GHI TỆP — có mục sai:")
        for x in loi:
            print(f"  - {x}")
        return 1

    if len(ngan_sach) == 0:
        print("KHÔNG GHI TỆP — không mục nào được nhận.")
        return 1

    thu_muc_ra = Path(thu_muc) if thu_muc else SUBMISSIONS_DIR

    if ten_tep_ep:
        ten_tep = ten_tep_ep
    else:
        # submission_filename() tự thêm "query-". Cắt tiền tố thừa để không ra
        # "query-query-1-kis.csv" khi người dùng gõ --ma-cau query-1.
        goc = ma_cau[len("query-"):] if ma_cau.lower().startswith("query-") else ma_cau
        ten_tep = submission_filename(goc, loai)

    duong_dan = thu_muc_ra / ten_tep
    ngan_sach.write(duong_dan)

    print(ngan_sach.report())
    print(f"Đã ghi: {duong_dan}")
    print()
    print("BƯỚC TIẾP THEO — bắt buộc, kể cả khi đang gấp:")
    print(f"  python -m scripts.check_submission {duong_dan}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
