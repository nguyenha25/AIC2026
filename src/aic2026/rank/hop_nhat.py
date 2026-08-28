"""
Hợp nhất kiểu trả về và cắm nhánh gộp vào mạch của Ngân — Task 10.

Tệp này làm hai việc:

  1. `gop_nguon()` — gộp nhiều danh sách `Hit` bằng RRF, trả về `list[Hit]`.
     Dùng cho UI (`ui/app.py`), nơi mọi thứ chạy trên `Hit`.

  2. `tim_ung_vien_gop()` — nhà máy sinh hàm đúng chữ ký `(câu chữ, số ứng viên)
     -> list[Hit]` để truyền vào `run_query(tim_ung_vien=...)`.

CHỖ CẮM ĐÃ CÓ SẴN, KHÔNG PHẢI SỬA TỆP CỦA NGÂN
----------------------------------------------
`rank/search.py` ghi rõ ở phần đầu:

    "Việc 10 của Nghi (gộp hai bảng xếp hạng) sẽ cắm vào tham số `tim_ung_vien`
     của run_query(): truyền một hàm nhận câu chữ, trả về danh sách Hit đã xếp
     hạng. Mạch từ bước 3 trở đi không phải sửa gì."

Đi qua `run_query()` thay vì tự ghi CSV là BẮT BUỘC, không phải cho gọn: mạch đó
mới có `_gom_trake()` gom nhiều mốc vào MỘT dòng (định dạng TRAKE là
`video_id,frame_1,frame_2,...`), có `loc_trung()` dùng `pts_time` thật thay vì
giả định fps cứng, có `SubmissionBudget` khử trùng dòng, và có `tim_cap_qua_gan()`
soát lại cổng thoát số 5.

THANG ĐIỂM SAU KHI GỘP
----------------------
Điểm RRF cỡ 1/(k+hạng), tối đa ~0.016 với k=60 — KHÔNG cùng thang với cosine
CLIP (0..1) hay BM25. Mọi ngưỡng cắt theo điểm phải áp TRƯỚC khi gộp, trên từng
nhánh. Tệp này không tự cắt ngưỡng.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from aic2026.frame_map import lookup
from aic2026.index.faiss_index import Hit
from aic2026.rank.fuse import khoa_gop, reciprocal_rank_fusion

NGUON_CLIP = "clip"
NGUON_OCR = "ocr"          # OCR qua OCRReranker (khớp từ khoá trên derived/ocr/)
NGUON_OCR_FTS = "ocr_fts"  # OCR qua TextSearchIndex (BM25 trên bảng ocr_fts)
NGUON_ASR = "asr"          # Lời nói qua TextSearchIndex (BM25 trên bảng asr_fts)
NGUON_OBJECT = "object"    # Vật thể qua ObjectSearchIndex (Việc 3, bảng objects_fts)
NGUON_CAPTION = "caption"  # Mô tả tự sinh qua CaptionSearchIndex (Việc 10)
NGUON_CLIP_L = "clip_l"    # Chỉ mục ảnh THỨ HAI, mô hình mạnh hơn (Việc 8)

# Trọng số bất đối xứng: chữ làm tie-breaker, không lật đổ hình ảnh.
#
# CẢNH BÁO — 0.45 CHƯA được hiệu chỉnh. Đo trên bộ dev 32 câu: ở câu 11, nhánh
# OCR chạy riêng đưa frame đúng vào top-5 (0.80 điểm), nhưng sau khi gộp frame
# đó rơi xuống hạng ~70 (còn 0.20). Khi OCR trả rất ít ứng viên, ít ứng viên là
# dấu hiệu ĐỘ CHÍNH XÁC CAO chứ không phải yếu — trọng số hiện tại đang phạt
# nhầm. Cần bộ dev 60-80 câu mới hiệu chỉnh tử tế.
TRONG_SO_MAC_DINH = {
    NGUON_CLIP: 1.0,
    NGUON_OCR: 0.45,
    # 0.6 theo config.py của nhóm (khoá "asr"). Chưa đo riêng, giữ nguyên
    # con số đã khai thay vì tự đặt số khác.
    NGUON_OCR_FTS: 0.6,
    NGUON_ASR: 0.6,
    # Việc 3 và Việc 10 — CHƯA hiệu chỉnh, đây chỉ là số khởi đầu để chạy
    # được. Việc 6 quét lưới trên dev v2 rồi ghi số chốt vào
    # config/rrf_weights.yaml.
    NGUON_OBJECT: 0.4,
    NGUON_CAPTION: 0.5,
    # Việc 8 — mặc định 0. Bật lên chỉ sau khi do_trong_so_rrf đo được. Chỉ
    # mục này có thể chỉ phủ một phần kho (mỗi máy mã hoá shard của mình), nên
    # trọng số phải do phép đo quyết định, không đoán.
    NGUON_CLIP_L: 0.0,
}


def hit_sang_dict(h: Hit) -> dict:
    return {
        "video_id": h.video_id,
        "n": h.n,
        "frame_idx": h.frame_idx,
        "pts_time": h.pts_time,
        "score": h.score,
        "source": h.source,
    }


def dict_sang_hit(d: dict, nguon: str = NGUON_OCR) -> Hit | None:
    """
    Kết quả OCR (dict) -> Hit. Trả None nếu không tra được, để phía gọi đếm được
    bao nhiêu ứng viên bị bỏ thay vì im lặng mất.
    """
    video_id = d.get("video_id")
    n = d.get("n")

    if video_id is None or n is None:
        return None

    frame_idx = d.get("frame_idx")
    pts_time = d.get("pts_time")

    if frame_idx is None or pts_time is None:
        try:
            frame_idx, pts_time = lookup(str(video_id), int(n))
        except Exception:
            return None

    return Hit(
        video_id=str(video_id),
        n=int(n),
        score=float(d.get("score", 0.0)),
        frame_idx=int(frame_idx),
        pts_time=float(pts_time),
        source=nguon,
    )


# Bảng tra đã gom nhóm theo video, dựng MỘT LẦN rồi dùng lại.
#
# Bản đầu lọc thẳng trên DataFrame cho TỪNG kết quả ASR:
#     frame_map_df[(frame_map_df["video_id"] == vid) & (...)]
# Với top_k=500 thì đó là 500 lần quét toàn bộ 177.321 dòng cho MỖI câu hỏi —
# chạy 32 câu mất hàng phút và nhìn như treo. Gom nhóm trước rồi tìm nhị phân
# trong mảng pts_time đã sắp xếp của đúng video đó thì chỉ còn vài mili giây.
_BANG_THEO_VIDEO: dict | None = None


def _lay_bang_theo_video(frame_map_df=None) -> dict:
    """{video_id: (mảng pts_time đã sắp, mảng n, mảng frame_idx)}."""
    global _BANG_THEO_VIDEO

    if _BANG_THEO_VIDEO is not None and frame_map_df is None:
        return _BANG_THEO_VIDEO

    from aic2026.frame_map import load_frame_map

    if frame_map_df is None:
        frame_map_df = load_frame_map()

    bang: dict = {}

    for vid, nhom in frame_map_df.groupby("video_id", sort=False):
        nhom = nhom.sort_values("pts_time")
        bang[str(vid)] = (
            nhom["pts_time"].to_numpy(),
            nhom["n"].to_numpy(),
            nhom["frame_idx"].to_numpy(),
        )

    if frame_map_df is None:
        _BANG_THEO_VIDEO = bang
    else:
        _BANG_THEO_VIDEO = bang

    return bang


def no_khoang_asr(
    ket_qua_asr,
    frame_map_df=None,
    toi_da_moi_khoang: int = 8,
):
    """
    Nở kết quả ASR (KHOẢNG THỜI GIAN) thành các khung hình CÓ THẬT trong kho.

    VÌ SAO KHÔNG DÙNG THẲNG frame_idx_start
    ---------------------------------------
    search_asr() trả về [start, end] của một câu nói. `frame_idx_start` là khung
    hình lúc BẮT ĐẦU CÂU NÓI — gần như chắc chắn KHÔNG phải một keyframe có trong
    kho, vì keyframe lấy thưa (~65-90 khung một tấm). Nộp thẳng số đó là trỏ tới
    tấm ảnh không tồn tại: tệp nộp vẫn đúng định dạng, vẫn chấm được, nhưng luôn
    trượt — không có thông báo lỗi nào. Cùng họ với lỗi nhầm `n` với `frame_idx`.

    scripts/nap_asr_vao_fts.py nhắc đúng điều này:
        "Việc 10 phải đối chiếu khoảng [start, end] với frame_map để lấy ra các
         tấm ảnh nằm trong khoảng, rồi mới đưa vào RRF."

    toi_da_moi_khoang giới hạn số khung lấy từ MỘT câu nói. Câu nói dài 30 giây
    có thể phủ hàng chục keyframe; không chặn thì một câu chiếm hết danh sách
    ứng viên và các câu khác không còn chỗ.
    """
    import numpy as np

    bang = _lay_bang_theo_video(frame_map_df)
    ra: list[Hit] = []

    for muc in ket_qua_asr:
        vid = str(muc.get("video_id", ""))
        nhom = bang.get(vid)

        if nhom is None:
            continue

        try:
            bat_dau = float(muc["start"])
            ket_thuc = float(muc["end"])
        except (KeyError, TypeError, ValueError):
            continue

        pts, cot_n, cot_frame = nhom

        # pts đã sắp xếp -> tìm nhị phân hai đầu khoảng.
        trai = int(np.searchsorted(pts, bat_dau, side="left"))
        phai = int(np.searchsorted(pts, ket_thuc, side="right"))

        if phai <= trai:
            # Không keyframe nào rơi vào câu nói này. Bỏ qua — thà mất ứng viên
            # còn hơn nộp khung hình không có trong kho.
            continue

        phai = min(phai, trai + toi_da_moi_khoang)
        diem = float(muc.get("score", 0.0))

        for k in range(trai, phai):
            ra.append(
                Hit(
                    video_id=vid,
                    n=int(cot_n[k]),
                    score=diem,
                    frame_idx=int(cot_frame[k]),
                    pts_time=float(pts[k]),
                    source=NGUON_ASR,
                )
            )

    return ra


def gop_nguon(
    cac_nguon: Mapping[str, Sequence[Hit]],
    trong_so: Mapping[str, float] | None = None,
    k_rrf: int = 60,
) -> list[Hit]:
    """
    Gộp nhiều nguồn `Hit` bằng RRF, trả về `list[Hit]` đã xếp hạng.

    Nguồn rỗng được bỏ qua, nên gọi được cả khi chỉ một nhánh có kết quả (nhánh
    OCR rỗng với câu tả cảnh là chuyện bình thường, không phải lỗi).

    `source` của Hit trả về ghi rõ nguồn đóng góp: "clip", "ocr", hay "clip+ocr".
    `n` và `pts_time` lấy từ Hit gốc — thiếu chúng thì bước lọc trùng theo giây
    và bước tra ảnh đều hỏng.
    """
    cac_nguon_that = {ten: list(ds) for ten, ds in cac_nguon.items() if ds}
    if not cac_nguon_that:
        return []

    # VIỆC 7: khoá là (video_id, pts_time), KHÔNG phải (video_id, frame_idx).
    # Xem phần đầu rank/fuse.py để biết vì sao khoá cũ gộp nhầm hai tấm ảnh.
    goc: dict[tuple, Hit] = {}
    dau_vao: dict[str, list[dict]] = {}

    for ten, danh_sach in cac_nguon_that.items():
        dau_vao[ten] = []
        for h in danh_sach:
            d = hit_sang_dict(h)
            khoa = khoa_gop(d)
            if khoa not in goc:
                goc[khoa] = h
            dau_vao[ten].append(d)

    ket_qua = reciprocal_rank_fusion(
        dau_vao,
        weights=dict(trong_so) if trong_so else None,
        k_rrf=k_rrf,
    )

    ra: list[Hit] = []
    for r in ket_qua:
        khoa = khoa_gop(r)
        h_goc = goc.get(khoa)
        if h_goc is None:
            continue

        ranks = r.get("ranks") or {}
        nhan_nguon = "+".join(sorted(ranks.keys())) if ranks else "rrf"

        ra.append(
            Hit(
                video_id=h_goc.video_id,
                n=h_goc.n,
                score=float(r.get("score", 0.0)),
                frame_idx=h_goc.frame_idx,
                pts_time=h_goc.pts_time,
                source=nhan_nguon,
            )
        )

    return ra


def tim_ung_vien_gop(
    ocr_engine=None,
    kho_chu=None,
    kho_vat_the=None,
    kho_caption=None,
    dung_clip: bool = True,
    dung_ocr: bool = False,
    dung_ocr_fts: bool = False,
    dung_asr: bool = False,
    dung_object: bool = False,
    loc_vat_the: bool = False,
    loc_token_hiem_ocr: bool = False,
    dung_caption: bool = False,
    dung_clip_l: bool = False,
    mo_rong_truy_van: bool = False,
    nguon_mo_rong: str | None = None,
    nguon_clip_l: str | None = None,
    dang_cau: str | None = None,
    trong_so=None,
    k_rrf: int = 60,
):
    """
    Sinh hàm để truyền vào `run_query(tim_ung_vien=...)`.

    Bốn nguồn bật/tắt độc lập, nên so được từng nguồn riêng lẻ lẫn mọi tổ hợp mà
    vẫn đi qua ĐÚNG MỘT đường ống — nếu mỗi chế độ một lối ghi tệp thì chênh lệch
    đo được sẽ lẫn cả khác biệt về đường ống, không còn đo riêng đóng góp của nguồn.

        dung_clip     CLIP (FAISS)
        dung_ocr      OCR qua OCRReranker   — khớp từ khoá, lọc conf 0.40
        dung_ocr_fts  OCR qua TextSearchIndex — BM25 trên bảng ocr_fts
        dung_asr      Lời nói qua TextSearchIndex — BM25 trên bảng asr_fts

    Hai nhánh OCR CỐ Ý tách riêng: nhóm đang có hai bộ máy đọc cùng dữ liệu OCR
    theo hai cách khác nhau (giao diện dùng FTS, nghiệm thu Task 10 dùng
    OCRReranker). Bật riêng từng cái mới trả lời được cái nào tốt hơn, thay vì
    đoán.

        dung_object   Vật thể làm NHÁNH XẾP HẠNG — BM25 trên bảng objects_fts
        loc_vat_the   Vật thể làm BỘ LỌC, thu hẹp ứng viên TRƯỚC khi gộp.
                      Đây mới là cách checklist yêu cầu. Hai cờ độc lập nhau,
                      bật cả hai được, nhưng nên đo riêng từng cờ.
        dung_caption  Mô tả tự sinh qua CaptionSearchIndex — BM25 trên caption_fts

    ocr_engine cần cho dung_ocr; kho_chu (TextSearchIndex) cần cho dung_ocr_fts
    và dung_asr; kho_vat_the cho dung_object; kho_caption cho dung_caption.

    VIỆC 5 — mo_rong_truy_van CHỈ ĐỔI CÂU CHO NHÁNH CLIP
    ----------------------------------------------------
    Bật lên thì nhánh CLIP nhận cụm TIẾNG ANH đã rút gọn, còn bốn nhánh chữ
    (ocr, ocr_fts, asr, object) vẫn nhận NGUYÊN câu tiếng Việt. Đưa câu tiếng
    Anh vào nhánh OCR/ASR là bảo đảm 0 kết quả — chữ trên hình và lời nói
    trong video đều là tiếng Việt.
    """
    # VIỆC 6 — TRỌNG SỐ TÁCH THEO DẠNG CÂU.
    #
    # Đo trên 24 câu dev: nhánh ASR chạy một mình được KIS 0,0000 nhưng
    # Q&A 0,1667. Nhánh OCR ngược lại: KIS 0,3000, Q&A 0,0000. Dùng CHUNG một
    # vector trọng số cho cả hai dạng là bắt mỗi dạng gánh nhánh vô dụng của
    # dạng kia — đo được: thêm ASR vào KIS làm điểm TỤT 0,3833 -> 0,3500.
    #
    # Thứ tự ưu tiên: tham số truyền vào > config/rrf_weights.yaml theo dạng
    # > settings.yaml > bảng mặc định ở đầu tệp này.
    if trong_so is None:
        try:
            from aic2026.rank.config import trong_so_theo_dang

            trong_so = trong_so_theo_dang(dang_cau)
        except Exception:
            trong_so = TRONG_SO_MAC_DINH

    trong_so = trong_so or TRONG_SO_MAC_DINH

    def _tim(cau_hoi: str, so_ung_vien: int):
        cac_nguon: dict[str, list[Hit]] = {}

        if dung_clip:
            # Import TRONG hàm để giữ đúng thứ tự torch-trước-faiss mà Ngân đã ghi
            # trong search.py: nạp faiss trước torch trên Windows thì tiến trình
            # chết với 0xC0000005, không traceback, màn hình trống.
            if mo_rong_truy_van:
                from aic2026.query_expand import tim_ung_vien_clip_mo_rong

                _clip = tim_ung_vien_clip_mo_rong(nguon=nguon_mo_rong)
            else:
                from aic2026.rank.search import tim_ung_vien_clip as _clip

            cac_nguon[NGUON_CLIP] = list(_clip(cau_hoi, so_ung_vien))

        if dung_ocr and ocr_engine is not None:
            tho = ocr_engine.search_ocr(cau_hoi, top_k=so_ung_vien)
            hits = [dict_sang_hit(d, NGUON_OCR) for d in tho]
            cac_nguon[NGUON_OCR] = [h for h in hits if h is not None]

        if dung_ocr_fts and kho_chu is not None:
            # search_text() trả sẵn cả n lẫn frame_idx nên đổi thẳng sang Hit được.
            # loc_token_hiem_ocr: bỏ token phổ biến khỏi truy vấn OCR.
            # Câu dev 11 có ~20 token nên AND luôn rỗng rồi rơi về OR trên
            # toàn bộ; khung đúng bật ra ngoài 1000. Chỉ tra "Mac Cuu" thì nó
            # ở hạng 6.
            tho = kho_chu.search_text(
                cau_hoi, top_k=so_ung_vien, loc_hiem=loc_token_hiem_ocr
            )
            hits = [dict_sang_hit(d, NGUON_OCR_FTS) for d in tho]
            cac_nguon[NGUON_OCR_FTS] = [h for h in hits if h is not None]

        if dung_asr and kho_chu is not None:
            # search_asr() trả KHOẢNG THỜI GIAN, không phải keyframe — phải nở
            # khoảng ra khung hình có thật. Xem no_khoang_asr().
            tho = kho_chu.search_asr(cau_hoi, top_k=so_ung_vien)
            cac_nguon[NGUON_ASR] = no_khoang_asr(tho)

        if loc_vat_the and kho_vat_the is not None:
            # VIỆC 3 — BỘ LỌC, chạy TRƯỚC khi gộp.
            #
            # Checklist viết: "lọc theo vật thể thu hẹp 177.321 keyframe xuống
            # vài trăm TRƯỚC khi tính CLIP". Bản đầu dựng nhánh vật thể thành
            # một bảng xếp hạng song song rồi gộp RRF — sai kiến trúc, và đo
            # được 0,0167, tức gần như vô dụng.
            #
            # Ở đây nó làm đúng việc của mình: giữ lại các khung CÓ vật thể
            # hiếm mà câu hỏi nhắc tới, bỏ phần còn lại. Không tìm thấy khái
            # niệm hiếm nào thì khung_chua() trả None và KHÔNG lọc gì — thà
            # mất cơ hội thu hẹp còn hơn giết mất đáp án đúng.
            tap_giu = kho_vat_the.khung_chua(cau_hoi)
            if tap_giu is not None:
                for ten, ds in cac_nguon.items():
                    cac_nguon[ten] = [
                        h for h in ds
                        if (str(h.video_id), round(float(h.pts_time), 3)) in tap_giu
                    ] or ds       # lọc sạch một nhánh thì trả nhánh đó về nguyên

        if dung_object and kho_vat_the is not None:
            # Câu không nhắc vật thể nào -> trả [] -> nhánh này tự vắng mặt
            # khỏi phép gộp. Đó là hành vi ĐÚNG: nhánh vật thể chỉ nên lên
            # tiếng khi câu hỏi thật sự tả vật thể.
            tho = kho_vat_the.tra_bang_cau_viet(cau_hoi, top_k=so_ung_vien)
            hits = [dict_sang_hit(d, NGUON_OBJECT) for d in tho]
            cac_nguon[NGUON_OBJECT] = [h for h in hits if h is not None]

        if dung_clip_l:
            # VIỆC 8 — chỉ mục ảnh THỨ HAI, tìm ĐỘC LẬP.
            #
            # Khác rerank (Việc 4) ở chỗ căn bản: rerank chỉ xếp lại top-100 do
            # B/32 trả về, không kéo vào được thứ B/32 đã bỏ sót. Câu dev 11
            # không nằm trong top-300 của CLIP nên rerank vô dụng với nó.
            #
            # Câu vào phải là TIẾNG ANH, giống nhánh clip gốc.
            from aic2026.index.clip_l_index import tim as _tim_clip_l

            cau_anh = cau_hoi
            if nguon_clip_l:
                try:
                    from aic2026.query_expand import mo_rong

                    cau_anh = mo_rong(
                        cau_hoi,
                        nguon=nguon_clip_l,
                        bat_buoc=True,
                    ).cum_chinh
                except Exception:
                    # Mạch sản phẩm vẫn phải trả kết quả nếu bộ dịch lỗi.
                    cau_anh = cau_hoi

            hits = [
                dict_sang_hit(d, NGUON_CLIP_L) for d in _tim_clip_l(cau_anh, so_ung_vien)
            ]
            cac_nguon[NGUON_CLIP_L] = [h for h in hits if h is not None]

        if dung_caption and kho_caption is not None:
            tho = kho_caption.tra_cuu(cau_hoi, top_k=so_ung_vien)
            hits = [dict_sang_hit(d, NGUON_CAPTION) for d in tho]
            cac_nguon[NGUON_CAPTION] = [h for h in hits if h is not None]

        # Một nguồn vẫn cho qua RRF, để thứ tự sinh ra theo cùng một cách ở mọi
        # chế độ.
        return gop_nguon(cac_nguon, trong_so=trong_so, k_rrf=k_rrf)

    return _tim