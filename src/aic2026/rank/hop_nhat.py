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
from aic2026.rank.fuse import reciprocal_rank_fusion

NGUON_CLIP = "clip"
NGUON_OCR = "ocr"

# Trọng số bất đối xứng: chữ làm tie-breaker, không lật đổ hình ảnh.
#
# CẢNH BÁO — 0.45 CHƯA được hiệu chỉnh. Đo trên bộ dev 32 câu: ở câu 11, nhánh
# OCR chạy riêng đưa frame đúng vào top-5 (0.80 điểm), nhưng sau khi gộp frame
# đó rơi xuống hạng ~70 (còn 0.20). Khi OCR trả rất ít ứng viên, ít ứng viên là
# dấu hiệu ĐỘ CHÍNH XÁC CAO chứ không phải yếu — trọng số hiện tại đang phạt
# nhầm. Cần bộ dev 60-80 câu mới hiệu chỉnh tử tế.
TRONG_SO_MAC_DINH = {NGUON_CLIP: 1.0, NGUON_OCR: 0.45}


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

    goc: dict[tuple[str, int], Hit] = {}
    dau_vao: dict[str, list[dict]] = {}

    for ten, danh_sach in cac_nguon_that.items():
        dau_vao[ten] = []
        for h in danh_sach:
            khoa = (str(h.video_id), int(h.frame_idx))
            if khoa not in goc:
                goc[khoa] = h
            dau_vao[ten].append(hit_sang_dict(h))

    ket_qua = reciprocal_rank_fusion(
        dau_vao,
        weights=dict(trong_so) if trong_so else None,
        k_rrf=k_rrf,
    )

    ra: list[Hit] = []
    for r in ket_qua:
        khoa = (str(r["video_id"]), int(r["frame_idx"]))
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
    dung_clip: bool = True,
    dung_ocr: bool = True,
    trong_so: Mapping[str, float] | None = None,
    k_rrf: int = 60,
) -> Callable[[str, int], list[Hit]]:
    """
    Sinh hàm để truyền vào `run_query(tim_ung_vien=...)`.

    `dung_clip` / `dung_ocr` cho phép chạy riêng từng nhánh mà vẫn đi qua ĐÚNG
    MỘT đường ống — nghiệm thu so ba chế độ nhờ vậy công bằng, không phải mỗi
    chế độ một lối ghi tệp khác nhau.
    """
    trong_so = trong_so or TRONG_SO_MAC_DINH

    def _tim(cau_hoi: str, so_ung_vien: int) -> list[Hit]:
        cac_nguon: dict[str, list[Hit]] = {}

        if dung_clip:
            # Import TRONG hàm để giữ đúng thứ tự torch-trước-faiss mà Ngân đã
            # ghi trong search.py: nạp faiss trước torch trên Windows thì tiến
            # trình chết với 0xC0000005, không traceback, màn hình trống.
            from aic2026.rank.search import tim_ung_vien_clip

            cac_nguon[NGUON_CLIP] = list(tim_ung_vien_clip(cau_hoi, so_ung_vien))

        if dung_ocr and ocr_engine is not None:
            tho = ocr_engine.search_ocr(cau_hoi, top_k=so_ung_vien)
            hits = [dict_sang_hit(d, NGUON_OCR) for d in tho]
            cac_nguon[NGUON_OCR] = [h for h in hits if h is not None]

        # Một nhánh vẫn cho qua RRF, để thứ tự sinh ra theo cùng một cách ở cả ba
        # chế độ. Nếu chế độ chỉ-CLIP đi lối khác thì phép so ba chế độ sẽ lẫn cả
        # khác biệt về đường ống, không còn đo riêng đóng góp của chữ.
        return gop_nguon(cac_nguon, trong_so=trong_so, k_rrf=k_rrf)

    return _tim
