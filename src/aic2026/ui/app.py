"""
Giao diện tra cứu AIC 2026.

Vị trí: src/aic2026/ui/app.py
Chạy:   streamlit run src/aic2026/ui/app.py

Nguyên tắc: giao diện KHÔNG tự nghĩ ra logic mà nhánh khác đã có.
Cửa sổ lọc trùng, số ứng viên, trần ảnh mỗi video, số dòng tối đa —
tất cả đọc từ rank/config.py. Lọc trùng gọi rank/dedupe.py.
Ứng viên gọi rank/search.py.

Lướt tìm KHÔNG đụng đĩa. Chỉ ghi tệp khi bấm xuất.
"""

from __future__ import annotations

import csv
import html
import inspect
import io
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import streamlit as st


# ============================================================
# 1. GỐC DỮ LIỆU
# ============================================================

def _doc_env_thu_cong(ten_bien: str) -> str | None:
    """Đọc một biến từ .env ở gốc repo khi chưa có python-dotenv."""

    tep_env = Path(__file__).resolve().parents[3] / ".env"

    if not tep_env.exists():
        return None

    # utf-8-sig để nuốt BOM do PowerShell 5.1 sinh ra.
    for dong in tep_env.read_text(encoding="utf-8-sig").splitlines():

        dong = dong.strip()

        if not dong or dong.startswith("#") or "=" not in dong:
            continue

        khoa, gia_tri = dong.split("=", 1)

        if khoa.strip() == ten_bien:
            return gia_tri.strip().strip('"').strip("'")

    return None


CANH_BAO_GOC_DU_LIEU: str | None = None


def _tim_goc_du_lieu() -> Path:

    global CANH_BAO_GOC_DU_LIEU

    try:
        from aic2026 import paths as _paths

        for ten in ("DATA_ROOT", "AIC_DATA_ROOT", "GOC_DU_LIEU"):

            gia_tri = getattr(_paths, ten, None)

            if gia_tri is not None:
                return Path(gia_tri)

    except Exception:
        pass

    for ten_khoa in ("AIC_DATA_ROOT", "DATA_ROOT", "AIC2026_DATA_ROOT"):

        gia_tri = os.environ.get(ten_khoa) or _doc_env_thu_cong(ten_khoa)

        if gia_tri:
            return Path(gia_tri)

    CANH_BAO_GOC_DU_LIEU = (
        "Không đọc được gốc dữ liệu từ paths.py hay .env — đang dùng mặc định D:/aic-data."
    )

    return Path("D:/aic-data")


GOC_DU_LIEU = _tim_goc_du_lieu()

THU_MUC_THUMBNAIL = GOC_DU_LIEU / "derived" / "thumbnails"
THU_MUC_KEYFRAME = GOC_DU_LIEU / "raw" / "keyframes"
THU_MUC_RUNS = GOC_DU_LIEU / "runs"
TEP_FTS = GOC_DU_LIEU / "index" / "fts" / "text.sqlite"

TEP_LICH_SU = THU_MUC_RUNS / "ui_history.jsonl"
TEP_GIO_NOP = THU_MUC_RUNS / "gio_nop.json"


# ============================================================
# 2. IMPORT AIC — THỨ TỰ NÀY QUAN TRỌNG
# ============================================================
#
# CẢNH BÁO: torch phải nạp TRƯỚC faiss. Đảo lại là tiến trình chết im lặng
# với mã 0xC0000005 trên Windows, không có traceback.
#
# Đường dẫn thật trên đĩa (liệt kê 16/08), KHÔNG theo mục 5.1 của tài liệu:
#   aic2026/index/encode/clip_encoder.py   (5.1 ghi aic2026/encode/...)
#   aic2026/rank/dedupe.py                 (5.1 ghi rank/dedup.py)

from aic2026.index.encode import clip_encoder as _nap_torch_truoc  # noqa: E402,F401
from aic2026.index.faiss_index import Hit                          # noqa: E402

from aic2026.rank import config as cfg                             # noqa: E402
from aic2026.rank.dedupe import loc_trung                          # noqa: E402
from aic2026.rank.search import tim_ung_vien_clip                  # noqa: E402
from aic2026.rank.hop_nhat import (                                # noqa: E402
    NGUON_ASR, NGUON_CLIP, NGUON_OCR, NGUON_OCR_FTS,
    tim_ung_vien_gop,
)


TRANG_THAI_MO_DUN: dict[str, str] = {}


def _nap_mem(ten_hien: str, duong_dan: str, ten_doi_tuong: str):
    """Nạp mềm nhưng GHI LẠI lý do — hỏng mà im lặng thì không ai biết."""

    try:
        mo_dun = __import__(duong_dan, fromlist=[ten_doi_tuong])
        doi_tuong = getattr(mo_dun, ten_doi_tuong)

        TRANG_THAI_MO_DUN[ten_hien] = "OK"

        return doi_tuong

    except Exception as loi:
        TRANG_THAI_MO_DUN[ten_hien] = f"{type(loi).__name__}: {loi}"
        return None


_TextSearchIndex = _nap_mem("Kho chữ FTS5", "aic2026.index.fts_index", "TextSearchIndex")
_load_frame_map = _nap_mem("Bảng đối chiếu", "aic2026.frame_map", "load_frame_map")
_lookup = _nap_mem("Tra ngược frame", "aic2026.frame_map", "lookup")


# ============================================================
# 3. HẰNG SỐ CỦA RIÊNG GIAO DIỆN
# ============================================================
#
# Chỉ đặt ở đây những thứ THUỘC VỀ cách trình bày. Mọi tham số ảnh hưởng
# tới kết quả đều lấy từ rank/config.py.

SO_COT = 4                 # 5 cột làm nhãn nút vỡ hai dòng
SO_KET_QUA_MOI_TRANG = 24  # chia hết cho 4
SO_LAN_CAN = 4

# Chiều cao ảnh nằm ở --aic-cao-anh trong style.css, KHÔNG lặp lại ở đây.
# Hai nguồn cho một con số là hai nguồn sẽ lệch nhau.

_LA_TEN_VIDEO = re.compile(r"L\d+_V\d+")


# ------------------------------------------------------------
# TƯƠNG THÍCH PHIÊN BẢN STREAMLIT
# ------------------------------------------------------------
#
# Tham số "cho rộng hết khung" đổi tên ba lần qua các bản:
#   bản cũ    use_column_width=True
#   bản giữa  use_container_width=True
#   bản mới   width="stretch"          (st.image đã BỎ hẳn hai tên kia)
#
# Bốn máy trong nhóm có thể cài bản khác nhau, nên đừng viết cứng.
# Dò chữ ký hàm rồi chọn đúng tên.

def _kw_rong(ham) -> dict:

    try:
        tham_so = inspect.signature(ham).parameters

    except (TypeError, ValueError):
        return {}

    if "use_container_width" in tham_so:
        return {"use_container_width": True}

    if "use_column_width" in tham_so:
        return {"use_column_width": True}

    if "width" in tham_so:
        return {"width": "stretch"}

    return {}


KW_ANH = _kw_rong(st.image)
KW_NUT = _kw_rong(st.button)
KW_TAI = _kw_rong(st.download_button)
KW_BANG = _kw_rong(st.dataframe)

# vertical_alignment của st.columns cũng chỉ có từ một bản nhất định.
# Viết cứng là máy cài bản cũ hơn sẽ văng TypeError giữa vòng thi.
KW_CANH = (
    {"vertical_alignment": "center"}
    if "vertical_alignment" in inspect.signature(st.columns).parameters
    else {}
)


def hien_anh(duong_dan) -> None:
    st.image(str(duong_dan), **KW_ANH)


st.set_page_config(
    page_title="AIC 2026 | Video Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 4. ĐỊNH DẠNG TÊN ẢNH — DÒ TRÊN ĐĨA
# ============================================================
#
# Bản trước tôi tin KeyframeRow.image_name. Sai: nó trả "0052.jpg" (bốn
# chữ số theo tài liệu) trong khi tệp thật là "052.jpg". Cả ảnh gốc BTC
# phát lẫn thumbnail đều BA chữ số, bắt đầu từ 001.
#
# Đĩa là sự thật. Không tin tài liệu, cũng không tin code.
# Đây là lỗi trong frame_map.py — cần báo Thi, vì ai gọi image_name
# cũng sẽ trỏ nhầm tệp.

def _do_dinh_dang(thu_muc_goc: Path) -> tuple[int, str]:
    """Lấy một tệp ảnh bất kỳ rồi suy ra (số chữ số, đuôi tệp)."""

    mac_dinh = (3, ".jpg")

    if not thu_muc_goc.exists():
        return mac_dinh

    try:
        for thu_muc_con in thu_muc_goc.iterdir():

            if not thu_muc_con.is_dir():
                continue

            for tep in sorted(thu_muc_con.iterdir()):

                if tep.suffix.lower() in (".jpg", ".jpeg", ".png") and tep.stem.isdigit():
                    return len(tep.stem), tep.suffix

    except OSError:
        pass

    return mac_dinh


@st.cache_resource(show_spinner=False)
def dinh_dang_thumbnail() -> tuple[int, str]:
    return _do_dinh_dang(THU_MUC_THUMBNAIL)


@st.cache_resource(show_spinner=False)
def dinh_dang_keyframe() -> tuple[int, str]:
    """Ảnh gốc nằm sâu hơn một tầng nên phải dò riêng."""

    mac_dinh = (3, ".jpg")

    if not THU_MUC_KEYFRAME.exists():
        return mac_dinh

    for cap1 in THU_MUC_KEYFRAME.iterdir():

        if not cap1.is_dir():
            continue

        for ung_vien in (cap1 / "keyframes", cap1):

            if ung_vien.is_dir():

                ket_qua = _do_dinh_dang(ung_vien)

                if ket_qua != mac_dinh:
                    return ket_qua

    return mac_dinh


def duong_dan_thumbnail(video_id: str, n: int) -> Path:

    so_chu_so, duoi = dinh_dang_thumbnail()

    return THU_MUC_THUMBNAIL / video_id / f"{n:0{so_chu_so}d}{duoi}"


@st.cache_resource(show_spinner=False)
def video_co_thumbnail() -> set[str]:
    """Danh sách video ĐÃ có thư mục thumbnail trên máy này."""

    if not THU_MUC_THUMBNAIL.exists():
        return set()

    return {
        p.name for p in THU_MUC_THUMBNAIL.iterdir()
        if p.is_dir() and _LA_TEN_VIDEO.fullmatch(p.name)
    }


@st.cache_resource(show_spinner=False)
def ban_do_thu_muc_keyframe() -> dict[str, Path]:
    """
    Quét một lần toàn bộ cây keyframe.

    Cây thật: raw/keyframes/Keyframes_L23/keyframes/L23_V001/001.jpg
    Vẫn thử cả kiểu không có tầng trung gian phòng khi bản giải nén khác.
    """

    ban_do: dict[str, Path] = {}

    if not THU_MUC_KEYFRAME.exists():
        return ban_do

    for cap1 in THU_MUC_KEYFRAME.iterdir():

        if not cap1.is_dir():
            continue

        for ung_vien in (cap1 / "keyframes", cap1):

            if not ung_vien.is_dir():
                continue

            for thu_muc_video in ung_vien.iterdir():

                if thu_muc_video.is_dir() and _LA_TEN_VIDEO.fullmatch(thu_muc_video.name):
                    ban_do.setdefault(thu_muc_video.name, thu_muc_video)

    return ban_do


def duong_dan_keyframe(video_id: str, n: int) -> Path | None:
    """Trả None thay vì raise — thiếu ảnh không được làm sập giao diện."""

    thu_muc = ban_do_thu_muc_keyframe().get(video_id)

    if thu_muc is None:
        return None

    so_chu_so, duoi = dinh_dang_keyframe()

    duong_dan = thu_muc / f"{n:0{so_chu_so}d}{duoi}"

    return duong_dan if duong_dan.exists() else None


def tinh_trang_anh(video_id: str, n: int) -> str:
    """
    Ba trạng thái khác hẳn nhau, đừng gộp làm một:

      "co"          — có ảnh, hiển thị bình thường
      "chua_tai"    — video thuộc shard người khác, máy này chưa tải. BÌNH THƯỜNG.
      "thieu_anh"   — có shard mà thiếu ảnh. BẤT THƯỜNG, phải báo Nghi.
    """

    if duong_dan_thumbnail(video_id, n).exists():
        return "co"

    if video_id in video_co_thumbnail():
        return "thieu_anh"

    return "chua_tai"


# ============================================================
# 5. LỊCH SỬ VÀ GIỎ NỘP
# ============================================================
#
# Ghi ở tầng giao diện, KHÔNG nhét vào rank/. Lý do: quy tắc
# "hàm hợp đồng không được đụng đĩa mỗi lần gọi".

def _bao_dam_runs() -> None:
    THU_MUC_RUNS.mkdir(parents=True, exist_ok=True)


def ghi_lich_su(ban_ghi: dict) -> None:

    try:
        _bao_dam_runs()

        with TEP_LICH_SU.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(ban_ghi, ensure_ascii=False) + "\n")

    except OSError as loi:
        st.session_state["canh_bao"] = f"Không ghi được lịch sử: {loi}"


def doc_lich_su(gioi_han: int = 30) -> list[dict]:

    if not TEP_LICH_SU.exists():
        return []

    ban_ghi: list[dict] = []

    try:
        with TEP_LICH_SU.open("r", encoding="utf-8-sig") as f:

            for dong in f:

                dong = dong.strip()

                if not dong:
                    continue

                try:
                    ban_ghi.append(json.loads(dong))
                except json.JSONDecodeError:
                    continue

    except OSError:
        return []

    return list(reversed(ban_ghi))[:gioi_han]


def doc_gio_nop() -> dict:

    if not TEP_GIO_NOP.exists():
        return {}

    try:
        return json.loads(TEP_GIO_NOP.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def ghi_gio_nop(gio: dict) -> None:

    try:
        _bao_dam_runs()

        TEP_GIO_NOP.write_text(
            json.dumps(gio, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )

    except OSError as loi:
        st.session_state["canh_bao"] = f"Không ghi được giỏ nộp: {loi}"


def _khoa(video_id: str, frame_idx: int) -> str:
    return f"{video_id}#{frame_idx}"


def dang_trong_gio(ma_cau: str, video_id: str, frame_idx: int) -> bool:

    return any(
        _khoa(x["video_id"], x["frame_idx"]) == _khoa(video_id, frame_idx)
        for x in st.session_state["gio_nop"].get(ma_cau, [])
    )


def bat_tat_dap_an(ma_cau: str, video_id: str, frame_idx: int,
                   pts_time: float, n: int, score: float, nguon: str) -> None:

    gio = st.session_state["gio_nop"]
    danh_sach = gio.setdefault(ma_cau, [])

    khoa = _khoa(video_id, frame_idx)

    con_lai = [x for x in danh_sach if _khoa(x["video_id"], x["frame_idx"]) != khoa]

    if len(con_lai) == len(danh_sach):

        tran = cfg.so_dong_toi_da()

        if len(danh_sach) >= tran:
            st.session_state["canh_bao"] = f"Giỏ nộp đã đủ {tran} dòng."
            return

        danh_sach.append(
            {
                "video_id": video_id,
                "frame_idx": int(frame_idx),
                "pts_time": float(pts_time),
                "n": int(n),
                "score": float(score),
                "nguon": nguon,
            }
        )

    else:
        gio[ma_cau] = con_lai

    ghi_gio_nop(gio)


def xuat_csv_gio_nop(ma_cau: str) -> bytes:
    """
    TODO — cần Ngân chốt.

    submit/formatter.py mới là nơi định nghĩa định dạng bài nộp, nhưng nó
    nhận kết quả từ run_query() chứ không nhận danh sách người chọn tay.
    Tạm xuất video_id,frame_idx ở đây. Đối chiếu lại trước khi nộp thật.
    """

    danh_sach = sorted(
        st.session_state["gio_nop"].get(ma_cau, []),
        key=lambda x: -x["score"],
    )

    dem = io.StringIO()
    ghi = csv.writer(dem, lineterminator="\n")

    for muc in danh_sach[:cfg.so_dong_toi_da()]:
        ghi.writerow([muc["video_id"], muc["frame_idx"]])

    return dem.getvalue().encode("utf-8")


# ============================================================
# 6. NẠP TÀI NGUYÊN
# ============================================================

@st.cache_resource(show_spinner="Đang nạp bảng đối chiếu...")
def lay_frame_map():

    if _load_frame_map is None:
        return None

    try:
        return _load_frame_map()
    except Exception:
        return None


@st.cache_resource(show_spinner="Đang mở kho chữ...")
def lay_kho_chu():
    """
    db_path mặc định của TextSearchIndex là đường dẫn TƯƠNG ĐỐI
    ('index/fts/text.sqlite'), tính từ thư mục đang đứng. Chạy streamlit từ
    D:/AIC2026 thì nó tìm trong thư mục mã nguồn chứ không phải gốc dữ liệu.
    Nên phải truyền tường minh.
    """

    if _TextSearchIndex is None or not TEP_FTS.exists():
        return None

    try:
        return _TextSearchIndex(db_path=TEP_FTS)
    except Exception:
        return None


@st.cache_resource(show_spinner="Đang nạp kho OCR (khớp từ khoá)...")
def lay_ocr_reranker():
    """
    OCRReranker dựng index trong RAM từ derived/ocr/ — hơn 140 nghìn khung hình,
    mất vài giây. cache_resource để chỉ dựng MỘT lần cho cả phiên.

    Đây là nhánh OCR KHÁC với kho chữ FTS: nó khớp từ khoá trích từ câu hỏi và
    lọc box dưới conf 0.40, còn FTS xếp hạng BM25 trên toàn văn. Đo trên bộ dev
    32 câu, hai nhánh BỔ SUNG cho nhau chứ không thay thế: gộp cả hai được 7.000
    điểm, chỉ giữ FTS được 6.200.
    """
    try:
        from aic2026.rank.ocr_rerank import OCRReranker

        return OCRReranker()
    except Exception:
        return None


# ============================================================
# 7. CHUYỂN KẾT QUẢ OCR THÀNH Hit
# ============================================================
#
# TODO — cần Nghi và Ngân chốt, mục 5.3 đang thiếu phần này.
#
# Nhánh CLIP trả list[Hit]. Nhánh OCR trả List[Dict]. Hai kiểu khác nhau
# nên lưới kết quả không dùng chung được. Hàm này là miếng vá tạm.

_KHOA_VIDEO = ("video_id", "video", "vid")
_KHOA_N = ("n", "keyframe", "keyframe_n", "idx")
_KHOA_DIEM = ("score", "bm25", "rank_score", "diem")


def _lay_khoa(d: dict, cac_khoa: tuple[str, ...]):

    for khoa in cac_khoa:

        if khoa in d:
            return d[khoa]

    return None


def hit_tu_dict(d: dict) -> Hit | None:

    video_id = _lay_khoa(d, _KHOA_VIDEO)
    n = _lay_khoa(d, _KHOA_N)

    if video_id is None or n is None:
        return None

    frame_idx = d.get("frame_idx")
    pts_time = d.get("pts_time")

    # Thiếu thì tra ngược từ bảng đối chiếu — dùng hàm có sẵn của Thi.
    if (frame_idx is None or pts_time is None) and _lookup is not None:

        try:
            frame_idx, pts_time = _lookup(str(video_id), int(n))
        except Exception:
            return None

    if frame_idx is None or pts_time is None:
        return None

    diem = _lay_khoa(d, _KHOA_DIEM)

    return Hit(
        video_id=str(video_id),
        n=int(n),
        score=float(diem) if diem is not None else 0.0,
        frame_idx=int(frame_idx),
        pts_time=float(pts_time),
        source="ocr",
    )


# ============================================================
# 8. THEO DÕI TỪNG BƯỚC
# ============================================================

class NhatKyBuoc:
    """
    Một cái spinner duy nhất không cho biết đang kẹt ở đâu. Lần tìm đầu
    tiên mất vài giây vì phải nạp CLIP, nhưng người ngồi máy không phân
    biệt được "đang nạp mô hình" với "máy treo".
    """

    def __init__(self, khi_bat_dau=None, khi_xong=None):
        self.cac_buoc: list[dict] = []
        self._khi_bat_dau = khi_bat_dau
        self._khi_xong = khi_xong

    @contextmanager
    def buoc(self, nhan: str):

        if self._khi_bat_dau is not None:
            self._khi_bat_dau(nhan)

        moc = time.perf_counter()

        try:
            yield

        finally:
            ms = (time.perf_counter() - moc) * 1000

            self.cac_buoc.append({"nhan": nhan, "ms": ms})

            if self._khi_xong is not None:
                self._khi_xong(nhan, ms)

    @property
    def tong_ms(self) -> float:
        return sum(b["ms"] for b in self.cac_buoc)


# ============================================================
# 9. TÌM KIẾM
# ============================================================

# Trọng số RRF lấy từ TRONG_SO_MAC_DINH của rank/hop_nhat.py — để MỘT chỗ, giao
# diện và scripts/benchmark_rrf.py dùng chung, tránh hai bên lệch nhau rồi so
# điểm nhầm.
#
# CẢNH BÁO — các con số đó CHƯA được hiệu chỉnh, xem PR Task 10. Đo trên bộ dev
# 32 câu: ở câu 11, nhánh OCR chạy riêng đưa frame đúng vào top-5 (0.80 điểm),
# nhưng sau khi gộp chỉ còn 0.40. Khi một nguồn trả rất ít ứng viên, ít ứng viên
# là dấu hiệu ĐỘ CHÍNH XÁC CAO chứ không phải yếu — trọng số đang phạt nhầm.

def chay_tim_kiem(cau_hoi: str, cac_nguon: set, cua_so_giay: float,
                  diem_toi_thieu: float, nhat_ky: NhatKyBuoc | None = None) -> dict:
    """
    Lướt tìm — KHÔNG ghi tệp. Chỉ ghi khi người dùng bấm xuất.

    cac_nguon: tập con của {clip, ocr, ocr_fts, asr}. Bật/tắt độc lập nên chạy
    được cả nguồn đơn lẫn mọi tổ hợp, đi qua ĐÚNG MỘT đường ống — giống hệt
    scripts/benchmark_rrf.py, nên kết quả trên giao diện so được với số nghiệm thu.
    """

    nhat_ky = nhat_ky or NhatKyBuoc()

    so_ung_vien = cfg.so_ung_vien_moi_nguon()
    tran_moi_video = cfg.so_anh_toi_da_moi_video()

    so_bo_qua = 0
    chu_ocr: dict[str, str] = {}

    if not cac_nguon:
        raise RuntimeError("Chưa chọn nguồn nào.")

    kho_chu = lay_kho_chu() if {NGUON_OCR_FTS, NGUON_ASR} & cac_nguon else None

    if {NGUON_OCR_FTS, NGUON_ASR} & cac_nguon and kho_chu is None:
        raise RuntimeError(f"Chưa mở được kho chữ. Kiểm tra {TEP_FTS}")

    ocr_engine = lay_ocr_reranker() if NGUON_OCR in cac_nguon else None

    if NGUON_OCR in cac_nguon and ocr_engine is None:
        raise RuntimeError("Không nạp được OCRReranker (kiểm tra derived/ocr/).")

    # Chữ OCR để hiện dưới mỗi ảnh. Lấy từ kho FTS vì nó trả sẵn text theo
    # frame; nhánh OCRReranker không dùng cho việc hiển thị.
    if kho_chu is not None and NGUON_OCR_FTS in cac_nguon:
        try:
            with nhat_ky.buoc("Lấy chữ OCR để hiển thị"):
                for d in kho_chu.search_text(cau_hoi, top_k=so_ung_vien):
                    chu = d.get("text")
                    if chu and d.get("video_id") is not None and d.get("n") is not None:
                        chu_ocr[f'{d["video_id"]}#{d["n"]}'] = str(chu)
        except Exception:
            pass

    nguon = tim_ung_vien_gop(
        ocr_engine=ocr_engine,
        kho_chu=kho_chu,
        dung_clip=NGUON_CLIP in cac_nguon,
        dung_ocr=NGUON_OCR in cac_nguon,
        dung_ocr_fts=NGUON_OCR_FTS in cac_nguon,
        dung_asr=NGUON_ASR in cac_nguon,
        # VIỆC 6: để None thì đọc config/rrf_weights.yaml. Truyền cứng
        # TRONG_SO_MAC_DINH như bản cũ là đè lên tệp đó, và mọi con số đo
        # được ở Việc 6 không tới được giao diện.
        #
        # Giao diện chưa có ô chọn dạng câu nên dùng bộ "mac_dinh" (= bộ của
        # KIS). Thêm ô chọn dạng thì truyền dang_cau vào đây — việc của Thi.
        trong_so=None,
        k_rrf=cfg.rrf_k(),
    )

    with nhat_ky.buoc("Tra cứu + gộp RRF"):
        tho = nguon(cau_hoi, so_ung_vien)

    # Ngưỡng điểm CHỈ dùng được khi đúng một nguồn CLIP: lúc đó score còn là
    # cosine (0..1). Hễ qua RRF thì score là điểm gộp (cỡ 1/(k+hạng), tối đa
    # ~0.016 với k=60) — áp ngưỡng cosine lên đó sẽ xoá sạch kết quả.
    da_cat_nguong = cac_nguon != {NGUON_CLIP}
    so_bi_nguong_tam = 0

    so_tho = len(tho)
    so_bi_nguong = so_bi_nguong_tam

    # Che do gop da cat nguong tren tung nhanh TRUOC khi gop (thang diem RRF
    # khac thang cosine), nen khong cat lai o day.
    if diem_toi_thieu > 0 and not da_cat_nguong:

        with nhat_ky.buoc(f"Cắt ngưỡng điểm ≥ {diem_toi_thieu:.2f}"):
            truoc = len(tho)
            tho = [h for h in tho if h.score >= diem_toi_thieu]
            so_bi_nguong = truoc - len(tho)

    with nhat_ky.buoc(f"Lọc trùng cửa sổ {cua_so_giay:.1f}s"):

        # Trả về TUPLE (danh sách, báo cáo) — không phải list.
        sach, bao_cao = loc_trung(
            tho,
            cua_so_giay=cua_so_giay,
            so_anh_toi_da_moi_video=tran_moi_video,
        )

        sach = list(sach)[:cfg.so_dong_toi_da()]

    with nhat_ky.buoc("Kiểm ảnh thu nhỏ"):

        dem_tinh_trang = {"co": 0, "chua_tai": 0, "thieu_anh": 0}

        for h in sach:
            dem_tinh_trang[tinh_trang_anh(h.video_id, h.n)] += 1

    return {
        "hits": sach,
        "so_tho": so_tho,
        "so_bo_qua": so_bo_qua,
        "chu_ocr": chu_ocr,
        "so_bi_nguong": so_bi_nguong,
        "so_sach": len(sach),
        "dem_tinh_trang": dem_tinh_trang,
        "bao_cao_loc_trung": str(bao_cao),
        "do_tre_ms": nhat_ky.tong_ms,
        "cac_buoc": nhat_ky.cac_buoc,
        "cau_hoi": cau_hoi,
        "che_do": "+".join(sorted(cac_nguon)),
        "cua_so_giay": cua_so_giay,
        "so_ung_vien": so_ung_vien,
    }


# ============================================================
# 10. KHỞI TẠO TRẠNG THÁI
# ============================================================

if "gio_nop" not in st.session_state:
    st.session_state["gio_nop"] = doc_gio_nop()

for khoa_mac_dinh, gia_tri in (
    ("ma_cau", "cau_01"),
    ("trang", 0),
    ("cau_hoi_dien_san", ""),
):
    if khoa_mac_dinh not in st.session_state:
        st.session_state[khoa_mac_dinh] = gia_tri


# ============================================================
# 11. GIAO DIỆN
# ============================================================

_TEP_CSS = Path(__file__).resolve().parent / "style.css"

if _TEP_CSS.exists():
    st.markdown(
        f"<style>{_TEP_CSS.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )
else:
    st.warning(f"Thiếu tệp giao diện: {_TEP_CSS}")


def _ten_nguon(che_do: str) -> str:
    """Đổi mã nguồn nội bộ thành nhãn ngắn, dễ quét trên giao diện."""

    bang_ten = {
        NGUON_CLIP: "CLIP",
        NGUON_OCR: "OCR từ khoá",
        NGUON_OCR_FTS: "OCR BM25",
        NGUON_ASR: "ASR",
    }

    return " + ".join(
        bang_ten.get(nguon, nguon.upper())
        for nguon in che_do.split("+")
        if nguon
    )


def _xoa_gio_hien_tai() -> None:
    ma_cau_hien_tai = str(st.session_state.get("ma_cau", "")).strip()

    if not ma_cau_hien_tai:
        return

    st.session_state["gio_nop"][ma_cau_hien_tai] = []
    ghi_gio_nop(st.session_state["gio_nop"])


if "cau_hoi_input" not in st.session_state:
    st.session_state["cau_hoi_input"] = st.session_state.get("cau_hoi_dien_san", "")


# ============================================================
# 12. ĐẦU TRANG VÀ CẢNH BÁO
# ============================================================

ma_cau = str(st.session_state.get("ma_cau", "")).strip() or "cau_01"
so_da_chon = len(st.session_state["gio_nop"].get(ma_cau, []))

st.markdown(
    '<div class="app-nav">'
    '<div class="nav-brand">'
    '<span class="nav-mark">A</span>'
    '<div class="nav-copy"><strong>AIC 2026</strong>'
    '<small>Video Search Workspace</small></div>'
    '</div>'
    '<div class="nav-status"><span class="status-pulse"></span>'
    'Hệ thống sẵn sàng</div>'
    '</div>'
    '<div class="app-hero">'
    '<div>'
    '<div class="brand-kicker">SMART VIDEO RETRIEVAL</div>'
    '<h1>Tìm đúng khoảnh khắc, chốt đúng frame.</h1>'
    '<p>Một không gian làm việc gọn cho truy vấn, kiểm tra lân cận và xuất đáp án.</p>'
    '</div>'
    f'<div class="hero-progress"><strong>{so_da_chon}</strong>'
    f'<span>frame đã chọn · {html.escape(ma_cau)}</span></div>'
    '</div>',
    unsafe_allow_html=True,
)

_canh_bao_he_thong: list[str] = []

if CANH_BAO_GOC_DU_LIEU:
    _canh_bao_he_thong.append(CANH_BAO_GOC_DU_LIEU)

try:
    _canh_bao_he_thong.extend(
        f"Cấu hình: {canh_bao}" for canh_bao in cfg.canh_bao_cau_hinh()
    )
except Exception as loi:
    _canh_bao_he_thong.append(f"Không đọc được cảnh báo cấu hình: {loi}")

if _canh_bao_he_thong:
    with st.expander(
        f"Kiểm tra thiết lập ({len(_canh_bao_he_thong)} cảnh báo)",
        expanded=False,
    ):
        for canh_bao in _canh_bao_he_thong:
            st.warning(canh_bao)


# ============================================================
# 13. KHUNG TÌM KIẾM
# ============================================================

co_kho_chu = _TextSearchIndex is not None and TEP_FTS.exists()

if not co_kho_chu:
    st.session_state["ng_ocr_fts"] = False
    st.session_state["ng_asr"] = False

with st.form("form_tim_kiem", clear_on_submit=False):
    st.markdown(
        '<div class="form-intro">'
        '<span class="form-step">01</span>'
        '<div><strong>Nhập truy vấn</strong>'
        '<small>Mô tả càng cụ thể về người, hành động, vật thể và bối cảnh càng tốt.</small>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    cot_ma, cot_cau_hoi, cot_tim = st.columns([1.35, 5.8, 1.35], gap="medium")

    with cot_ma:
        ma_cau_nhap = st.text_input(
            "Mã câu",
            key="ma_cau",
            placeholder="cau_01",
            help="Mã này dùng để tách riêng danh sách frame đã chọn.",
        )

    with cot_cau_hoi:
        cau_hoi = st.text_area(
            "Mô tả cần tìm",
            key="cau_hoi_input",
            placeholder=(
                "Ví dụ: Một người phụ nữ mặc áo dài đỏ đang phát biểu "
                "trên bục, phía sau có nhiều cây xanh"
            ),
            height=88,
        )

    with cot_tim:
        st.markdown('<div class="submit-spacer"></div>', unsafe_allow_html=True)
        da_gui = st.form_submit_button(
            "Tìm kiếm",
            type="primary",
            **KW_NUT,
        )

    with st.expander("Tùy chọn tìm kiếm", expanded=False):
        st.caption("Nguồn dữ liệu")

        cot_clip, cot_ocr, cot_fts, cot_asr = st.columns(4)

        bat_clip = cot_clip.checkbox(
            "Hình ảnh (CLIP)",
            value=True,
            key="ng_clip",
            help="Tìm theo độ tương đồng ngữ nghĩa hình ảnh - văn bản.",
        )
        bat_ocr = cot_ocr.checkbox(
            "OCR từ khoá",
            value=False,
            key="ng_ocr",
            help="Khớp từ khoá xuất hiện trong khung hình.",
        )
        bat_ocr_fts = cot_fts.checkbox(
            "OCR BM25",
            value=co_kho_chu,
            key="ng_ocr_fts",
            disabled=not co_kho_chu,
            help="Tìm văn bản trên ảnh bằng chỉ mục BM25.",
        )
        bat_asr = cot_asr.checkbox(
            "Lời nói (ASR)",
            value=co_kho_chu,
            key="ng_asr",
            disabled=not co_kho_chu,
            help="Tìm nội dung được nói trong video.",
        )

        cot_loc, cot_nguong = st.columns(2)

        cua_so_giay = cot_loc.slider(
            "Khoảng gộp các frame gần nhau",
            min_value=0.0,
            max_value=30.0,
            value=float(cfg.cua_so_giay()),
            step=0.5,
            format="%.1f giây",
            help="Giảm các kết quả gần như trùng nhau trong cùng video.",
        )
        diem_toi_thieu = cot_nguong.slider(
            "Ngưỡng CLIP tối thiểu",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.01,
            help="Chỉ áp dụng khi tìm bằng riêng nguồn CLIP.",
        )

        if not co_kho_chu:
            st.caption("Kho chữ chưa sẵn sàng nên OCR BM25 và ASR đang tắt.")

    cac_nguon = set()

    if bat_clip:
        cac_nguon.add(NGUON_CLIP)
    if bat_ocr:
        cac_nguon.add(NGUON_OCR)
    if bat_ocr_fts:
        cac_nguon.add(NGUON_OCR_FTS)
    if bat_asr:
        cac_nguon.add(NGUON_ASR)


# ============================================================
# 14. CHẠY TÌM KIẾM
# ============================================================

if da_gui:
    if not ma_cau_nhap.strip():
        st.warning("Hãy nhập mã câu trước khi tìm kiếm.")

    elif not cau_hoi.strip():
        st.warning("Hãy nhập mô tả khoảnh khắc cần tìm.")

    elif not cac_nguon:
        st.warning("Hãy bật ít nhất một nguồn tìm kiếm.")

    else:
        try:
            with st.spinner("Đang tìm và xếp hạng các keyframe phù hợp..."):
                ket_qua_moi = chay_tim_kiem(
                    cau_hoi.strip(),
                    cac_nguon,
                    cua_so_giay,
                    diem_toi_thieu,
                    nhat_ky=NhatKyBuoc(),
                )

            st.session_state["ket_qua"] = ket_qua_moi
            st.session_state["trang"] = 0
            st.session_state.pop("lan_can", None)
            st.session_state.pop("phong_to", None)

            ghi_lich_su(
                {
                    "luc": datetime.now().isoformat(timespec="seconds"),
                    "ma_cau": ma_cau_nhap.strip(),
                    "cau_hoi": ket_qua_moi["cau_hoi"],
                    "che_do": ket_qua_moi["che_do"],
                    "cua_so_giay": ket_qua_moi["cua_so_giay"],
                    "so_ung_vien": ket_qua_moi["so_ung_vien"],
                    "so_tho": ket_qua_moi["so_tho"],
                    "so_sach": ket_qua_moi["so_sach"],
                    "dem_tinh_trang": ket_qua_moi["dem_tinh_trang"],
                    "do_tre_ms": round(ket_qua_moi["do_tre_ms"], 1),
                    "cac_buoc": [
                        {"nhan": buoc["nhan"], "ms": round(buoc["ms"], 1)}
                        for buoc in ket_qua_moi["cac_buoc"]
                    ],
                }
            )

            st.success(
                f"Đã tìm thấy {ket_qua_moi['so_sach']} kết quả "
                f"trong {ket_qua_moi['do_tre_ms'] / 1000:.2f} giây."
            )

        except Exception as loi:
            st.error(f"Tìm kiếm thất bại: {type(loi).__name__}: {loi}")

            cac_mo_dun_loi = {
                ten: tinh_trang
                for ten, tinh_trang in TRANG_THAI_MO_DUN.items()
                if tinh_trang != "OK"
            }

            if cac_mo_dun_loi:
                with st.expander("Chi tiết để kiểm tra"):
                    for ten, tinh_trang in cac_mo_dun_loi.items():
                        st.code(f"{ten}: {tinh_trang}", language=None)

if "canh_bao" in st.session_state:
    st.warning(st.session_state.pop("canh_bao"))


# ============================================================
# 15. ĐÁP ÁN ĐÃ CHỌN
# ============================================================

ma_cau = str(st.session_state.get("ma_cau", "")).strip() or "cau_01"
danh_sach_nop = st.session_state["gio_nop"].get(ma_cau, [])

with st.container(border=True):
    st.markdown('<div class="selection-panel-marker"></div>', unsafe_allow_html=True)

    cot_tieu_de, cot_xoa, cot_tai = st.columns([5.2, 1.25, 1.35], **KW_CANH)

    cot_tieu_de.markdown(
        '<div class="section-heading">'
        '<span class="form-step">02</span>'
        '<div><strong>Đáp án đã chọn</strong>'
        f'<small>{html.escape(ma_cau)} · {len(danh_sach_nop)} / '
        f'{cfg.so_dong_toi_da()} frame</small></div></div>',
        unsafe_allow_html=True,
    )

    if cot_xoa.button(
        "Xóa hết",
        key="xoa_gio_hien_tai",
        disabled=not danh_sach_nop,
        on_click=_xoa_gio_hien_tai,
        **KW_NUT,
    ):
        st.rerun()

    cot_tai.download_button(
        "Tải CSV",
        data=xuat_csv_gio_nop(ma_cau),
        file_name=f"{ma_cau}.csv",
        mime="text/csv",
        disabled=not danh_sach_nop,
        help="Xuất đúng các frame đang được chọn, không chạy lại tìm kiếm.",
        **KW_TAI,
    )

    if danh_sach_nop:
        with st.expander(f"Xem {len(danh_sach_nop)} frame đã chọn", expanded=False):
            bang_gio = [
                {
                    "video_id": muc["video_id"],
                    "frame_idx": muc["frame_idx"],
                    "thời điểm": f"{muc['pts_time']:.2f}s",
                    "keyframe": muc["n"],
                    "nguồn": muc["nguon"],
                }
                for muc in sorted(danh_sach_nop, key=lambda x: -x["score"])
            ]
            st.dataframe(bang_gio, hide_index=True, **KW_BANG)
    else:
        st.caption("Chọn một kết quả bên dưới để thêm frame vào đáp án.")


# ============================================================
# 16. XEM LÂN CẬN VÀ ẢNH GỐC
# ============================================================

def mo_lan_can(video_id: str, n: int) -> None:
    st.session_state["lan_can"] = {"video_id": video_id, "n": n}


def mo_phong_to(
    duong_dan: str,
    video_id: str,
    n: int,
    frame_idx: int,
    pts_time: float,
    score: float,
) -> None:
    st.session_state["phong_to"] = {
        "duong_dan": duong_dan,
        "video_id": video_id,
        "n": n,
        "frame_idx": frame_idx,
        "pts_time": pts_time,
        "score": score,
    }


def ve_phong_to(muc: dict) -> None:
    hien_anh(muc["duong_dan"])

    cac_cot = st.columns(4)
    cac_cot[0].metric("Video", muc["video_id"])
    cac_cot[1].metric("Frame", muc["frame_idx"])
    cac_cot[2].metric("Thời điểm", f"{muc['pts_time']:.2f}s")
    cac_cot[3].metric("Keyframe", f"n={muc['n']}")


if "phong_to" in st.session_state:
    muc_phong_to = st.session_state["phong_to"]

    with st.container(border=True):
        st.markdown('<div class="viewer-panel-marker"></div>', unsafe_allow_html=True)

        cot_tieu_de, cot_dong = st.columns([6, 1], **KW_CANH)
        cot_tieu_de.markdown(
            '<div class="panel-title">Ảnh keyframe gốc</div>'
            f'<div class="panel-subtitle">{html.escape(muc_phong_to["video_id"])}</div>',
            unsafe_allow_html=True,
        )

        if cot_dong.button("Đóng", key="dong_phong_to", **KW_NUT):
            st.session_state.pop("phong_to", None)
            st.rerun()

        ve_phong_to(muc_phong_to)


if "lan_can" in st.session_state:
    thong_tin = st.session_state["lan_can"]
    bang = lay_frame_map()

    with st.container(border=True):
        st.markdown('<div class="neighbor-panel-marker"></div>', unsafe_allow_html=True)

        cot_tieu_de, cot_dong = st.columns([6, 1], **KW_CANH)
        cot_tieu_de.markdown(
            '<div class="panel-title">Các keyframe lân cận</div>'
            f'<div class="panel-subtitle">{html.escape(thong_tin["video_id"])} '
            f'· quanh n={thong_tin["n"]}</div>',
            unsafe_allow_html=True,
        )

        if cot_dong.button("Đóng", key="dong_lan_can", **KW_NUT):
            st.session_state.pop("lan_can", None)
            st.rerun()

        if bang is None:
            st.info("Chưa nạp được bảng đối chiếu nên không thể xem lân cận.")

        else:
            cua_so = bang[
                (bang["video_id"] == thong_tin["video_id"])
                & (bang["n"] >= thong_tin["n"] - SO_LAN_CAN)
                & (bang["n"] <= thong_tin["n"] + SO_LAN_CAN)
            ].sort_values("n")

            if cua_so.empty:
                st.info("Không tìm thấy keyframe lân cận trong bảng đối chiếu.")

            for bat_dau in range(0, len(cua_so), 5):
                nhom = cua_so.iloc[bat_dau:bat_dau + 5]
                cac_cot = st.columns(5, gap="small")

                for cot, (_, dong) in zip(cac_cot, nhom.iterrows()):
                    with cot:
                        duong_dan = duong_dan_thumbnail(
                            str(dong["video_id"]),
                            int(dong["n"]),
                        )

                        if duong_dan.exists():
                            hien_anh(duong_dan)
                        else:
                            st.markdown(
                                '<div class="neighbor-placeholder">chưa có ảnh</div>',
                                unsafe_allow_html=True,
                            )

                        o_giua = int(dong["n"]) == thong_tin["n"]

                        st.markdown(
                            f'<div class="neighbor-meta{" is-center" if o_giua else ""}">'
                            f'{"Đang xem · " if o_giua else ""}'
                            f'n={int(dong["n"])}<br>'
                            f'frame {int(dong["frame_idx"])} · '
                            f'{float(dong["pts_time"]):.2f}s</div>',
                            unsafe_allow_html=True,
                        )

                        da_chon = dang_trong_gio(
                            ma_cau,
                            str(dong["video_id"]),
                            int(dong["frame_idx"]),
                        )

                        st.button(
                            "Bỏ chọn" if da_chon else "Chọn",
                            key=f"chon_lc_{dong['video_id']}_{int(dong['n'])}",
                            type="primary" if not da_chon else "secondary",
                            on_click=bat_tat_dap_an,
                            args=(
                                ma_cau,
                                str(dong["video_id"]),
                                int(dong["frame_idx"]),
                                float(dong["pts_time"]),
                                int(dong["n"]),
                                0.0,
                                "lân cận",
                            ),
                            **KW_NUT,
                        )


# ============================================================
# 17. LƯỚI KẾT QUẢ
# ============================================================

if "ket_qua" in st.session_state:
    ket_qua = st.session_state["ket_qua"]
    dem = ket_qua.get("dem_tinh_trang", {})

    cot_tieu_de, cot_bo_loc = st.columns([6, 1.7], **KW_CANH)

    cot_tieu_de.markdown(
        '<div class="results-heading">'
        '<span class="form-step">03</span>'
        '<div><strong>Kết quả tìm kiếm</strong>'
        f'<small>{_ten_nguon(ket_qua["che_do"])} · '
        f'{ket_qua["so_sach"]} kết quả · '
        f'{ket_qua["do_tre_ms"] / 1000:.2f}s</small></div></div>',
        unsafe_allow_html=True,
    )

    chi_video_co_anh = cot_bo_loc.toggle(
        "Chỉ ảnh có sẵn",
        value=False,
        key="chi_video_co_anh",
        help="Ẩn các kết quả thuộc shard chưa được tải trên máy này.",
    )

    hits = list(ket_qua["hits"])

    if chi_video_co_anh:
        hits = [
            hit for hit in hits
            if tinh_trang_anh(hit.video_id, hit.n) == "co"
        ]

    so_trang = max(
        1,
        (len(hits) + SO_KET_QUA_MOI_TRANG - 1) // SO_KET_QUA_MOI_TRANG,
    )
    st.session_state["trang"] = min(
        int(st.session_state.get("trang", 0)),
        so_trang - 1,
    )

    st.markdown(
        '<div class="query-summary">'
        f'<span>“{html.escape(ket_qua["cau_hoi"]) }”</span>'
        f'<small>{len(hits)} đang hiển thị · '
        f'{dem.get("co", 0)} có ảnh · '
        f'{dem.get("chua_tai", 0)} chưa tải</small>'
        '</div>',
        unsafe_allow_html=True,
    )

    if dem.get("thieu_anh", 0):
        st.warning(
            f"{dem['thieu_anh']} kết quả thuộc video đã tải nhưng thiếu ảnh tương ứng."
        )

    if not hits:
        st.info("Không có kết quả phù hợp với bộ lọc hiện tại.")

    else:
        dau = st.session_state["trang"] * SO_KET_QUA_MOI_TRANG
        hits_trang = hits[dau:dau + SO_KET_QUA_MOI_TRANG]

        for khoi in range(0, len(hits_trang), SO_COT):
            cac_cot = st.columns(SO_COT, gap="small")

            for lech, (cot, hit) in enumerate(
                zip(cac_cot, hits_trang[khoi:khoi + SO_COT])
            ):
                hang = dau + khoi + lech + 1
                da_chon = dang_trong_gio(
                    ma_cau,
                    hit.video_id,
                    int(hit.frame_idx),
                )
                trang_thai_anh = tinh_trang_anh(hit.video_id, hit.n)

                with cot:
                    # Không bọc thêm st.container(border=True): ở Streamlit 1.36,
                    # selector :has() sẽ bắt cả hai lớp wrapper và tạo khung kép.
                    # Cột kết quả là chính thẻ; CSS nhận diện cột qua marker này.
                    lop_chon = " is-selected" if da_chon else ""
                    st.markdown(
                        f'<div class="result-card-marker{lop_chon}"></div>',
                        unsafe_allow_html=True,
                    )

                    if trang_thai_anh == "co":
                        hien_anh(duong_dan_thumbnail(hit.video_id, hit.n))
                    elif trang_thai_anh == "chua_tai":
                        st.markdown(
                            '<div class="image-placeholder">'
                            '<span>Shard chưa tải</span></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="image-placeholder is-missing">'
                            '<span>Thiếu ảnh trong shard</span></div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        '<div class="card-topline">'
                        f'<span class="rank-badge">#{hang}</span>'
                        f'<span class="source-badge">{html.escape(str(hit.source))}</span>'
                        '</div>'
                        f'<div class="video-id">{html.escape(hit.video_id)}</div>'
                        '<div class="frame-meta">'
                        '<div class="frame-meta-primary">'
                        f'<strong>frame {int(hit.frame_idx)}</strong>'
                        f'<span class="score-value">{float(hit.score):.4f}</span>'
                        '</div>'
                        '<div class="frame-meta-secondary">'
                        f'<span>n={int(hit.n)} · {float(hit.pts_time):.2f}s</span>'
                        '</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    chu = ket_qua.get("chu_ocr", {}).get(
                        f"{hit.video_id}#{hit.n}"
                    )

                    if chu:
                        chu_gon = chu if len(chu) <= 120 else chu[:120] + "…"
                        st.markdown(
                            f'<div class="ocr-snippet">{html.escape(chu_gon)}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        # Giữ một vùng OCR rỗng để các nút ở bốn thẻ cùng hàng
                        # luôn thẳng nhau, không cần đặt position hay chiều cao âm.
                        st.markdown(
                            '<div class="ocr-snippet is-empty" aria-hidden="true">'
                            '&nbsp;</div>',
                            unsafe_allow_html=True,
                        )

                    st.button(
                        "✓ Đã chọn" if da_chon else "Chọn frame này",
                        key=f"chon_{hang}_{hit.video_id}_{hit.n}",
                        type="secondary" if da_chon else "primary",
                        on_click=bat_tat_dap_an,
                        args=(
                            ma_cau,
                            hit.video_id,
                            int(hit.frame_idx),
                            float(hit.pts_time),
                            int(hit.n),
                            float(hit.score),
                            hit.source,
                        ),
                        **KW_NUT,
                    )

                    cot_lan_can, cot_anh_goc = st.columns(2, gap="small")

                    cot_lan_can.button(
                        "Lân cận",
                        key=f"lc_{hang}_{hit.video_id}_{hit.n}",
                        on_click=mo_lan_can,
                        args=(hit.video_id, int(hit.n)),
                        **KW_NUT,
                    )

                    duong_dan_goc = duong_dan_keyframe(hit.video_id, hit.n)

                    if duong_dan_goc is not None:
                        cot_anh_goc.button(
                            "Ảnh gốc",
                            key=f"pt_{hang}_{hit.video_id}_{hit.n}",
                            on_click=mo_phong_to,
                            args=(
                                str(duong_dan_goc),
                                hit.video_id,
                                int(hit.n),
                                int(hit.frame_idx),
                                float(hit.pts_time),
                                float(hit.score),
                            ),
                            **KW_NUT,
                        )
                    else:
                        cot_anh_goc.button(
                            "Ảnh gốc",
                            key=f"pt_x_{hang}_{hit.video_id}_{hit.n}",
                            disabled=True,
                            **KW_NUT,
                        )

        if so_trang > 1:
            def _lui_trang() -> None:
                st.session_state["trang"] = max(
                    0,
                    st.session_state["trang"] - 1,
                )

            def _toi_trang() -> None:
                st.session_state["trang"] = min(
                    so_trang - 1,
                    st.session_state["trang"] + 1,
                )

            cot_truoc, cot_so_trang, cot_sau = st.columns(
                [1.25, 2.4, 1.25],
                **KW_CANH,
            )

            cot_truoc.button(
                "← Trang trước",
                key="trang_lui",
                disabled=st.session_state["trang"] <= 0,
                on_click=_lui_trang,
                **KW_NUT,
            )
            cot_so_trang.markdown(
                f'<div class="page-indicator">Trang '
                f'{st.session_state["trang"] + 1} / {so_trang}</div>',
                unsafe_allow_html=True,
            )
            cot_sau.button(
                "Trang sau →",
                key="trang_toi",
                disabled=st.session_state["trang"] >= so_trang - 1,
                on_click=_toi_trang,
                **KW_NUT,
            )

else:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">⌕</div>'
        '<strong>Sẵn sàng tìm kiếm</strong>'
        '<p>Nhập mã câu và mô tả khoảnh khắc. Kết quả sẽ xuất hiện ở đây để '
        'bạn xem lân cận, mở ảnh gốc và chọn frame nộp.</p>'
        '</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="app-footer">AIC 2026 · Video Search Workspace</div>',
    unsafe_allow_html=True,
)