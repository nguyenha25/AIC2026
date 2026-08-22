"""
Việc 12 — GHÉP THỜI GIAN CHO TRAKE: TỪ MÔ TẢ SỰ KIỆN SUY RA THỨ TỰ KHUNG HÌNH.

ĐỌC ĐOẠN NÀY TRƯỚC KHI TRÔNG CHỜ ĐIỂM TĂNG
------------------------------------------
TRAKE đang 0,000 điểm trên bộ dev vì lý do ở TẦNG DỮ LIỆU, không phải tầng
xếp hạng: keyframe cách nhau 2,6-3,5 giây, còn cửa sổ đáp án chỉ khoảng 0,4
giây. Ở vòng p1, hai trong ba sự kiện KHÔNG có keyframe nào rơi vào khoảng
đúng. Không thuật toán ghép nào cứu được chuyện đó.

Tệp này chỉ có tác dụng khi đã có khung dày (Việc 9). Chạy nó trên keyframe
thưa vẫn ra kết quả, vẫn đúng định dạng, và vẫn gần như 0 điểm — `bao_cao`
trả về `du_day` để nói thẳng điều đó thay vì để người đọc tưởng đã xong.

QUY ƯỚC TÊN TỆP KHUNG DÀY (Việc 9 phải sinh ra đúng dạng này)
-------------------------------------------------------------
    derived/frames_dense/<video_id>/<frame_idx 6 chữ số>.jpg

Đặt tên theo frame_idx chứ KHÔNG theo số thứ tự: số thứ tự là thứ đã gây ra
cái bẫy n với frame_idx. Tên tệp mang thẳng số phải nộp thì không còn chỗ nào
để nhầm.

THUẬT TOÁN
----------
1. Tách mô tả thành N sự kiện theo dấu phẩy / "rồi" / "sau đó".
2. Chọn MỘT video. Sai video là 0 điểm ngay, không tính trung bình — nên đây
   là quyết định đắt nhất, và hàm trả về cả danh sách video á quân để người
   ngồi máy soi lại.
3. Chấm điểm mọi khung hình của video đó với TỪNG sự kiện -> ma trận N × F.
4. Quy hoạch động: chọn dãy khung TĂNG DẦN theo thời gian, mỗi sự kiện một
   khung, tổng điểm lớn nhất. Ràng buộc tăng dần là bắt buộc — BTC chấm theo
   thứ tự: mốc j phải nằm trong khoảng của sự kiện thứ j.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# Từ nối báo hiệu sự kiện tiếp theo.
_TU_NOI = [
    "rồi", "sau đó", "tiếp theo", "tiếp đó", "kế tiếp", "cuối cùng",
    "trước tiên", "đầu tiên", "sau cùng",
]

# Khoảng cách khung tối đa còn coi là "dày". Cửa sổ đáp án ~0,4 giây.
NGUONG_DAY_GIAY = 0.5

# Cụm tiếng Anh đã dùng để chấm điểm ở lần chạy gần nhất. Chỉ để soi lại —
# ba mô tả sự kiện mà rút về cùng một cụm thì ma trận điểm có ba hàng giống
# hệt nhau, và quy hoạch động chỉ còn cách lấy đỉnh cao nhất với hai khung kề.
_CUM_GAN_NHAT: list[str] = []


@dataclass
class MocTrake:
    """Một mốc đã chọn cho một sự kiện."""

    thu_tu: int
    mo_ta: str
    frame_idx: int
    pts_time: float
    diem: float


@dataclass
class KetQuaTrake:
    video_id: str
    moc: list[MocTrake]
    du_day: bool
    khoang_cach_khung_giay: float
    so_khung_ung_vien: int = 0
    # Điểm cao nhất trừ điểm thấp nhất, trung bình trên các sự kiện. Nhỏ nghĩa
    # là mô hình nhìn mọi khung gần như giống nhau — mọi thứ phía sau chỉ là
    # xếp hạng nhiễu.
    do_trai_diem: float = 0.0
    su_kien: list[str] = field(default_factory=list)
    cum_tieng_anh: list[str] = field(default_factory=list)
    # Top-5 khung theo điểm của TỪNG sự kiện, để soi xem các hàng ma trận có
    # bị giống hệt nhau không.
    dinh_moi_su_kien: list[list[tuple[int, float]]] = field(default_factory=list)
    video_a_quan: list[tuple[str, float]] = field(default_factory=list)
    canh_bao: list[str] = field(default_factory=list)

    @property
    def frame_ids(self) -> list[int]:
        """Dãy số để nộp — đã sắp tăng dần theo thời gian."""
        return [m.frame_idx for m in self.moc]

    def tom_tat(self) -> str:
        dong = [
            f"{self.video_id} | {len(self.moc)} mốc | "
            f"khoảng cách khung {self.khoang_cach_khung_giay:.2f}s "
            f"({'ĐỦ DÀY' if self.du_day else 'QUÁ THƯA — điểm sẽ gần 0'})"
        ]
        for m in self.moc:
            dong.append(
                f"   {m.thu_tu + 1}. {m.mo_ta[:44]:<44} "
                f"frame_idx={m.frame_idx:<8} giây={m.pts_time:>8.2f} "
                f"điểm={m.diem:.3f}"
            )
        for c in self.canh_bao:
            dong.append(f"   ! {c}")
        return "\n".join(dong)


# ---------------------------------------------------------------------------
# 1. Tách mô tả thành các sự kiện
# ---------------------------------------------------------------------------

def tach_su_kien(mo_ta: str, so_moc: int | None = None) -> list[str]:
    """'chạy đà, giậm nhảy, bay qua xà, tiếp đất' -> 4 cụm.

    so_moc=None: lấy đúng số cụm tách được.
    so_moc=k   : ép về k cụm. Thừa thì gộp hai cụm cuối, thiếu thì lặp cụm
                 cuối — vì BTC đòi ĐỦ số mốc, thiếu mốc là mất điểm mốc đó.
    """
    van_ban = mo_ta.strip()

    # Bỏ phần dẫn trước dấu hai chấm: "Vận động viên nhảy cao: chạy đà, ..."
    if ":" in van_ban:
        van_ban = van_ban.split(":", 1)[1]

    for tu in _TU_NOI:
        van_ban = re.sub(rf"\b{re.escape(tu)}\b", ",", van_ban, flags=re.IGNORECASE)

    # Bỏ đánh số đầu cụm. Đề THẬT viết "(1) nhấc bánh khỏi rổ" — có ngoặc mở.
    # Bản đầu chỉ bắt "1)" nên tiền tố "(1) " còn nguyên, đi thẳng vào phần
    # chấm điểm và làm hỏng cụm tiếng Anh. Đã cắn ở câu 15 và 16 bộ dev.
    cum = [
        re.sub(r"^\s*[(\[]?\s*\d+\s*[).\]\-]\s*", "", c).strip(" ,.;")
        for c in re.split(r"[,;]|->|→", van_ban)
    ]
    cum = [c for c in cum if len(c) >= 2]

    if not cum:
        cum = [mo_ta.strip()]

    if so_moc is None:
        return cum

    while len(cum) > so_moc:
        cum[-2] = f"{cum[-2]} {cum[-1]}"
        cum.pop()
    while len(cum) < so_moc:
        cum.append(cum[-1])

    return cum


# ---------------------------------------------------------------------------
# 2. Chọn video
# ---------------------------------------------------------------------------

def chon_video(ung_vien: Sequence, so_a_quan: int = 3) -> tuple[str, list[tuple[str, float]]]:
    """Chọn video có nhiều ứng viên mạnh nhất.

    Cộng điểm theo HẠNG (1/(60+hạng)) chứ không cộng score thô: score của các
    nhánh khác nhau không cùng thang, cộng thẳng là để nhánh có thang lớn
    quyết định hết.
    """
    diem: dict[str, float] = {}
    for hang, h in enumerate(ung_vien, start=1):
        diem[str(h.video_id)] = diem.get(str(h.video_id), 0.0) + 1.0 / (60 + hang)

    xep = sorted(diem.items(), key=lambda t: (-t[1], t[0]))
    if not xep:
        raise ValueError("Không có ứng viên nào để chọn video.")
    return xep[0][0], xep[1 : 1 + so_a_quan]


# ---------------------------------------------------------------------------
# 3. Danh sách khung hình của một video
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Khung:
    frame_idx: int
    pts_time: float
    duong_dan: Path


def liet_ke_khung(video_id: str, uu_tien_khung_day: bool = True) -> tuple[list[Khung], bool]:
    """Mọi khung hình dùng được của một video. Trả (danh sách, có_dùng_khung_dày).

    Ưu tiên derived/frames_dense/ nếu có — đó là điều kiện duy nhất để TRAKE
    có cơ hội ăn điểm.
    """
    from .paths import FRAMES_DENSE_DIR, keyframe_image

    if uu_tien_khung_day:
        thu_muc = FRAMES_DENSE_DIR / video_id
        if thu_muc.is_dir():
            tep = sorted(thu_muc.glob("*.jpg"))
            if tep:
                fps = _fps_cua_video(video_id)
                khung = []
                for t in tep:
                    khop = re.search(r"(\d+)", t.stem)
                    if not khop:
                        continue
                    fi = int(khop.group(1))
                    khung.append(Khung(fi, fi / fps if fps else 0.0, t))
                if khung:
                    return sorted(khung, key=lambda k: k.frame_idx), True

    # Đọc thẳng frame_map thay vì mượn hàm riêng tư của kf_index.py: đó là tệp
    # của Thi, và gọi hàm gạch dưới của người khác thì gói này không trộn độc
    # lập được — đã cắn một lần khi chạy Việc 12 trên nhánh chưa có gói 02.
    nhom = _bang_cua_video(video_id)
    khung = [
        Khung(int(r.frame_idx), float(r.pts_time), keyframe_image(video_id, int(r.n)))
        for r in nhom.itertuples()
    ]
    return khung, False


def _bang_cua_video(video_id: str):
    """Các dòng frame_map của MỘT video, đã sắp theo n tăng dần."""
    from .frame_map import load_frame_map

    bang = load_frame_map()
    nhom = bang[bang["video_id"] == video_id]
    if nhom.empty:
        raise KeyError(
            f"{video_id} không có trong frame_map. Máy này đã tải "
            "map-keyframes của video đó chưa?"
        )
    return nhom.sort_values("n")


def _fps_cua_video(video_id: str) -> float:
    return float(_bang_cua_video(video_id)["fps"].iloc[0])


def khoang_cach_trung_binh(khung: Sequence[Khung]) -> float:
    if len(khung) < 2:
        return float("inf")
    pts = np.array([k.pts_time for k in khung], dtype=np.float64)
    return float(np.median(np.diff(pts)))


# ---------------------------------------------------------------------------
# 4. Quy hoạch động — chọn dãy tăng dần
# ---------------------------------------------------------------------------

def ghep_tang_dan(ma_tran_diem: np.ndarray) -> list[int]:
    """Chọn cột tăng dần, mỗi hàng một cột, tổng điểm lớn nhất.

    ma_tran_diem: (số sự kiện N) × (số khung F).
    Trả về danh sách N chỉ số cột, tăng NGHIÊM NGẶT.

    Điều kiện tăng nghiêm ngặt là bắt buộc: BTC chấm mốc thứ j theo khoảng
    của sự kiện thứ j, nên hai sự kiện dùng chung một khung thì chắc chắn
    một trong hai sai.

    O(N × F). Với F cỡ vài nghìn khung dày thì vẫn tức thì.
    """
    n_su_kien, n_khung = ma_tran_diem.shape

    if n_khung < n_su_kien:
        raise ValueError(
            f"Chỉ có {n_khung} khung cho {n_su_kien} sự kiện. Không đủ để "
            "chọn dãy tăng dần — cần trích khung dày (Việc 9)."
        )

    AM_VO_CUC = -np.inf
    tot = np.full((n_su_kien, n_khung), AM_VO_CUC)
    tu_dau = np.full((n_su_kien, n_khung), -1, dtype=np.int64)

    tot[0] = ma_tran_diem[0]

    for i in range(1, n_su_kien):
        # tot_nhat_truoc[j] = max(tot[i-1][:j]) — cột j phải LỚN HƠN cột trước
        chay = AM_VO_CUC
        chi_so_chay = -1
        for j in range(n_khung):
            if j > 0:
                if tot[i - 1][j - 1] > chay:
                    chay = tot[i - 1][j - 1]
                    chi_so_chay = j - 1
            if chay > AM_VO_CUC:
                tot[i][j] = chay + ma_tran_diem[i][j]
                tu_dau[i][j] = chi_so_chay

    cuoi = int(np.argmax(tot[n_su_kien - 1]))
    if not np.isfinite(tot[n_su_kien - 1][cuoi]):
        raise ValueError("Không dựng được dãy tăng dần nào.")

    duong: list[int] = [cuoi]
    for i in range(n_su_kien - 1, 0, -1):
        cuoi = int(tu_dau[i][cuoi])
        duong.append(cuoi)
    duong.reverse()
    return duong


# ---------------------------------------------------------------------------
# Cửa chính
# ---------------------------------------------------------------------------

def ghep(
    mo_ta: str,
    ung_vien: Sequence,
    so_moc: int = 4,
    cham_diem: Callable[[list[str], list[Khung]], np.ndarray] | None = None,
    video_id: str | None = None,
    uu_tien_khung_day: bool = True,
    bo_ma_hoa: str = "b32",
) -> KetQuaTrake:
    """Mô tả sự kiện + ứng viên -> dãy frame_idx theo thứ tự.

    cham_diem: hàm (danh sách mô tả, danh sách khung) -> ma trận N × F.
               None thì dùng CLIP qua Reranker (mô hình mạnh, chỉ chạy trên
               khung của MỘT video nên chịu được).
    """
    su_kien = tach_su_kien(mo_ta, so_moc)

    a_quan: list[tuple[str, float]] = []
    if video_id is None:
        video_id, a_quan = chon_video(ung_vien)

    khung, dung_khung_day = liet_ke_khung(video_id, uu_tien_khung_day)
    if not khung:
        raise ValueError(f"Không liệt kê được khung hình nào của {video_id}.")

    khoang_cach = khoang_cach_trung_binh(khung)
    du_day = khoang_cach <= NGUONG_DAY_GIAY

    canh_bao: list[str] = []
    if not du_day:
        canh_bao.append(
            f"Khung cách nhau {khoang_cach:.2f} giây, cửa sổ đáp án ~0,4 giây. "
            "Điểm TRAKE sẽ gần 0 bất kể ghép tốt đến đâu. Cần Việc 9 "
            "(trích khung dày) cho video này."
        )
    if not dung_khung_day:
        canh_bao.append(
            f"Đang dùng keyframe thưa của BTC. Chưa thấy "
            f"derived/frames_dense/{video_id}/"
        )

    if cham_diem is None:
        def cham_diem(sk, kh):
            return _cham_diem_bang_clip(sk, kh, bo_ma_hoa=bo_ma_hoa)

    ma_tran = cham_diem(su_kien, khung)

    if ma_tran.shape != (len(su_kien), len(khung)):
        raise ValueError(
            f"Ma trận điểm có shape {ma_tran.shape}, cần "
            f"({len(su_kien)}, {len(khung)})."
        )

    chi_so = ghep_tang_dan(ma_tran)

    dinh = []
    for i in range(len(su_kien)):
        thu_tu = np.argsort(-ma_tran[i])[:5]
        dinh.append([(khung[j].frame_idx, float(ma_tran[i][j])) for j in thu_tu])

    moc = [
        MocTrake(
            thu_tu=i,
            mo_ta=su_kien[i],
            frame_idx=khung[j].frame_idx,
            pts_time=khung[j].pts_time,
            diem=float(ma_tran[i][j]),
        )
        for i, j in enumerate(chi_so)
    ]

    for i, c in [
        (i + 1, c) for i, c in enumerate(_CUM_GAN_NHAT[: len(su_kien)])
        if any("\u00e0" <= ch <= "\u1ef9" for ch in c)
    ]:
        canh_bao.append(
            f"Sự kiện {i} KHÔNG dịch được, đang đưa nguyên tiếng Việt {c!r} vào "
            "CLIP. Đó là MỐC SÀN, không phải phương án — điểm của mốc này gần "
            "như là nhiễu. Dùng --nguon-mo-rong llm hoặc marian."
        )

    return KetQuaTrake(
        video_id=video_id,
        moc=moc,
        du_day=du_day,
        khoang_cach_khung_giay=khoang_cach,
        so_khung_ung_vien=len(khung),
        do_trai_diem=float(np.mean(ma_tran.max(axis=1) - ma_tran.min(axis=1))),
        su_kien=su_kien,
        cum_tieng_anh=_CUM_GAN_NHAT[: len(su_kien)],
        dinh_moi_su_kien=dinh,
        video_a_quan=a_quan,
        canh_bao=canh_bao,
    )


def _cham_diem_bang_clip(
    su_kien: list[str],
    khung: list[Khung],
    bo_ma_hoa: str = "b32",
) -> np.ndarray:
    """Ma trận cosine giữa mỗi mô tả sự kiện và mỗi khung hình.

    bo_ma_hoa:
        "b32"  CLIP B/32 — cùng mô hình BTC phát vector sẵn. Nhanh (~0,1 giây
               một ảnh trên CPU), đủ để TRẢ LỜI câu hỏi "khung dày có gỡ được
               trần 0,000 không". Dùng cái này trước.
        "lon"  Mô hình của Việc 4 (ViT-L/14). Chính xác hơn nhưng chậm hơn
               khoảng mười lần — 600 khung dày mất cỡ 15 phút trên CPU.

    Chọn "b32" làm mặc định là CỐ Ý: câu hỏi đang cần trả lời là câu hỏi về
    TẦNG DỮ LIỆU, không phải về sức mạnh mô hình. Đổi hai biến cùng lúc thì
    không biết cái nào tạo ra chênh lệch.
    """
    from .query_expand import mo_rong

    if bo_ma_hoa == "lon":
        from .rerank import Reranker

        bo = Reranker()
        ma_hoa_cau, ma_hoa_anh = bo.ma_hoa_cau, bo.ma_hoa_anh
    else:
        from .index.encode.clip_encoder import ClipEncoder
        from PIL import Image

        enc = ClipEncoder()

        def ma_hoa_cau(chu: str) -> np.ndarray:
            v = np.asarray(enc.encode_text(chu), dtype=np.float64).ravel()
            return v / (np.linalg.norm(v) or 1.0)

        def ma_hoa_anh(duong_dan) -> np.ndarray:
            with Image.open(duong_dan) as anh:
                v = np.asarray(
                    enc.encode_image(anh.convert("RGB")), dtype=np.float64
                ).ravel()
            return v / (np.linalg.norm(v) or 1.0)

    _CUM_GAN_NHAT.clear()
    v_su_kien = []
    for s in su_kien:
        try:
            cum = mo_rong(s).cum_chinh
        except Exception:
            cum = s
        _CUM_GAN_NHAT.append(cum)
        v_su_kien.append(ma_hoa_cau(cum))

    v_khung, giu_lai = [], []
    for k in khung:
        if not k.duong_dan.exists():
            continue
        v_khung.append(ma_hoa_anh(k.duong_dan))
        giu_lai.append(k)

    if not v_khung:
        raise FileNotFoundError(
            "Không mở được ảnh nào. Việc 12 cần ảnh gốc raw/keyframes/ hoặc "
            "derived/frames_dense/ của video này trên máy."
        )

    if len(giu_lai) != len(khung):
        # Đổi danh sách khung tại chỗ để phía gọi lấy đúng frame_idx.
        khung[:] = giu_lai

    return np.array(v_su_kien) @ np.array(v_khung).T
