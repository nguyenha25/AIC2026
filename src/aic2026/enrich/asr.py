"""
Việc 15 — Chuyển lời nói thành chữ, KÈM MỐC THỜI GIAN từng câu.

Nhận vào : derived/audio/<video_id>.wav   (Việc 14 tách ra, 16 kHz mono)
Trả ra   : derived/asr/<video_id>.jsonl   (mẫu đã chốt, docs/schema/README.md mục 3)

Mỗi dòng một ĐOẠN LỜI NÓI, không phải một dòng một ảnh:

    {"video_id": "L23_V001", "start": 4.32, "end": 11.05,
     "text": "Hôm nay tại Thành phố Hồ Chí Minh ...",
     "frame_idx_start": 108, "frame_idx_end": 276,
     "lang": "vi", "engine": "faster-whisper-small-int8"}

BA ĐIỀU KHÔNG ĐƯỢC ĐỔI NẾU CHƯA BÁO NHÓM
1. `start`/`end` tính từ ĐẦU TỆP WAV. Việc 14 đã chặn mọi video có luồng tiếng
   bắt đầu trễ (START_TIME_TOLERANCE_SEC = 0.1), nên mốc trong tệp wav bằng
   đúng mốc trong video, bằng đúng cột `pts_time` của bảng đối chiếu.
   Nếu ai đó nới ngưỡng đó ra thì mọi mốc thời gian ở đây lệch theo mà KHÔNG
   có triệu chứng gì ở các bước sau.
2. `fps` lấy từ raw/map-keyframes/<video_id>.csv, KHÔNG lấy từ ffprobe.
   Bốn máy đều có đủ map-keyframes; video thì mỗi máy chỉ giữ một phần.
   Lấy hai nguồn khác nhau là tự tạo ra hai sự thật.
3. `frame_idx_* = round(giây × fps)` — cùng công thức kiểm chứng của Việc 1
   (`frame_idx ≈ pts_time × fps`). Đây là số ĐEM NỘP, không phải số thứ tự ảnh.

CẤU HÌNH đã chốt ở Việc 13: faster-whisper, nén int8, bản small hoặc medium.
KHÔNG dùng large — máy nhóm không có card đồ hoạ rời.

---------------------------------------------------------------------------
LUẬT THỨ TỰ NẠP TRÊN WINDOWS — ĐÃ ĐO, KHÔNG PHẢI ĐOÁN
---------------------------------------------------------------------------
    onnxruntime PHẢI được nạp TRƯỚC pandas.

Đo được bằng scripts/chuan_doan_vad.py trên máy nhóm (Windows, Python 3.11,
onnxruntime 1.28.0, pandas 2.x):

    import numpy;       import onnxruntime   -> ok
    import ctranslate2; import onnxruntime   -> ok
    import faster_whisper; import onnxruntime -> ok
    import pandas;      import onnxruntime   -> HỎNG
    import pyarrow;     import onnxruntime   -> HỎNG

        ImportError: DLL load failed while importing onnxruntime_pybind11_state:
        A dynamic link library (DLL) initialization routine failed.

pandas kéo pyarrow vào, pyarrow nạp bộ DLL của nó, và sau đó onnxruntime nạp
không nổi nữa. Cùng họ với luật đã biết của nhóm: nạp faiss trước torch thì
chết 0xC0000005.

Bộ lọc VAD cần onnxruntime, mà faster-whisper 1.0.2 bắt ImportError rồi thay
bằng câu "Applying the VAD filter requires the onnxruntime package" — NUỐT MẤT
lý do thật. Nên triệu chứng nhìn thấy là "chưa cài gói", trong khi
`python -c "import onnxruntime"` chạy ngon lành. Sai hướng hoàn toàn.

VÌ VẬY câu import onnxruntime ở dưới nằm TRÊN dòng
`from ..frame_map import FrameMap` (dòng đó kéo pandas vào). Nạp xong thì
onnxruntime nằm sẵn trong sys.modules; câu import bên trong faster-whisper
chỉ còn là tra bảng, không nạp DLL lần nữa, không hỏng.

ĐỪNG sắp xếp lại khối import ở đầu tệp này cho "đúng thứ tự bảng chữ cái".
Nó trông như một dòng đặt lộn chỗ, và đó chính là lý do phải viết chú thích
dài thế này. Đổi thứ tự là 137 video chạy lại từ đầu.

Kiểm lại luật này bất cứ lúc nào:  python -m scripts.chuan_doan_vad

Vì sao đáng bật VAD: trên đoạn im lặng dài, whisper hay "ảo giác" — lặp lại
câu cuối hàng chục lần với mốc thời gian bịa. VAD cắt đoạn im lặng đi thì
không còn chỗ cho nó bịa.

Chạy KHÔNG có VAD vẫn ra kết quả dùng được, nhưng phải bù lại bằng ba ngưỡng
trong `_THAM_SO_KHONG_VAD` và bằng bộ lọc lặp `_bo_lap_lai` ở dưới.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# DÒNG NÀY PHẢI NẰM TRÊN CÙNG — TRÊN CẢ `from ..frame_map import FrameMap`,
# vì frame_map kéo pandas vào và pandas làm onnxruntime nạp không được.
# Đọc phần "LUẬT THỨ TỰ NẠP" ở đầu tệp trước khi định dời dòng này.
# ---------------------------------------------------------------------------
try:
    import onnxruntime as _onnxruntime  # noqa: F401
    _VAD_SAN_SANG = True
    _VAD_LY_DO = ""
except Exception as _e:                  # ImportError, DLL hỏng, gói lỗi kiểu khác
    _VAD_SAN_SANG = False
    _VAD_LY_DO = f"{type(_e).__name__}: {_e}"

from ..frame_map import FrameMap         # noqa: E402 — cố ý đặt SAU onnxruntime

# --- Cấu hình chốt ở Việc 13 ------------------------------------------------
MODEL_MAC_DINH = "small"          # hoặc "medium". KHÔNG dùng "large".
KIEU_TINH = "int8"                # bắt buộc — nhanh gấp 4-6 lần trên CPU
NGON_NGU_MAC_DINH = "vi"

VAD_IM_LANG_MS = 500              # im lặng quá 0,5 giây thì cắt đoạn
NOI_TIEP_VAN_BAN_CU = False       # không cho mô hình nhìn lại câu trước

# Khi KHÔNG có VAD: siết ba ngưỡng lọc sẵn có của whisper để bù lại.
#   no_speech_threshold  thấp hơn mặc định (0.6) -> dễ vứt đoạn không phải lời nói
#   compression_ratio_threshold thấp hơn mặc định (2.4) -> câu lặp lại nhiều thì
#       nén được nhiều; vượt ngưỡng là whisper tự giải mã lại đoạn đó
#   log_prob_threshold   giữ mặc định -1.0
_THAM_SO_KHONG_VAD = {
    "no_speech_threshold": 0.45,
    "compression_ratio_threshold": 2.2,
    "log_prob_threshold": -1.0,
}

# Một câu giống hệt câu liền trước lặp quá số lần này thì cắt phần thừa.
# Để 2 chứ không để 1: trong tin tức có chỗ nhắc lại thật (tên chương trình,
# khẩu hiệu), cắt ngay từ lần thứ hai là mất dữ liệu thật.
SO_LAN_LAP_TOI_DA = 2

_KHOANG_TRANG = re.compile(r"\s+", re.UNICODE)

# Nạp model một lần rồi dùng lại — nạp lại mỗi video thì mất thêm vài giây/video.
_kho_model: dict[tuple[str, str, int], object] = {}


class LoiCauHinh(RuntimeError):
    """Lỗi của MÔI TRƯỜNG, không phải của một video cụ thể.

    Phân biệt hai loại lỗi này là điều đáng giá nhất trong tệp: lỗi một video
    thì bỏ qua video đó rồi chạy tiếp; lỗi cấu hình thì DỪNG NGAY, vì 137 video
    sau sẽ hỏng y hệt và người ngồi máy chỉ tốn công đọc lại một câu 137 lần.
    """


@dataclass(frozen=True)
class DoanLoiNoi:
    """Một đoạn lời nói đã có mốc thời gian và vị trí khung hình."""

    video_id: str
    start: float
    end: float
    text: str
    frame_idx_start: int
    frame_idx_end: int
    lang: str
    engine: str

    def to_dict(self) -> dict:
        """Đúng thứ tự khoá của docs/schema/example_asr.jsonl."""
        return {
            "video_id": self.video_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "frame_idx_start": self.frame_idx_start,
            "frame_idx_end": self.frame_idx_end,
            "lang": self.lang,
            "engine": self.engine,
        }


def kiem_onnxruntime() -> tuple[bool, str]:
    """VAD có chạy được trên máy này không, và nếu không thì VÌ SAO.

    Trả về kết quả của lần nạp Ở ĐẦU TỆP, không thử nạp lại. Thử lại ở đây là
    vô nghĩa: lúc hàm này được gọi thì ctranslate2 có thể đã vào rồi, và câu
    trả lời sẽ khác câu trả lời thật sự quan trọng — câu lúc nạp sớm.

    faster-whisper giấu lý do thật đi; hàm này moi ra để người ngồi máy biết
    phải sửa cái gì.
    """
    return _VAD_SAN_SANG, _VAD_LY_DO


def ten_engine(ten_model: str, kieu_tinh: str = KIEU_TINH) -> str:
    """Chuỗi ghi vào khoá `engine` — phải đọc ra được đã chạy bằng gì.

    CỐ Ý không nhét trạng thái VAD vào đây: bốn người phải ra cùng một chuỗi
    `engine` thì Nguyên gộp mới không thấy bốn giá trị khác nhau trong cùng
    một cột. Có bật VAD hay không ghi ở derived/asr/_nhat_ky.jsonl.
    """
    return f"faster-whisper-{ten_model}-{kieu_tinh}"


def nap_model(
    ten_model: str = MODEL_MAC_DINH,
    kieu_tinh: str = KIEU_TINH,
    so_thread: int = 0,
):
    """Nạp faster-whisper (CPU). Gọi lại với cùng tham số thì dùng lại model cũ.

    Import đặt trong hàm để `--dry-run` chạy được trên máy chưa cài faster-whisper.
    """
    khoa = (ten_model, kieu_tinh, so_thread)
    if khoa in _kho_model:
        return _kho_model[khoa]

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise LoiCauHinh(
            "Chưa cài faster-whisper. Chạy: pip install faster-whisper\n"
            f"Chi tiết: {e}"
        ) from e

    model = WhisperModel(
        ten_model,
        device="cpu",
        compute_type=kieu_tinh,
        cpu_threads=so_thread,   # 0 = để thư viện tự chọn theo số nhân
    )
    _kho_model[khoa] = model
    return model


def lay_fps(video_id: str) -> float:
    """fps của một video, đọc từ raw/map-keyframes/<video_id>.csv.

    Cột fps trong một video là hằng số. Nếu không phải hằng số thì bảng đối
    chiếu có vấn đề — báo lỗi ngay chứ không lặng lẽ lấy dòng đầu, vì mọi
    frame_idx_* của video đó sẽ sai theo.
    """
    bang = FrameMap.load(video_id)
    if not bang.rows:
        raise ValueError(f"{video_id}: bảng đối chiếu rỗng, không lấy được fps")

    cac_fps = {row.fps for row in bang.rows}
    if len(cac_fps) > 1:
        raise ValueError(
            f"{video_id}: fps không đồng nhất trong map-keyframes ({sorted(cac_fps)}). "
            "Cần xem lại bảng đối chiếu trước khi chạy ASR video này."
        )
    return cac_fps.pop()


def _don_chu(text: str) -> str:
    """Bỏ khoảng trắng thừa. KHÔNG bỏ dấu tiếng Việt — kho tra cứu cần chữ có dấu."""
    return _KHOANG_TRANG.sub(" ", text).strip()


def _bo_lap_lai(
    cac_doan: list[DoanLoiNoi],
    toi_da: int = SO_LAN_LAP_TOI_DA,
) -> tuple[list[DoanLoiNoi], int]:
    """Cắt chuỗi câu giống hệt nhau lặp LIÊN TIẾP — dấu hiệu whisper 'ảo giác'.

    Chỉ cắt khi câu giống hệt câu LIỀN TRƯỚC. Câu giống nhau nhưng nằm xa nhau
    trong video là chuyện bình thường (khẩu hiệu, tên chương trình) và không
    bị đụng tới.

    Trả về (danh sách đã lọc, số dòng bị cắt).
    """
    giu: list[DoanLoiNoi] = []
    truoc: str | None = None
    dem = 0
    bi_cat = 0

    for doan in cac_doan:
        if doan.text == truoc:
            dem += 1
        else:
            truoc = doan.text
            dem = 1

        if dem <= toi_da:
            giu.append(doan)
        else:
            bi_cat += 1

    return giu, bi_cat


def chuyen_mot_video(
    duong_dan_wav: Path,
    video_id: str,
    fps: float,
    model,
    ten_model: str = MODEL_MAC_DINH,
    ngon_ngu: str | None = NGON_NGU_MAC_DINH,
    beam_size: int = 5,
    dung_vad: bool = True,
) -> tuple[list[DoanLoiNoi], dict]:
    """Chạy ASR trên một tệp .wav.

    ngon_ngu = None  -> để mô hình tự nhận ngôn ngữ (chậm hơn một chút).
    ngon_ngu = "vi"  -> ép tiếng Việt. Mặc định, vì kho là tin tức tiếng Việt;
                        để mô hình tự đoán trên đoạn nhạc/không lời thì nó hay
                        nhận nhầm sang tiếng khác rồi dịch bậy.
    dung_vad = False -> chạy được trên máy không nạp được onnxruntime; đổi lại
                        siết ngưỡng lọc và cắt câu lặp ở bước sau.

    Trả về (danh sách đoạn, thông tin chạy).
    """
    tham_so: dict = dict(
        language=ngon_ngu,
        beam_size=beam_size,
        condition_on_previous_text=NOI_TIEP_VAN_BAN_CU,
        word_timestamps=False,
    )
    if dung_vad:
        tham_so["vad_filter"] = True
        tham_so["vad_parameters"] = {"min_silence_duration_ms": VAD_IM_LANG_MS}
    else:
        tham_so.update(_THAM_SO_KHONG_VAD)

    try:
        segments, info = model.transcribe(str(duong_dan_wav), **tham_so)
        # `segments` là generator: mô hình chạy THẬT ở dòng list() dưới đây,
        # nên lỗi VAD cũng chỉ nổ ra ở đó chứ không nổ ở dòng transcribe.
        doan_tho = list(segments)
    except RuntimeError as e:
        loi = str(e).lower()
        if "onnxruntime" in loi or "vad" in loi:
            raise LoiCauHinh(str(e)) from e
        raise

    lang = ngon_ngu or getattr(info, "language", "vi")
    engine = ten_engine(ten_model)

    ket_qua: list[DoanLoiNoi] = []
    for seg in doan_tho:
        chu = _don_chu(seg.text or "")
        if not chu:
            continue

        bat_dau = max(0.0, float(seg.start))
        ket_thuc = max(bat_dau, float(seg.end))

        ket_qua.append(
            DoanLoiNoi(
                video_id=video_id,
                start=bat_dau,
                end=ket_thuc,
                text=chu,
                frame_idx_start=round(bat_dau * fps),
                frame_idx_end=round(ket_thuc * fps),
                lang=lang,
                engine=engine,
            )
        )

    ket_qua, so_cat = _bo_lap_lai(ket_qua)

    thong_tin = {
        "lang": lang,
        "lang_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
        "do_dai_audio_giay": round(float(getattr(info, "duration", 0.0) or 0.0), 2),
        "engine": engine,
        "fps": fps,
        "vad": dung_vad,
        "so_dong_lap_da_cat": so_cat,
    }
    return ket_qua, thong_tin


def dong_danh_dau_khong_loi_noi(video_id: str, fps: float, lang: str, engine: str) -> dict:
    """Video có tiếng nhưng không có lời nói nào (nhạc nền, tiếng ồn).

    Vẫn ghi MỘT dòng `text` rỗng, cùng lý do như bên OCR: tệp rỗng thì người
    đọc sau không phân biệt được "chưa chạy" với "chạy rồi mà không có gì".
    Bộ nạp FTS bỏ qua dòng `text` rỗng nên nó không vào kho tra cứu.
    """
    return DoanLoiNoi(
        video_id=video_id,
        start=0.0,
        end=0.0,
        text="",
        frame_idx_start=0,
        frame_idx_end=0,
        lang=lang,
        engine=engine,
    ).to_dict()