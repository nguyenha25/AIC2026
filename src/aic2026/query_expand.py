"""
Việc 5 — MỞ RỘNG TRUY VẤN: TIẾNG VIỆT -> CỤM TIẾNG ANH NGẮN CHO NHÁNH CLIP.

VẤN ĐỀ
------
CLIP B/32 huấn luyện trên chú thích ảnh TIẾNG ANH, phần lớn dài 5-15 chữ. Đề
thi lại viết tiếng Việt, dài dòng, nhiều chữ đưa đẩy ("Trong đoạn video, hãy
tìm khoảnh khắc mà..."). Nhét nguyên câu vào là bắt mô hình xử lý thứ nó
chưa từng thấy.

RANH GIỚI PHẢI GIỮ — ĐỌC TRƯỚC KHI CẮM VÀO RRF
----------------------------------------------
Bản dịch tiếng Anh CHỈ dùng cho nhánh CLIP. Nhánh OCR và ASR đọc chữ tiếng
Việt trên hình và lời nói tiếng Việt — đưa câu tiếng Anh vào hai nhánh đó là
bảo đảm 0 kết quả. Đây là mâu thuẫn đã ghi trong Task 10 và tệp này giải
quyết bằng cách tách hẳn: `tim_ung_vien_clip_mo_rong()` chỉ đổi câu cho nhánh
CLIP, hai nhánh chữ nhận NGUYÊN câu tiếng Việt.

BA NGUỒN DỊCH, CHỌN TRONG settings.yaml
---------------------------------------
    tu_dien   Tra bảng + cắt chữ thừa. Chạy offline, không tải gì, TẤT ĐỊNH.
              Đây là mốc sàn để so, và là phương án dự phòng khi mạng hỏng.
    marian    Mô hình dịch máy đã có sẵn trong clip_encoder (nhánh "dich").
    llm       Gọi mô hình ngôn ngữ. Dịch tốt nhất, nhưng CẦN MẠNG và không
              tất định giữa các phiên bản mô hình -> BẮT BUỘC dùng bộ nhớ đệm.

BỘ NHỚ ĐỆM LÀ BẮT BUỘC, KHÔNG PHẢI ĐỂ CHẠY NHANH
------------------------------------------------
derived/mo_rong_truy_van.json lưu mọi bản dịch. Hai lý do:
  1. Đo lại điểm sau một tuần phải ra đúng con số cũ. Không đệm thì mỗi lần
     chạy là một bản dịch khác và không so được gì với gì.
  2. Người ngồi máy MỞ TỆP RA SỬA TAY được. Dịch máy sai một từ khoá là mất
     câu đó; sửa trong tệp đệm thì lần chạy sau dùng bản đã sửa.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Chữ đưa đẩy trong đề — bỏ đi thì cụm còn lại mới là nội dung thật.
# VIẾT CÓ DẤU. Xem chuan_hoa() ở objects_index.py: bỏ dấu làm "tìm" thành
# "tim" và trùng với "tím" (purple), "trong" trùng "trống" (Drum).
CHU_THUA = [
    "trong đoạn video", "trong video", "trong đoạn clip", "trong clip",
    "hãy tìm", "tìm kiếm", "tìm ra", "hãy cho biết", "cho biết",
    "khoảnh khắc mà", "khoảnh khắc", "cảnh quay", "cảnh nào", "hình ảnh nào",
    "hình ảnh", "đoạn phim", "người xem thấy", "hãy xác định", "xác định",
]

# Từ nội dung hay gặp mà bảng nhãn vật thể không có: màu, hành động, bối cảnh.
# CÓ DẤU, và so theo ranh giới từ.
TU_DIEN_BO_SUNG: dict[str, str] = {
    # màu
    "đỏ": "red", "xanh lá": "green", "xanh dương": "blue", "xanh": "blue",
    "vàng": "yellow", "trắng": "white", "đen": "black", "cam": "orange",
    "tím": "purple", "hồng": "pink", "nâu": "brown", "xám": "grey",
    # số đếm
    "một": "one", "hai": "two", "ba": "three", "bốn": "four", "năm": "five",
    # hành động
    "chạy": "running", "đi bộ": "walking", "nhảy": "jumping",
    "ngồi": "sitting", "đứng": "standing", "nằm": "lying down",
    "cầm": "holding", "nấu ăn": "cooking", "ăn": "eating", "uống": "drinking",
    "nói": "speaking", "hát": "singing", "chơi đàn": "playing instrument",
    "vỗ tay": "clapping", "bắt tay": "shaking hands", "cười": "smiling",
    "lái xe": "driving", "bơi": "swimming", "leo": "climbing",
    "phát biểu": "giving a speech", "trao giải": "awarding a prize",
    "cắt băng": "ribbon cutting", "đua thuyền": "boat racing",
    # bối cảnh
    "ngoài trời": "outdoors", "trong nhà": "indoors", "ban đêm": "at night",
    "ban ngày": "daytime", "trên sân khấu": "on a stage",
    "hội trường": "conference hall", "lớp học": "classroom",
    "bệnh viện": "hospital", "chợ": "market", "biển": "beach",
    "núi": "mountain", "rừng": "forest", "sông": "river", "ruộng": "rice field",
    "đường phố": "street", "công viên": "park", "sân vận động": "stadium",
    "nhà hàng": "restaurant", "bếp": "kitchen", "văn phòng": "office",
    "lễ hội": "festival", "đám cưới": "wedding", "họp báo": "press conference",
    "biểu diễn": "performance", "diễu hành": "parade", "mưa lũ": "flood",
    # góc quay
    "cận cảnh": "close-up", "toàn cảnh": "wide shot", "từ trên cao": "aerial view",

    # --- ĐỘNG TỪ: thứ phân biệt các mốc của một chuỗi TRAKE ---
    # Bảng chỉ có danh từ thì ba mốc "nhấc bánh / đặt bánh / đậy nắp" đều rút
    # về cùng một cụm, ma trận điểm có ba hàng giống hệt nhau, và phần ghép
    # chỉ lấy được đỉnh cao nhất với vài khung kề bên. Đã đo thật ở câu 15.
    "nhấc": "lifting up", "nhấc lên": "lifting up",
    "đặt": "placing down", "đặt vào": "placing into", "bỏ vào": "putting into",
    "lấy ra": "taking out", "cầm lên": "picking up", "đưa cho": "handing over",
    "đậy": "covering with a lid", "đậy nắp": "putting on a lid",
    "mở nắp": "opening a lid", "mở": "opening", "đóng": "closing",
    "rót": "pouring", "khuấy": "stirring", "cắt": "cutting", "thái": "slicing",
    "xếp": "arranging", "gói": "wrapping", "buộc": "tying",
    "bước vào": "entering", "bước ra": "exiting", "quay lại": "turning around",
    "cúi xuống": "bending down", "đứng dậy": "standing up", "ngồi xuống": "sitting down",
    "chỉ tay": "pointing", "vẫy tay": "waving", "ôm": "hugging",
    "ném": "throwing", "bắt": "catching", "đá": "kicking", "đẩy": "pushing",
    "kéo": "pulling", "nâng": "raising", "hạ": "lowering",
    "bứt lên": "sprinting ahead", "vượt lên": "overtaking",
    "về đích": "crossing the finish line", "xuất phát": "starting line",
    "ngang nhau": "side by side", "dẫn đầu": "leading in front",
    "giơ tay": "raising a hand", "trao": "handing over",

    # --- danh từ bếp núc và thể thao hay gặp ---
    "nồi": "cooking pot", "chảo": "frying pan", "nắp": "lid", "rổ": "basket",
    "bếp": "stove", "bếp củi": "wood fire stove", "lửa": "fire", "khói": "smoke",
    "luộc": "boiling", "nướng": "grilling", "chiên": "frying",
    "tay đua": "racing cyclist", "đua xe đạp": "bicycle race",
    "vận động viên": "athlete", "trọng tài": "referee", "khán giả": "crowd",
    "huy chương": "medal", "cúp": "trophy", "đường đua": "race track",
}

TEN_TEP_DEM = "mo_rong_truy_van.json"

# Tăng số này khi phát hiện đệm cũ chứa dữ liệu sai — mục cũ sẽ bị bỏ qua
# thay vì phải dặn từng người đi xoá tệp.
#   v1 -> v2: bản v1 đệm cả kết quả LÙI VỀ tu_dien dưới khoá "llm"/"marian",
#             nên có khoá API rồi vẫn đọc lại bản hỏng.
PHIEN_BAN_DEM = 2


def con_tieng_viet(s: str) -> bool:
    """Chuỗi còn chữ cái tiếng Việt có dấu hay không.

    Dùng để soát BẢN DỊCH: mô hình dịch trả lại nguyên câu tiếng Việt là
    chuyện có thật (VietAI/envit5 thỉnh thoảng chép lại đầu vào). Chuỗi đó
    KHÔNG rỗng nên mọi phép kiểm "có trả về gì không" đều cho qua, rồi cả
    mạch chạy ở mốc sàn mà không ai biết.
    """
    return any("\u00c0" <= ch <= "\u1ef9" for ch in (s or ""))


def chuan_hoa(s: str) -> str:
    """Hạ chữ thường bằng Python, GIỮ DẤU. Xem objects_index.chuan_hoa()."""
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


@dataclass
class KetQuaMoRong:
    """Kết quả mở rộng MỘT câu hỏi."""

    cau_goc: str
    cum_tieng_anh: list[str]
    nguon: str                      # tu_dien / marian / llm / dem / nguyen_ban
    ghi_chu: list[str] = field(default_factory=list)

    @property
    def cum_chinh(self) -> str:
        return self.cum_tieng_anh[0] if self.cum_tieng_anh else self.cau_goc


# ---------------------------------------------------------------------------
# Bộ nhớ đệm trên đĩa
# ---------------------------------------------------------------------------

class BoNhoDem:
    """{nguồn: {câu tiếng Việt: [cụm tiếng Anh]}} lưu ở derived/."""

    def __init__(self, duong_dan: Path | None = None):
        if duong_dan is None:
            from .paths import DERIVED_DIR

            duong_dan = DERIVED_DIR / TEN_TEP_DEM
        self.duong_dan = Path(duong_dan)
        self._du_lieu: dict[str, dict[str, list[str]]] = {}
        self._nap()

    def _nap(self) -> None:
        if not self.duong_dan.exists():
            self._du_lieu = {"_phien_ban": PHIEN_BAN_DEM}
            return

        try:
            with self.duong_dan.open("r", encoding="utf-8") as f:
                du_lieu = json.load(f)
        except (OSError, json.JSONDecodeError):
            du_lieu = {}

        # Kiểm phiên bản LÚC NẠP, không phải lúc tra: bản đệm mới dựng trong
        # RAM chưa kịp ghi xuống đĩa vẫn phải dùng được ngay.
        if du_lieu.get("_phien_ban") != PHIEN_BAN_DEM:
            du_lieu = {}

        du_lieu["_phien_ban"] = PHIEN_BAN_DEM
        self._du_lieu = du_lieu

    def lay(self, nguon: str, cau: str) -> list[str] | None:
        return self._du_lieu.get(nguon, {}).get(cau.strip())

    def dat(self, nguon: str, cau: str, cum: list[str]) -> None:
        self._du_lieu.setdefault(nguon, {})[cau.strip()] = list(cum)

    def ghi(self) -> None:
        self.duong_dan.parent.mkdir(parents=True, exist_ok=True)
        tam = self.duong_dan.with_suffix(".partial")
        with tam.open("w", encoding="utf-8") as f:
            json.dump(self._du_lieu, f, ensure_ascii=False, indent=2, sort_keys=True)
        tam.replace(self.duong_dan)


# ---------------------------------------------------------------------------
# Nguồn 1 — tra bảng, offline, tất định
# ---------------------------------------------------------------------------

def cat_chu_thua(cau: str) -> str:
    """Bỏ các cụm đưa đẩy, giữ phần nội dung. Giữ nguyên dấu tiếng Việt."""
    van_ban = chuan_hoa(cau)
    for cum in sorted(CHU_THUA, key=len, reverse=True):
        van_ban = van_ban.replace(cum, " ")
    return re.sub(r"\s+", " ", van_ban).strip(" ,.:;?!")


def dich_bang_tu_dien(cau: str, bang_nhan=None) -> KetQuaMoRong:
    """Rút các từ khoá nhận ra được thành một cụm tiếng Anh ngắn.

    KHÔNG phải dịch thật — chỉ nhặt danh từ, màu, hành động, bối cảnh. Chấp
    nhận mất ngữ pháp: CLIP quan tâm từ khoá hơn cú pháp.
    """
    if bang_nhan is None:
        from .index.objects_index import BangNhan

        bang_nhan = BangNhan.nap()

    goc_da_cat = cat_chu_thua(cau)
    tu_khoa: list[str] = []

    # a) danh từ vật thể — dùng lại bảng của Việc 3, không dựng bảng thứ hai
    for nhan in bang_nhan.tim_nhan(cau):
        t = nhan.lower()
        if t not in tu_khoa:
            tu_khoa.append(t)

    # b) màu / hành động / bối cảnh — so theo RANH GIỚI TỪ
    for tu_viet in sorted(TU_DIEN_BO_SUNG, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(tu_viet)}(?!\w)", goc_da_cat):
            tu_anh = TU_DIEN_BO_SUNG[tu_viet]
            if tu_anh not in tu_khoa:
                tu_khoa.append(tu_anh)

    ghi_chu: list[str] = []
    if not tu_khoa:
        ghi_chu.append(
            "Không nhận ra từ khoá nào. Nhánh tu_dien trả nguyên câu tiếng Việt "
            "— với CLIP đây là mốc SÀN, không phải phương án."
        )
        return KetQuaMoRong(cau, [cau.strip()], "nguyen_ban", ghi_chu)

    cum = " ".join(tu_khoa)
    bien_the = [cum]
    # Biến thể 2: thêm khung câu chú thích ảnh, gần với dữ liệu huấn luyện CLIP.
    bien_the.append(f"a photo of {cum}")

    return KetQuaMoRong(cau, bien_the, "tu_dien", ghi_chu)


# ---------------------------------------------------------------------------
# Nguồn 2 — mô hình dịch máy đã có sẵn
# ---------------------------------------------------------------------------

# Bộ dịch dùng chung. Mô hình VietAI/envit5-translation nặng ~900 MB và mất
# vài chục giây để nạp — dựng một lần rồi giữ lại cho cả tiến trình.
_BO_DICH = None


def _lay_bo_dich():
    """ClipEncoder với nhánh chữ ép cứng là 'dich'.

    Truyền thẳng nhanh_chu="dich" thay vì bắt người dùng sửa settings.yaml.
    Bản đầu dựng ClipEncoder() suông, nên nó đọc settings.yaml thấy 'da_ngu'
    và KHÔNG dịch gì cả — hàm này trả về rỗng rồi lùi nhánh, dù mô hình dịch
    đã có sẵn trên máy.

    Cũng KHÔNG đụng vào biến môi trường AIC_NHANH_CHU: làm vậy sẽ đổi luôn
    nhánh chữ của phần tra cứu chính trong cùng tiến trình.
    """
    global _BO_DICH

    if _BO_DICH is None:
        from .index.encode.clip_encoder import ClipEncoder

        _BO_DICH = ClipEncoder(nhanh_chu="dich")

    return _BO_DICH


def dich_bang_marian(cau: str) -> KetQuaMoRong:
    """Dịch Việt -> Anh bằng mô hình chạy LOCAL, không cần khoá API.

    Lần chạy đầu tải ~900 MB. Sau đó chạy offline hoàn toàn.
    Bản dịch được đệm vào derived/mo_rong_truy_van.json và SỬA TAY ĐƯỢC —
    dịch sai một từ khoá thì mở tệp ra sửa, lần chạy sau dùng bản đã sửa.
    """
    try:
        encoder = _lay_bo_dich()
    except Exception as loi:
        return KetQuaMoRong(
            cau, [cau.strip()], "nguyen_ban",
            [f"Không dựng được bộ dịch: {loi}"],
        )

    # Câu đưa vào bộ dịch phải GIỮ hoa thường và dấu câu. Đo thật trên envit5:
    #   'vi: Một người đàn ông cầm ô.' -> 'en: A man with an umbrella.'
    #   'một người đàn ông cầm ô'      -> 'en: A man with a umbrella'
    # Bản có dấu câu dịch chuẩn hơn. cat_chu_thua() hạ chữ thường và cắt dấu
    # câu, nên chỉ dùng nó để BỎ CỤM ĐƯA ĐẨY rồi khôi phục lại dáng câu.
    dau_vao = cat_chu_thua(cau).strip()
    if dau_vao:
        dau_vao = dau_vao[0].upper() + dau_vao[1:]
        if dau_vao[-1] not in ".!?":
            dau_vao += "."
    else:
        dau_vao = cau.strip()

    try:
        encoder.encode_text(dau_vao)                 # kích hoạt nhánh dịch
        # ban_dich_gan_nhat là @property, KHÔNG phải hàm. Bản đầu gọi nó như
        # hàm -> TypeError -> bị chính khối except này nuốt và báo "dịch hỏng",
        # trong khi mô hình dịch chạy hoàn hảo. Lỗi nằm ở một cặp ngoặc.
        ban_dich = (encoder.ban_dich_gan_nhat or "").strip()
    except Exception as loi:
        return KetQuaMoRong(
            cau, [cau.strip()], "nguyen_ban",
            [f"Dịch hỏng: {type(loi).__name__}: {loi}"],
        )

    # envit5 là mô hình HAI CHIỀU và thỉnh thoảng trả lại y nguyên đầu vào.
    # Bỏ mọi tiền tố tác vụ trước khi soát.
    for tien_to in ("en:", "vi:", "en :", "vi :"):
        if ban_dich.lower().startswith(tien_to):
            ban_dich = ban_dich[len(tien_to):].strip()

    if not ban_dich:
        return KetQuaMoRong(
            cau, [cau.strip()], "nguyen_ban",
            ["Bộ dịch trả về chuỗi rỗng."],
        )

    if con_tieng_viet(ban_dich):
        return KetQuaMoRong(
            cau, [cau.strip()], "nguyen_ban",
            [
                f"Bộ dịch trả lại tiếng Việt: {ban_dich!r}. Bản dịch KHÔNG rỗng "
                "nên phép kiểm 'có trả về gì không' cho qua, nhưng nhánh CLIP "
                "vẫn nhận tiếng Việt — đó là mốc sàn."
            ],
        )

    return KetQuaMoRong(cau, [ban_dich, f"a photo of {ban_dich}"], "marian")


def cat_chu_thua_giu_dau(cau: str) -> str:
    """Giữ lại vì tên gọi cũ. Nay cat_chu_thua() đã tự giữ dấu."""
    return cat_chu_thua(cau)


# ---------------------------------------------------------------------------
# Nguồn 3 — mô hình ngôn ngữ
# ---------------------------------------------------------------------------

LOI_NHAC_LLM = (
    "Bạn đang giúp một hệ thống tìm ảnh bằng CLIP. Hãy đổi câu mô tả tiếng "
    "Việt dưới đây thành 3 cụm mô tả ảnh TIẾNG ANH ngắn (mỗi cụm dưới 15 từ), "
    "cụ thể, chỉ tả THỨ NHÌN THẤY ĐƯỢC trong một khung hình. GIỮ LẠI ĐỘNG TỪ "
    "chỉ hành động — đó là thứ phân biệt các khoảnh khắc với nhau. Bỏ mọi chữ "
    "đưa đẩy và mọi suy đoán. Trả về JSON dạng {\"cum\": [\"...\", \"...\", "
    "\"...\"]}, không thêm chữ nào khác.\n\nCâu tiếng Việt: "
)


def dich_bang_llm(cau: str, goi_llm: Callable[[str], str] | None = None) -> KetQuaMoRong:
    """Gọi mô hình ngôn ngữ. `goi_llm` cho phép cắm bất kỳ nhà cung cấp nào.

    Mặc định dùng Anthropic khi có biến môi trường ANTHROPIC_API_KEY. Không có
    khoá thì tự lùi về nhánh tu_dien thay vì ném lỗi giữa lúc đang chấm điểm.
    """
    if goi_llm is None:
        goi_llm = _goi_anthropic

    try:
        tra_loi = goi_llm(LOI_NHAC_LLM + cau.strip())
    except Exception as loi:
        kq = dich_bang_tu_dien(cau)
        kq.ghi_chu.append(f"LLM lỗi ({loi}) — đã lùi về nhánh tu_dien.")
        return kq

    cum = [c for c in _doc_mang_json(tra_loi) if not con_tieng_viet(c)]
    if not cum:
        kq = dich_bang_tu_dien(cau)
        kq.ghi_chu.append("LLM trả về thứ không đọc được — đã lùi về tu_dien.")
        return kq

    return KetQuaMoRong(cau, cum, "llm")


def _doc_mang_json(van_ban: str) -> list[str]:
    van_ban = re.sub(r"^```(?:json)?|```$", "", van_ban.strip(), flags=re.MULTILINE)
    try:
        du_lieu = json.loads(van_ban.strip())
    except json.JSONDecodeError:
        khop = re.search(r"\[.*\]", van_ban, re.DOTALL)
        if not khop:
            return []
        try:
            du_lieu = json.loads(khop.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(du_lieu, dict):
        du_lieu = du_lieu.get("cum", [])
    if isinstance(du_lieu, list):
        return [str(x).strip() for x in du_lieu if str(x).strip()]
    return []


# Tên mô hình mặc định. Đổi bằng $env:AIC_MO_HINH_LLM mà không phải sửa mã.
# Dịch một cụm ngắn là việc nhẹ, không cần mô hình lớn.
MO_HINH_LLM_MAC_DINH = "claude-haiku-4-5-20251001"

# Ràng buộc đầu ra bằng JSON schema thay vì nhờ mô hình "trả về đúng một mảng
# JSON" rồi tự bóc bằng regex. Thư viện anthropic từ bản 1.0 hỗ trợ sẵn.
_SCHEMA_CUM = {
    "type": "object",
    "properties": {
        "cum": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        }
    },
    "required": ["cum"],
}


def _giai_thich_loi_api(loi: Exception, ten_mo_hinh: str) -> str:
    """Đổi lỗi API thành một câu nói rõ PHẢI LÀM GÌ.

    Bản đầu in cùng một lời khuyên "đổi tên mô hình" cho mọi loại lỗi, kể cả
    lỗi xác thực — dẫn người đọc đi sai hướng ngay khi họ đang bí.
    """
    van_ban = str(loi)
    ma = getattr(loi, "status_code", None)

    if ma == 401 or "authentication" in van_ban.lower() or "invalid" in van_ban.lower() and "key" in van_ban.lower():
        return (
            "KHOÁ API KHÔNG HỢP LỆ (lỗi 401).\n\n"
            "Kiểm tra xem .env có đang chứa đúng chuỗi chỗ điền không:\n"
            "    type .env\n"
            "Thấy 'sk-ant-...' với ba dấu chấm nghĩa là chưa thay bằng khoá thật.\n"
            "Lấy khoá tại console.anthropic.com -> API keys, rồi:\n"
            '    Set-Content .env "DATA_ROOT=D:/aic-data"\n'
            '    Add-Content .env "ANTHROPIC_API_KEY=<dán khoá thật vào đây>"'
        )

    if ma in (404, 400) or "model" in van_ban.lower():
        return (
            f"Mô hình {ten_mo_hinh!r} không dùng được: {loi}\n"
            'Đổi bằng: $env:AIC_MO_HINH_LLM = "claude-sonnet-5"'
        )

    if ma == 429 or "rate" in van_ban.lower():
        return (
            f"Bị giới hạn tốc độ gọi: {loi}\n"
            "Chờ một chút rồi chạy lại. Bản dịch đã xong được đệm lại nên "
            "không phải dịch lại từ đầu."
        )

    return f"Gọi API hỏng với mô hình {ten_mo_hinh!r}: {loi}"


def _goi_anthropic(loi_nhac: str) -> str:
    """Gọi API. Trả về chuỗi JSON để _doc_mang_json() bóc.

    KHÔNG truyền temperature: thư viện anthropic 1.0 đã BỎ HẲN tham số đó, và
    truyền vào là TypeError. Tính lặp lại của phép đo do BỘ NHỚ ĐỆM bảo đảm —
    đó vốn là lý do chính bộ đệm tồn tại, không phải để chạy nhanh.
    """
    khoa = os.getenv("ANTHROPIC_API_KEY")
    if not khoa:
        raise RuntimeError(
            "Chưa có ANTHROPIC_API_KEY trong .env. Thêm bằng PowerShell:\n"
            '    Add-Content .env "ANTHROPIC_API_KEY=sk-ant-..."\n'
            "Gõ dấu thăng ở đầu dòng là PowerShell coi cả dòng là chú thích và "
            "không làm gì cả."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=khoa)
    ten_mo_hinh = os.getenv("AIC_MO_HINH_LLM", MO_HINH_LLM_MAC_DINH)

    tham_so = {
        "model": ten_mo_hinh,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": loi_nhac}],
    }

    # Bản thư viện cũ chưa có output_config -> bỏ qua, dùng đường bóc JSON thường.
    try:
        import inspect

        if "output_config" in inspect.signature(client.messages.create).parameters:
            tham_so["output_config"] = {
                "format": {"type": "json_schema", "schema": _SCHEMA_CUM}
            }
    except Exception:
        pass

    try:
        tra_loi = client.messages.create(**tham_so)
    except Exception as loi:
        raise RuntimeError(_giai_thich_loi_api(loi, ten_mo_hinh)) from loi

    return "".join(
        k.text for k in tra_loi.content if getattr(k, "type", "") == "text"
    )


# ---------------------------------------------------------------------------
# Cửa chính
# ---------------------------------------------------------------------------

_NGUON = {
    "tu_dien": dich_bang_tu_dien,
    "marian": dich_bang_marian,
    "llm": dich_bang_llm,
}


def nguon_mac_dinh() -> str:
    """Nguồn dịch đang dùng.

    THỨ TỰ ƯU TIÊN — biến môi trường THẮNG settings.yaml:
        1. $env:AIC_NGUON_MO_RONG   (cờ --nguon-mo-rong đặt biến này)
        2. mo_rong_truy_van.nguon trong settings.yaml
        3. "tu_dien"

    Bản đầu đảo ngược hai mục đầu, nên cờ --nguon-mo-rong llm bị settings.yaml
    đè và im lặng chạy nhánh tu_dien. Hai lần chạy ra số giống hệt nhau mà
    người chạy tưởng đã thử xong nhánh LLM — đúng loại phép đo giả.

    Cùng quy ước với $env:AIC_NHANH_CHU đã có sẵn trong settings.yaml: cờ dòng
    lệnh là thứ người gõ NGAY LÚC ĐÓ, nên nó phải thắng tệp cấu hình.
    """
    tu_moi_truong = os.getenv("AIC_NGUON_MO_RONG")
    if tu_moi_truong:
        return tu_moi_truong

    try:
        from .rank.config import load_settings

        muc = load_settings().get("mo_rong_truy_van")
        if isinstance(muc, dict) and muc.get("nguon"):
            return str(muc["nguon"])
    except Exception:
        pass
    return "tu_dien"


class LuiNhanhKhongMongMuon(RuntimeError):
    """Đã chỉ định một nguồn dịch, nhưng nó hỏng và mạch lùi về nguồn khác."""


def mo_rong(
    cau_hoi: str,
    nguon: str | None = None,
    dung_dem: bool = True,
    dem: BoNhoDem | None = None,
    bat_buoc: bool | None = None,
) -> KetQuaMoRong:
    """Câu tiếng Việt -> các cụm tiếng Anh cho nhánh CLIP.

    bat_buoc=True: nguồn hỏng thì NÉM LỖI thay vì lùi về tu_dien.
    None (mặc định) nghĩa là tự bật khi người chạy có gõ cờ --nguon-mo-rong
    (tức có biến $env:AIC_NGUON_MO_RONG).

    VÌ SAO PHẢI CÓ CÁI NÀY
    ----------------------
    Bản đầu lùi âm thầm: thiếu ANTHROPIC_API_KEY thì dich_bang_llm() bắt mọi
    lỗi, gọi dich_bang_tu_dien(), ghi lý do vào ghi_chu — mà không ai in
    ghi_chu ra. Người chạy thấy dòng "nguồn mở rộng: llm", thấy kết quả, và
    tưởng đã thử xong nhánh LLM. Thực ra vẫn là tu_dien.

    Lùi âm thầm CHỈ chấp nhận được khi không ai chỉ định gì. Đã gõ cờ thì
    người chạy đang ĐO một nhánh cụ thể, và một phép đo nhầm nhánh còn tệ hơn
    không đo.
    """
    cau_hoi = (cau_hoi or "").strip()
    if not cau_hoi:
        raise ValueError("Câu truy vấn rỗng.")

    if bat_buoc is None:
        bat_buoc = bool(os.getenv("AIC_NGUON_MO_RONG"))

    nguon = nguon or nguon_mac_dinh()
    if nguon not in _NGUON:
        raise ValueError(f"Nguồn mở rộng lạ: {nguon}. Chọn: {sorted(_NGUON)}")

    dem = dem if dem is not None else (BoNhoDem() if dung_dem else None)

    if dem is not None:
        da_co = dem.lay(nguon, cau_hoi)
        if da_co:
            return KetQuaMoRong(cau_hoi, list(da_co), "dem")

    ket_qua = _NGUON[nguon](cau_hoi)

    # CHỈ tu_dien được phép trả "nguyen_ban": nó là bảng tra, không nhận ra từ
    # nào thì đành chịu, đó là hành vi bình thường.
    #
    # Với marian và llm thì "nguyen_ban" nghĩa là HỎNG — mô hình dịch có mà
    # không dịch được. Miễn trừ nhầm cho cả hai nguồn này là lý do phép kiểm
    # bat_buoc chưa từng nổ dù bộ dịch trả lại nguyên tiếng Việt.
    duoc_mien = {nguon} | ({"nguyen_ban"} if nguon == "tu_dien" else set())
    da_lui = ket_qua.nguon not in duoc_mien

    if da_lui and bat_buoc:
        raise LuiNhanhKhongMongMuon(
            f"Đã chỉ định nguồn {nguon!r} nhưng nó hỏng và mạch lùi về "
            f"{ket_qua.nguon!r}. Lý do: "
            + ("; ".join(ket_qua.ghi_chu) or "không rõ")
        )

    # KHÔNG đệm kết quả lùi dưới tên nguồn đã yêu cầu: làm vậy là đầu độc đệm,
    # lần sau có khoá API thật vẫn đọc lại bản hỏng.
    if dem is not None and ket_qua.nguon == nguon:
        dem.dat(nguon, cau_hoi, ket_qua.cum_tieng_anh)
        dem.ghi()

    return ket_qua


# ---------------------------------------------------------------------------
# Cắm vào mạch tìm kiếm
# ---------------------------------------------------------------------------

def tim_ung_vien_clip_mo_rong(
    nguon: str | None = None,
    k_rrf: int = 60,
    toi_da_bien_the: int = 2,
) -> Callable[[str, int], list]:
    """Nhà máy sinh hàm (câu chữ, số ứng viên) -> list[Hit] cho nhánh CLIP.

    Mỗi biến thể tiếng Anh chạy một lượt tra kho rồi gộp bằng RRF. Gộp theo
    HẠNG chứ không cộng cosine: cosine của hai câu khác nhau không cùng thang.

    CHỈ dùng cho nhánh CLIP. Nhánh OCR/ASR nhận nguyên câu tiếng Việt.
    """
    def _tim(cau_hoi: str, so_ung_vien: int):
        from .rank.fuse import reciprocal_rank_fusion
        from .rank.hop_nhat import hit_sang_dict
        from .rank.search import tim_ung_vien_clip

        ket_qua = mo_rong(cau_hoi, nguon=nguon)
        bien_the = ket_qua.cum_tieng_anh[:toi_da_bien_the] or [cau_hoi]

        if len(bien_the) == 1:
            return tim_ung_vien_clip(bien_the[0], so_ung_vien)

        goc: dict[tuple, object] = {}
        dau_vao: dict[str, list[dict]] = {}

        for thu_tu, cum in enumerate(bien_the):
            hits = tim_ung_vien_clip(cum, so_ung_vien)
            ten = f"clip_bt{thu_tu}"
            dau_vao[ten] = []
            for h in hits:
                khoa = (str(h.video_id), round(float(h.pts_time), 3))
                goc.setdefault(khoa, h)
                dau_vao[ten].append(hit_sang_dict(h))

        import dataclasses

        ra = []
        for r in reciprocal_rank_fusion(dau_vao, k_rrf=k_rrf):
            khoa = (str(r["video_id"]), round(float(r.get("pts_time", 0.0)), 3))
            h = goc.get(khoa)
            if h is None:
                continue
            ra.append(dataclasses.replace(h, score=float(r["score"]), source="clip"))
        return ra

    return _tim
