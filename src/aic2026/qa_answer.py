"""
Việc 11 — SINH ĐÁP ÁN Q&A TỪ ĐẦU ĐẾN CUỐI.

TÌNH TRẠNG TRƯỚC VIỆC NÀY
-------------------------
Đường này CHƯA TỪNG chạy trọn vẹn. Điểm baseline Q&A đo bằng đáp án ĐIỀN SẴN
— tức là đo mạch tìm ảnh, không đo khả năng trả lời. Vòng p1 mất trắng Q9 và
Q3 vì không có đáp án.

LUẬT CỨNG, KHÔNG CÓ NGOẠI LỆ
----------------------------
`tra_loi()` KHÔNG BAO GIỜ trả chuỗi rỗng. Vòng p1 đã có một ô trống làm BTC
loại nguyên tệp 100 dòng — mất cả câu chứ không phải mất điểm câu đó. Đoán
bừa vẫn giữ tệp hợp lệ và không mất gì thêm.

Khi mọi nguồn đều câm, hàm trả về `DAP_AN_DU_PHONG` kèm do_tin = 0.0. Con số
độ tin đó để người ngồi máy biết ô nào cần soi lại, KHÔNG phải để lọc bỏ.

BỐN NGUỒN, THEO ĐÚNG THỨ TỰ ƯU TIÊN
-----------------------------------
1. OCR trên đúng khung hình  — câu hỏi về chữ/số HIỆN TRÊN MÀN HÌNH. Chính
   xác nhất khi trúng, vì đó là chữ thật chứ không phải mô hình đoán.
2. VLM đọc ảnh              — câu hỏi về thứ nhìn thấy (đếm, màu, hành động).
3. ASR quanh mốc thời gian  — câu hỏi mà đáp án nằm trong lời nói.
4. Dự phòng                 — đoán theo dạng câu hỏi. Không bao giờ rỗng.

Ảnh phải là ẢNH GỐC raw/keyframes/, KHÔNG dùng thumbnail: thumbnail 224×126
không đọc nổi chữ và số, mà chữ và số đúng là thứ Q&A hay hỏi.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

DAP_AN_DU_PHONG = "khong ro"

# Số viết bằng chữ trong tiếng Việt -> chữ số, để chuẩn hoá đáp án đếm.
SO_CHU = {
    "khong": "0", "mot": "1", "hai": "2", "ba": "3", "bon": "4", "tu": "4",
    "nam": "5", "sau": "6", "bay": "7", "tam": "8", "chin": "9", "muoi": "10",
}


def _khong_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("\u0111", "d")


# ---------------------------------------------------------------------------
# Nhận dạng loại câu hỏi
# ---------------------------------------------------------------------------

DEM = "dem"          # bao nhiêu, mấy
CHU_TREN_HINH = "chu_tren_hinh"   # tên chương trình, biển số, dòng chữ
MAU = "mau"
THOI_GIAN = "thoi_gian"
DIA_DIEM = "dia_diem"
KHAC = "khac"

_MAU_CAU = [
    (DEM, ["bao nhieu", "may nguoi", "may cai", "may chiec", "so luong", "dem duoc"]),
    (CHU_TREN_HINH, [
        "dong chu", "chu tren", "bien so", "so hieu", "ten chuong trinh",
        "tieu de", "khau hieu", "bang hieu", "ghi gi", "viet gi", "so dien thoai",
    ]),
    (MAU, ["mau gi", "mau sac", "mau nao"]),
    (THOI_GIAN, ["nam nao", "ngay nao", "thang nao", "luc may gio", "thoi gian nao"]),
    (DIA_DIEM, ["o dau", "dia diem nao", "tinh nao", "thanh pho nao", "noi nao"]),
]


def loai_cau_hoi(cau_hoi: str) -> str:
    van_ban = _khong_dau(cau_hoi)
    for loai, dau_hieu in _MAU_CAU:
        if any(d in van_ban for d in dau_hieu):
            return loai
    return KHAC


# ---------------------------------------------------------------------------
# Kết quả
# ---------------------------------------------------------------------------

@dataclass
class DapAn:
    """Một đáp án Q&A. `van_ban` BẢO ĐẢM không rỗng."""

    van_ban: str
    do_tin: float                      # 0..1 — để soi lại, KHÔNG để lọc bỏ
    nguon: str                         # ocr / vlm / asr / du_phong
    video_id: str = ""
    frame_idx: int = -1
    giai_thich: str = ""
    ung_vien_khac: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.van_ban = (self.van_ban or "").strip()
        if not self.van_ban:
            # Luật cứng. Không nơi nào trong dự án được phép tạo ra DapAn rỗng.
            self.van_ban = DAP_AN_DU_PHONG
            self.do_tin = 0.0
            self.nguon = "du_phong"


# ---------------------------------------------------------------------------
# Nguồn 1 — OCR trên đúng khung hình
# ---------------------------------------------------------------------------

def _doc_ocr_cua_khung(video_id: str, n: int) -> list[dict]:
    """Các hộp OCR của MỘT keyframe, đọc từ derived/ocr/<video>.jsonl."""
    from .paths import ocr_file

    tep = ocr_file(video_id)
    if not tep.exists():
        return []

    for dong in tep.open("r", encoding="utf-8"):
        dong = dong.strip()
        if not dong:
            continue
        try:
            d = json.loads(dong)
        except json.JSONDecodeError:
            continue
        if int(d.get("n", -1)) == int(n):
            return d.get("boxes") or []
    return []


def tra_loi_bang_ocr(cau_hoi: str, video_id: str, n: int) -> DapAn | None:
    """Rút đáp án từ chữ đọc được trên đúng tấm ảnh đó."""
    hop = _doc_ocr_cua_khung(video_id, n)
    if not hop:
        return None

    loai = loai_cau_hoi(cau_hoi)

    # Hộp có độ tin cao đứng trước. 60% hộp dưới 0,40 (trung vị 0,276) nên
    # phải xếp theo conf chứ không lấy bừa hộp đầu tiên.
    hop = sorted(hop, key=lambda b: float(b.get("conf", 0.0)), reverse=True)

    if loai == THOI_GIAN:
        for b in hop:
            khop = re.search(r"\b(19|20)\d{2}\b", str(b.get("text", "")))
            if khop:
                return DapAn(
                    khop.group(0), float(b.get("conf", 0.5)), "ocr",
                    video_id, -1, "năm đọc được trên hình",
                )

    if loai in (CHU_TREN_HINH, KHAC):
        for b in hop:
            chu = str(b.get("text", "")).strip()
            conf = float(b.get("conf", 0.0))
            if len(chu) >= 2 and conf >= 0.40:
                return DapAn(
                    chu, conf, "ocr", video_id, -1,
                    "dòng chữ rõ nhất trên hình",
                    [str(x.get("text", "")).strip() for x in hop[1:4]],
                )

    return None


# ---------------------------------------------------------------------------
# Nguồn 2 — mô hình đọc ảnh
# ---------------------------------------------------------------------------

class BoDocAnh:
    """VLM trả lời câu hỏi về một tấm ảnh. Mặc định BLIP-VQA (nhẹ, tất định)."""

    def __init__(self, ten_mo_hinh: str = "Salesforce/blip-vqa-base", thiet_bi=None):
        self.ten_mo_hinh = ten_mo_hinh
        self._thiet_bi_yeu_cau = thiet_bi
        self._processor = None
        self._model = None
        self._thiet_bi = None

    def _nap(self):
        if self._model is not None:
            return
        import torch
        from transformers import BlipForQuestionAnswering, BlipProcessor

        self._thiet_bi = self._thiet_bi_yeu_cau or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._processor = BlipProcessor.from_pretrained(self.ten_mo_hinh)
        self._model = BlipForQuestionAnswering.from_pretrained(self.ten_mo_hinh)
        self._model.to(self._thiet_bi)
        self._model.eval()

    def hoi(self, duong_dan_anh: Path, cau_hoi_tieng_anh: str) -> str:
        import torch
        from PIL import Image

        self._nap()
        with Image.open(duong_dan_anh) as im:
            anh = im.convert("RGB")
            dau_vao = self._processor(
                anh, cau_hoi_tieng_anh, return_tensors="pt"
            ).to(self._thiet_bi)

        with torch.no_grad():
            ra = self._model.generate(**dau_vao, max_new_tokens=20, num_beams=3,
                                      do_sample=False)
        return self._processor.decode(ra[0], skip_special_tokens=True).strip()


def _cau_hoi_sang_tieng_anh(cau_hoi: str) -> str:
    """BLIP-VQA chỉ hiểu tiếng Anh. Dùng lại Việc 5, không dựng đường thứ hai."""
    try:
        from .query_expand import mo_rong

        loai = loai_cau_hoi(cau_hoi)
        loi = mo_rong(cau_hoi).cum_chinh
        if loai == DEM:
            return f"how many {loi}?"
        if loai == MAU:
            return f"what color is the {loi}?"
        return f"what is in this image? {loi}"
    except Exception:
        return cau_hoi


def tra_loi_bang_vlm(
    cau_hoi: str,
    video_id: str,
    n: int,
    bo_doc: BoDocAnh | None = None,
) -> DapAn | None:
    from .paths import keyframe_image

    anh = keyframe_image(video_id, n)
    if not anh.exists():
        return None

    bo_doc = bo_doc or BoDocAnh()
    try:
        tra_loi = bo_doc.hoi(anh, _cau_hoi_sang_tieng_anh(cau_hoi))
    except Exception as loi:
        return DapAn(DAP_AN_DU_PHONG, 0.0, "du_phong", video_id, -1, f"VLM lỗi: {loi}")

    if not tra_loi:
        return None

    if loai_cau_hoi(cau_hoi) == DEM:
        tra_loi = chuan_hoa_so(tra_loi)

    return DapAn(tra_loi, 0.6, "vlm", video_id, -1, f"VLM {bo_doc.ten_mo_hinh}")


def chuan_hoa_so(van_ban: str) -> str:
    """'three' / 'ba' / 'có 3 người' -> '3'. Trả nguyên bản nếu không thấy số."""
    khop = re.search(r"\d+", van_ban)
    if khop:
        return khop.group(0)

    anh_viet = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10",
    }
    tu = _khong_dau(van_ban).split()
    for t in tu:
        if t in anh_viet:
            return anh_viet[t]
        if t in SO_CHU:
            return SO_CHU[t]
    return van_ban.strip()


# ---------------------------------------------------------------------------
# Nguồn 3 — lời nói quanh mốc thời gian
# ---------------------------------------------------------------------------

def tra_loi_bang_asr(
    cau_hoi: str,
    video_id: str,
    pts_time: float,
    cua_so_giay: float = 8.0,
) -> DapAn | None:
    """Lấy câu nói bao quanh mốc thời gian. Chỉ hợp với câu hỏi về lời nói."""
    from .paths import asr_file

    tep = asr_file(video_id)
    if not tep.exists():
        return None

    gan_nhat = None
    for dong in tep.open("r", encoding="utf-8"):
        dong = dong.strip()
        if not dong:
            continue
        try:
            d = json.loads(dong)
        except json.JSONDecodeError:
            continue
        bat_dau, ket_thuc = float(d.get("start", 0)), float(d.get("end", 0))
        if bat_dau - cua_so_giay <= pts_time <= ket_thuc + cua_so_giay:
            khoang_cach = min(abs(pts_time - bat_dau), abs(pts_time - ket_thuc))
            if gan_nhat is None or khoang_cach < gan_nhat[0]:
                gan_nhat = (khoang_cach, d.get("text", ""))

    if not gan_nhat or not gan_nhat[1].strip():
        return None

    if loai_cau_hoi(cau_hoi) == THOI_GIAN:
        khop = re.search(r"\b(19|20)\d{2}\b", gan_nhat[1])
        if khop:
            return DapAn(khop.group(0), 0.5, "asr", video_id, -1, "năm nghe được")

    return DapAn(
        gan_nhat[1].strip()[:120], 0.35, "asr", video_id, -1,
        f"lời nói cách mốc {gan_nhat[0]:.1f} giây",
    )


# ---------------------------------------------------------------------------
# Nguồn 4 — dự phòng, KHÔNG BAO GIỜ RỖNG
# ---------------------------------------------------------------------------

def doan_du_phong(cau_hoi: str) -> DapAn:
    """Đoán theo dạng câu hỏi khi mọi nguồn đều câm.

    Không phải để ăn điểm — để tệp nộp không có ô trống. Ô trống làm BTC loại
    NGUYÊN TỆP 100 dòng; đoán sai chỉ mất đúng câu đó.
    """
    loai = loai_cau_hoi(cau_hoi)
    goi_y = {
        DEM: ("1", "câu đếm không đọc được -> điền 1 (giá trị hay gặp nhất)"),
        MAU: ("do", "câu hỏi màu không đọc được -> điền màu hay gặp nhất"),
        THOI_GIAN: ("2024", "câu hỏi năm không đọc được -> điền năm gần nhất"),
    }
    van_ban, ly_do = goi_y.get(loai, (DAP_AN_DU_PHONG, "không nguồn nào trả lời được"))
    return DapAn(van_ban, 0.0, "du_phong", giai_thich=ly_do)


# ---------------------------------------------------------------------------
# Cửa chính
# ---------------------------------------------------------------------------

def tra_loi(
    cau_hoi: str,
    ung_vien: Sequence,
    so_ung_vien_doc: int = 5,
    bo_doc_anh: BoDocAnh | None = None,
    dung_vlm: bool = True,
) -> DapAn:
    """Sinh đáp án cho MỘT câu Q&A từ danh sách ứng viên đã xếp hạng.

    Đọc `so_ung_vien_doc` ứng viên đầu chứ không chỉ ứng viên số 1: ảnh hạng 1
    có thể đúng cảnh mà chưa hiện con số cần đọc. Nhiều ứng viên cho cùng một
    đáp án thì độ tin được cộng thêm.

    BẢO ĐẢM: giá trị trả về có `van_ban` KHÁC RỖNG trong mọi trường hợp.
    """
    ung_vien = list(ung_vien)[:so_ung_vien_doc]
    if not ung_vien:
        return doan_du_phong(cau_hoi)

    thu_duoc: list[DapAn] = []

    for h in ung_vien:
        vid, n = str(h.video_id), int(h.n)

        d = tra_loi_bang_ocr(cau_hoi, vid, n)
        if d is not None:
            d.frame_idx = int(h.frame_idx)
            thu_duoc.append(d)
            continue

        if dung_vlm:
            d = tra_loi_bang_vlm(cau_hoi, vid, n, bo_doc_anh)
            if d is not None and d.nguon != "du_phong":
                d.frame_idx = int(h.frame_idx)
                thu_duoc.append(d)
                continue

        d = tra_loi_bang_asr(cau_hoi, vid, float(h.pts_time))
        if d is not None:
            d.frame_idx = int(h.frame_idx)
            thu_duoc.append(d)

    if not thu_duoc:
        du_phong = doan_du_phong(cau_hoi)
        du_phong.video_id = str(ung_vien[0].video_id)
        du_phong.frame_idx = int(ung_vien[0].frame_idx)
        return du_phong

    # Nhiều ứng viên cùng một đáp án là dấu hiệu mạnh -> cộng độ tin.
    dem = Counter(_khong_dau(d.van_ban) for d in thu_duoc)
    pho_bien, so_lan = dem.most_common(1)[0]

    tot_nhat = max(
        (d for d in thu_duoc if _khong_dau(d.van_ban) == pho_bien),
        key=lambda d: d.do_tin,
    )
    if so_lan > 1:
        tot_nhat.do_tin = min(1.0, tot_nhat.do_tin + 0.1 * (so_lan - 1))
        tot_nhat.giai_thich += f" ({so_lan}/{len(thu_duoc)} ứng viên cùng đáp án)"

    tot_nhat.ung_vien_khac = [
        d.van_ban for d in thu_duoc if _khong_dau(d.van_ban) != pho_bien
    ][:4]
    return tot_nhat


def kiem_khong_o_trong(dap_an_theo_cau: dict[str, str]) -> list[str]:
    """Trả về danh sách query_id có ô đáp án rỗng. Rỗng nghĩa là ĐẠT.

    Gọi hàm này TRƯỚC khi ghi tệp nộp. Xem thêm Việc 14 ở submit/formatter.py.
    """
    return [
        str(q)
        for q, a in dap_an_theo_cau.items()
        if a is None or not str(a).strip()
    ]
