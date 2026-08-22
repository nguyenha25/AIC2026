"""
Đọc config/settings.yaml cho nhánh xếp hạng.

Lý do có tệp này: run_query() cần bốn con số (cửa sổ lọc trùng, số ứng viên
mỗi nguồn, hằng số RRF, trọng số nguồn). Nếu gõ thẳng vào search.py thì mỗi
lần chỉnh tham số lại phải sửa mã nguồn, và runs/ sẽ không ghi lại được lần
chạy đó dùng số gì.

MỘT CHỖ LỆCH ĐÃ BIẾT — ĐỌC TRƯỚC KHI SỬA
-----------------------------------------
config/settings.yaml hiện ghi:

    loc_trung:
      khoang_cach_toi_thieu: 250      # đơn vị KHUNG HÌNH
      so_anh_toi_da_moi_video: 3

Luật chốt trong tài liệu Giai đoạn 1 (Việc 4) lại là:

    mỗi video giữ tối đa MỘT kết quả trong mỗi cửa sổ 10 GIÂY,
    tính theo cột giây của bảng đối chiếu

Hai luật này KHÔNG tương đương:

  - 250 khung ≈ 10 giây chỉ khi video 25fps. fps mỗi video một khác, nên
    cùng một con số 250 sẽ ra cửa sổ dài ngắn khác nhau tuỳ video.
  - Luật chốt KHÔNG có giới hạn 3 ảnh mỗi video. Nếu một video thật sự chứa
    năm cảnh khác nhau cách nhau hơn 10 giây thì cả năm đều hợp lệ.

Mã ở đây chạy theo LUẬT CHỐT (giây). Khoá cũ vẫn để nguyên trong settings.yaml,
không xoá, nhưng bị bỏ qua và có cảnh báo in ra. Người đứng tên Việc 4 (Ngân)
quyết định giữ khoá nào — sửa settings.yaml TRƯỚC, sửa mã sau.

Muốn đổi cửa sổ (tài liệu ghi 10 giây chỉ là mốc khởi đầu, chỉnh lại sau khi
có bộ câu hỏi tự chấm) thì thêm vào settings.yaml:

    loc_trung:
      cua_so_giay: 10.0
"""

from __future__ import annotations

from typing import Any

import yaml

from ..paths import CONFIG_DIR

SETTINGS_PATH = CONFIG_DIR / "settings.yaml"

# Luật chốt Việc 4. Chỉ dùng khi settings.yaml không khai cua_so_giay.
CUA_SO_GIAY_MAC_DINH = 10.0

# Lấy dư rồi mới gộp và lọc — lấy đúng 100 thì lọc trùng xong không còn đủ 100.
SO_UNG_VIEN_MAC_DINH = 500

RRF_K_MAC_DINH = 60

TRONG_SO_MAC_DINH = {
    "clip": 1.0,
    "ocr": 0.6,
    "ocr_fts": 0.6,
    "asr": 0.6,
    "object": 0.4,
    "caption": 0.5,
}

_CACHED: dict | None = None


def load_settings(refresh: bool = False) -> dict[str, Any]:
    """Đọc settings.yaml một lần rồi giữ trong RAM."""
    global _CACHED

    if refresh:
        _CACHED = None

    if _CACHED is not None:
        return _CACHED

    if not SETTINGS_PATH.exists():
        _CACHED = {}
        return _CACHED

    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        _CACHED = yaml.safe_load(f) or {}

    return _CACHED


def _muc(ten: str) -> dict[str, Any]:
    gia_tri = load_settings().get(ten)
    return gia_tri if isinstance(gia_tri, dict) else {}


# ---------------------------------------------------------------------------
# Bốn con số nhánh xếp hạng cần
# ---------------------------------------------------------------------------


def cua_so_giay() -> float:
    """Cửa sổ lọc trùng, tính bằng GIÂY. Mặc định 10.0 theo luật chốt."""
    gia_tri = _muc("loc_trung").get("cua_so_giay", CUA_SO_GIAY_MAC_DINH)
    return float(gia_tri)


def so_anh_toi_da_moi_video() -> int | None:
    """
    Trần số ảnh mỗi video, hoặc None nghĩa là không giới hạn.

    Trả về None theo mặc định: luật chốt Việc 4 KHÔNG có trần này. Muốn bật
    thì khai loc_trung.bat_tran_moi_video: true trong settings.yaml — và phải
    được Ngân đồng ý trước, vì nó đổi bản chất luật lọc trùng.
    """
    muc = _muc("loc_trung")

    if not muc.get("bat_tran_moi_video", False):
        return None

    gia_tri = muc.get("so_anh_toi_da_moi_video")
    return int(gia_tri) if gia_tri else None


def so_ung_vien_moi_nguon() -> int:
    """Số ứng viên lấy về từ MỖI nguồn trước khi gộp và lọc."""
    gia_tri = _muc("tra_cuu").get("so_ung_vien_moi_nguon", SO_UNG_VIEN_MAC_DINH)
    return int(gia_tri)


def rrf_k() -> int:
    """Hằng số k của Reciprocal Rank Fusion."""
    gia_tri = _muc("gop_ket_qua").get("rrf_k", RRF_K_MAC_DINH)
    return int(gia_tri)


def trong_so_nguon() -> dict[str, float]:
    """Trọng số từng nguồn khi gộp bằng RRF."""
    khai_bao = _muc("gop_ket_qua").get("trong_so")

    if not isinstance(khai_bao, dict):
        return dict(TRONG_SO_MAC_DINH)

    ket_qua = dict(TRONG_SO_MAC_DINH)
    ket_qua.update({str(k): float(v) for k, v in khai_bao.items()})
    return ket_qua


def trong_so_theo_dang(task: str | None = None) -> dict[str, float]:
    """Trọng số RRF do Việc 6 đo được, đọc từ config/rrf_weights.yaml.

    Chưa có tệp đó (hoặc dạng câu chưa đo được) thì lùi về trong_so_nguon()
    của settings.yaml. KHÔNG bịa số: TRAKE hiện chưa đo được nên nó dùng bộ
    mặc định, và điều đó phải nhìn thấy được chứ không giấu đi.
    """
    tep = CONFIG_DIR / "rrf_weights.yaml"
    if not tep.exists():
        return trong_so_nguon()

    with tep.open("r", encoding="utf-8") as f:
        noi_dung = yaml.safe_load(f) or {}

    if task:
        theo_dang = noi_dung.get("theo_dang") or {}
        rieng = theo_dang.get(str(task))
        if isinstance(rieng, dict) and rieng:
            ket_qua = trong_so_nguon()
            ket_qua.update({str(k): float(v) for k, v in rieng.items()})
            return ket_qua

    chung = noi_dung.get("mac_dinh")
    if isinstance(chung, dict) and chung:
        ket_qua = trong_so_nguon()
        ket_qua.update({str(k): float(v) for k, v in chung.items()})
        return ket_qua

    return trong_so_nguon()


def so_dong_toi_da() -> int:
    """Số dòng tối đa một truy vấn. BTC cho 100."""
    gia_tri = _muc("nop_bai").get("so_dong_toi_da", 100)
    return int(gia_tri)


# ---------------------------------------------------------------------------
# Cảnh báo lệch cấu hình
# ---------------------------------------------------------------------------


def canh_bao_cau_hinh() -> list[str]:
    """
    Danh sách cảnh báo về chỗ settings.yaml lệch với luật chốt.

    Rỗng nghĩa là không có gì phải báo. Script chạy đầu–cuối in danh sách này
    LÊN ĐẦU, không giấu xuống cuối — người chạy phải thấy trước khi tin kết quả.
    """
    canh_bao: list[str] = []
    muc = _muc("loc_trung")

    if "khoang_cach_toi_thieu" in muc and "cua_so_giay" not in muc:
        canh_bao.append(
            f"settings.yaml khai loc_trung.khoang_cach_toi_thieu="
            f"{muc['khoang_cach_toi_thieu']} (đơn vị KHUNG HÌNH). "
            f"Luật chốt Việc 4 tính bằng GIÂY nên khoá này bị BỎ QUA, "
            f"đang chạy với cửa sổ {cua_so_giay():.1f} giây. "
            "Ngân quyết định: đổi settings.yaml sang cua_so_giay, hoặc đổi luật."
        )

    if "so_anh_toi_da_moi_video" in muc and not muc.get("bat_tran_moi_video", False):
        canh_bao.append(
            f"settings.yaml khai loc_trung.so_anh_toi_da_moi_video="
            f"{muc['so_anh_toi_da_moi_video']} nhưng luật chốt không có trần này, "
            "nên đang BỎ QUA. Muốn bật thì thêm bat_tran_moi_video: true "
            "và báo Ngân trước."
        )

    return canh_bao


def tham_so_da_dung() -> dict[str, Any]:
    """Gói tham số để ghi vào runs/<lần chạy>/params.json."""
    return {
        "cua_so_giay": cua_so_giay(),
        "so_anh_toi_da_moi_video": so_anh_toi_da_moi_video(),
        "so_ung_vien_moi_nguon": so_ung_vien_moi_nguon(),
        "rrf_k": rrf_k(),
        "trong_so": trong_so_nguon(),
        "so_dong_toi_da": so_dong_toi_da(),
    }