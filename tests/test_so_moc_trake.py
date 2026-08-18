"""
Kiểm tra cách suy ra SỐ MỐC của một truy vấn TRAKE — Việc 4.

Bối cảnh: bản cũ `chuan_hoa_cau_hoi()` mặc định 4 mốc cho mọi câu TRAKE.
Đo trên bộ dev 32 câu thì 5/8 câu TRAKE có số mốc khác 4:

    câu 07          -> 2 mốc
    câu 08, 20, 24  -> 5 mốc
    câu 31, 32      -> 6 mốc
    câu 15, 16      -> 3 mốc

Sai hai đầu, và đầu nào cũng mất điểm lặng lẽ:

  - Câu nhiều hơn 4 mốc: dòng nộp thiếu cột, trần điểm bị cắt TRƯỚC khi
    tra cứu chạy. Câu 31 trần 0,167 thay vì 0,250.
  - Câu ít hơn 4 mốc: `_gom_trake()` loại mọi video không gom đủ 4 ứng
    viên, loại nhầm cả video lẽ ra hợp lệ. Câu 07 chỉ cần 2 mốc mà vẫn bị
    lọc theo chuẩn 4.

Chạy:

    pytest tests/test_so_moc_trake.py -v
"""

from __future__ import annotations

import pytest

from scripts.run_search import chuan_hoa_cau_hoi, dem_moc_tu_cau_hoi
from src.aic2026.rank.search import SO_MOC_TRAKE_MAC_DINH
from src.aic2026.submit import KIS, TRAKE


# ---------------------------------------------------------------------------
# 1. dem_moc_tu_cau_hoi — đường duy nhất dùng được ở VÒNG THI THẬT
# ---------------------------------------------------------------------------


def test_dem_dung_vi_du_trake_cua_btc():
    """Ví dụ nhảy cao mục 2.1.3 của BTC: bốn khoảnh khắc -> đếm ra 4."""
    cau = (
        "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: "
        "(1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy."
    )
    assert dem_moc_tu_cau_hoi(cau) == 4


def test_dem_dung_cau_hai_moc_va_sau_moc():
    """Đếm đúng ở cả hai đầu: câu 2 mốc và câu 6 mốc của bộ dev."""
    cau_07 = (
        "Các khoảnh khắc hành động cắt xả đập dập chính: "
        "(1) Dao vừa chạm vào cây xả, (2) Cây xả bị cắt rời ra"
    )
    cau_32 = (
        "Quá trình om thịt bê giấm nghệ: (1) Đặt nồi lên bếp, (2) mở lửa, "
        "(3) dầu được cho vào nồi, (4) sả được cho vào nồi, "
        "(5) hành tím được cho vào nồi, (6) gừng cắt lát được cho vào nồi"
    )
    assert dem_moc_tu_cau_hoi(cau_07) == 2
    assert dem_moc_tu_cau_hoi(cau_32) == 6


def test_dem_chiu_duoc_khoang_trang_trong_ngoac():
    """"( 1 )" viết thưa vẫn phải đếm được — người soạn tay gõ mỗi kiểu."""
    assert dem_moc_tu_cau_hoi("( 1 ) mở nắp, ( 2 ) đổ nước") == 2


def test_khong_dem_nham_so_trong_ngoac_binh_thuong():
    """
    Số trong ngoặc KHÔNG phải mốc thì không được đếm.

    Đây là lý do luật đếm bắt dãy phải bắt đầu từ 1 và tăng liên tục.
    Không có luật đó thì "(5) người mặc quân phục" thành 1 mốc, và câu mô
    tả nào có ngoặc đơn cũng thành TRAKE hỏng.
    """
    assert dem_moc_tu_cau_hoi("Có (5) người mặc quân phục xếp hàng.") is None
    assert dem_moc_tu_cau_hoi("Mạc Cửu (1655-1735) đứng trên bệ") is None
    assert dem_moc_tu_cau_hoi("Nhóm (2) và nhóm (4) cùng vào sân") is None


def test_khong_dem_khi_khong_co_gi():
    """Chuỗi rỗng, None, hoặc câu không đánh số -> None, không nổ."""
    assert dem_moc_tu_cau_hoi("") is None
    assert dem_moc_tu_cau_hoi(None) is None
    assert dem_moc_tu_cau_hoi("Một đĩa salad trái cây có hạt óc chó.") is None


def test_mot_moc_don_le_khong_tinh():
    """
    Chỉ có "(1)" thì không đủ để coi là dãy mốc.

    Một chuỗi sự kiện phải có ít nhất hai khoảnh khắc mới có nghĩa; "(1)"
    đứng một mình nhiều khả năng là đánh số đoạn văn.
    """
    assert dem_moc_tu_cau_hoi("(1) Bật bếp lên rồi nấu") is None


# ---------------------------------------------------------------------------
# 2. chuan_hoa_cau_hoi — thứ tự ưu tiên bốn nguồn
# ---------------------------------------------------------------------------


CAU_TRAKE_6_MOC = (
    "Quá trình ướp thịt bê: (1) hạt nêm, (2) đường, (3) bột cà ri, "
    "(4) hành tỏi băm, (5) xốt mayonnaise, (6) trộn đều"
)


def test_uu_tien_1_khai_thang_trong_record():
    """Record khai thẳng so_moc_trake thì tin con số đó, không đếm lại."""
    muc = {
        "id": "99",
        "loai_truy_van": "chuoi_su_kien",
        "cau_hoi": CAU_TRAKE_6_MOC,
        "so_moc_trake": 3,
        "cac_giai_doan": [{}, {}, {}, {}, {}, {}],
    }
    ket_qua = chuan_hoa_cau_hoi(muc, 1)
    assert ket_qua["so_moc_trake"] == 3
    assert ket_qua["nguon_so_moc"] == "khai trong record"


def test_uu_tien_2_dem_cac_giai_doan():
    """
    Không khai thẳng nhưng có cac_giai_doan -> đếm số giai đoạn.

    Đây là con số ĐÚNG THẬT của bộ dev tự soạn, chính xác hơn đếm câu chữ.
    """
    muc = {
        "id": "31",
        "loai_truy_van": "chuoi_su_kien",
        "cau_hoi": CAU_TRAKE_6_MOC,
        "video_id": "L26_V034",
        "cac_giai_doan": [{"frame_start": i, "frame_end": i + 5} for i in range(6)],
    }
    ket_qua = chuan_hoa_cau_hoi(muc, 1)
    assert ket_qua["so_moc_trake"] == 6
    assert ket_qua["nguon_so_moc"] == "đếm cac_giai_doan"


def test_uu_tien_3_dem_trong_cau_chu_khi_khong_co_dap_an():
    """
    Không có đáp án -> đếm trong câu chữ.

    ĐÂY LÀ ĐƯỜNG DUY NHẤT DÙNG ĐƯỢC Ở VÒNG THI THẬT: BTC phát câu hỏi,
    không phát cac_giai_doan. Bộ dev có đáp án nên hai nhánh trên che
    được, nhưng ngày thi thì chỉ còn nhánh này.
    """
    muc = {
        "id": "31",
        "loai_truy_van": "chuoi_su_kien",
        "cau_hoi": CAU_TRAKE_6_MOC,
    }
    ket_qua = chuan_hoa_cau_hoi(muc, 1)
    assert ket_qua["so_moc_trake"] == 6
    assert ket_qua["nguon_so_moc"] == "đếm trong câu chữ"


def test_uu_tien_4_mac_dinh_va_bao_ro_la_mac_dinh():
    """
    Không suy ra được -> dùng mặc định, NHƯNG phải ghi rõ nguồn là mặc
    định để người ngồi máy nhìn log biết con số này là đoán.

    Bản cũ im lặng dùng 4 cho mọi câu, nên không ai phát hiện ra 5/8 câu
    TRAKE của bộ dev đang bị cắt trần điểm.
    """
    muc = {
        "id": "88",
        "loai_truy_van": "chuoi_su_kien",
        "cau_hoi": "Tìm các khoảnh khắc người đàn ông xếp bánh vào nồi",
    }
    ket_qua = chuan_hoa_cau_hoi(muc, 1)
    assert ket_qua["so_moc_trake"] == SO_MOC_TRAKE_MAC_DINH
    assert "MẶC ĐỊNH" in ket_qua["nguon_so_moc"]


def test_so_moc_khai_sai_kieu_van_chay_tiep():
    """so_moc_trake khai bậy (chuỗi chữ) -> rơi xuống nguồn tiếp theo."""
    muc = {
        "id": "77",
        "loai_truy_van": "chuoi_su_kien",
        "cau_hoi": CAU_TRAKE_6_MOC,
        "so_moc_trake": "sáu",
    }
    ket_qua = chuan_hoa_cau_hoi(muc, 1)
    assert ket_qua["so_moc_trake"] == 6
    assert ket_qua["nguon_so_moc"] == "đếm trong câu chữ"


# ---------------------------------------------------------------------------
# 3. Không được đụng vào câu KIS / hỏi–đáp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cau_hoi",
    [
        "Có bao nhiêu người mặc quân phục dã ngoại rằn ri đang xếp hàng?",
        "Tượng màu đỏ của một vị quan đứng trên bệ có bảng ghi Mạc Cửu 1655-1735",
        "Đĩa salad gồm trái cây cắt nhỏ và hạt óc chó.",
    ],
)
def test_cau_khong_phai_trake_van_ra_so_moc_mac_dinh(cau_hoi):
    """
    Câu mô tả / hỏi–đáp: so_moc_trake không được dùng tới, nhưng cũng
    không được nổ. Giữ mặc định là đủ.
    """
    muc = {"id": "01", "loai_truy_van": "mo_ta", "cau_hoi": cau_hoi}
    ket_qua = chuan_hoa_cau_hoi(muc, 1)
    assert ket_qua["task"] == KIS
    assert ket_qua["so_moc_trake"] == SO_MOC_TRAKE_MAC_DINH


# ---------------------------------------------------------------------------
# 4. Chốt lại toàn bộ tám câu TRAKE của bộ dev hiện tại
# ---------------------------------------------------------------------------


SO_MOC_THAT_BO_DEV = {
    "07": 2,
    "08": 5,
    "15": 3,
    "16": 3,
    "20": 5,
    "24": 5,
    "31": 6,
    "32": 6,
}


@pytest.mark.parametrize("query_id,so_moc", sorted(SO_MOC_THAT_BO_DEV.items()))
def test_chot_so_moc_tam_cau_trake_bo_dev(query_id, so_moc):
    """
    Chốt cứng số mốc thật của tám câu TRAKE trong dev_questions.jsonl.

    Nếu ai sửa luật suy ra số mốc mà làm lệch một trong tám con số này thì
    test đỏ ngay, không phải chờ chấm xong mới thấy điểm tụt.
    """
    muc = {
        "id": query_id,
        "loai_truy_van": "chuoi_su_kien",
        "cau_hoi": "mô phỏng",
        "cac_giai_doan": [{"frame_start": 0, "frame_end": 1}] * so_moc,
    }
    ket_qua = chuan_hoa_cau_hoi(muc, 1)
    assert ket_qua["task"] == TRAKE
    assert ket_qua["so_moc_trake"] == so_moc