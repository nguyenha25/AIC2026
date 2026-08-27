"""
Việc 11 (sinh đáp án Q&A) và Việc 12 (ghép thời gian TRAKE) — Nguyên.

Chỉ import mã của Nguyên — chạy được ngay cả khi các gói khác CHƯA trộn.
Đây là lý do bộ test được tách theo chủ: một tệp test chung sẽ nạp cả 11 việc
ở cấp module, nên pytest hỏng ngay lúc thu thập nếu thiếu bất kỳ gói nào,
và người mới trộn một nhánh không chạy nổi test của chính mình.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pytest

# ===========================================================================
# VIỆC 11 — sinh đáp án, phần không cần mô hình
# ===========================================================================

from aic2026 import qa_answer  # noqa: E402


def test_du_phong_khong_bao_gio_rong():
    for cau in [
        "Có bao nhiêu người trên sân khấu?",
        "Dòng chữ trên biển hiệu là gì?",
        "Áo của diễn giả màu gì?",
        "",
        "abc xyz",
    ]:
        d = qa_answer.doan_du_phong(cau)
        assert d.van_ban.strip()


def test_dap_an_rong_tu_bien_thanh_du_phong():
    d = qa_answer.DapAn("   ", 0.9, "vlm")
    assert d.van_ban == qa_answer.DAP_AN_DU_PHONG
    assert d.do_tin == 0.0
    assert d.nguon == "du_phong"


def test_nhan_dang_loai_cau_hoi():
    assert qa_answer.loai_cau_hoi("Có bao nhiêu chiếc nón?") == qa_answer.DEM
    assert qa_answer.loai_cau_hoi("Biển số xe là gì?") == qa_answer.CHU_TREN_HINH
    assert qa_answer.loai_cau_hoi("Áo màu gì?") == qa_answer.MAU
    assert qa_answer.loai_cau_hoi("Sự kiện diễn ra năm nào?") == qa_answer.THOI_GIAN


def test_chuan_hoa_so():
    assert qa_answer.chuan_hoa_so("có 3 người") == "3"
    assert qa_answer.chuan_hoa_so("three") == "3"
    assert qa_answer.chuan_hoa_so("ba") == "3"
    assert qa_answer.chuan_hoa_so("đỏ") == "đỏ"


def test_tra_loi_khong_co_ung_vien_van_ra_dap_an():
    d = qa_answer.tra_loi("Có bao nhiêu người?", [])
    assert d.van_ban.strip()
    assert d.nguon == "du_phong"


def test_kiem_khong_o_trong():
    assert qa_answer.kiem_khong_o_trong({"1": "5", "2": "đỏ"}) == []
    assert set(qa_answer.kiem_khong_o_trong({"1": "", "2": None, "3": "x"})) == {"1", "2"}



# ===========================================================================
# VIỆC 12 — ghép thời gian TRAKE
# ===========================================================================

from aic2026.trake_align import ghep_tang_dan, tach_su_kien  # noqa: E402


def test_tach_su_kien_theo_dau_phay():
    cum = tach_su_kien(
        "Vận động viên nhảy cao: chạy đà, giậm nhảy, bay qua xà, tiếp đất"
    )
    assert cum == ["chạy đà", "giậm nhảy", "bay qua xà", "tiếp đất"]


def test_tach_su_kien_theo_tu_noi():
    cum = tach_su_kien("người đàn ông mở cửa rồi bước vào sau đó ngồi xuống")
    assert len(cum) == 3


def test_tach_su_kien_ep_dung_so_moc():
    assert len(tach_su_kien("a, b, c, d, e", so_moc=3)) == 3
    assert len(tach_su_kien("chỉ một sự kiện", so_moc=4)) == 4


def test_ghep_tang_dan_chon_dung_dinh_cao():
    # 3 sự kiện, 6 khung. Điểm cao nhất nằm ở khung 1, 3, 5 theo thứ tự.
    ma_tran = np.array(
        [
            [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.9, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.9, 0.1],
        ]
    )
    assert ghep_tang_dan(ma_tran) == [0, 2, 4]


def test_ghep_tang_dan_bat_buoc_tang_nghiem_ngat():
    """Cả ba sự kiện đều thích khung 2, nhưng không được dùng chung một khung."""
    ma_tran = np.array(
        [
            [0.1, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
        ]
    )
    chon = ghep_tang_dan(ma_tran)
    assert len(chon) == 3
    assert chon == sorted(chon)
    assert len(set(chon)) == 3


def test_ghep_tang_dan_thieu_khung_thi_bao_loi():
    with pytest.raises(ValueError, match="khung"):
        ghep_tang_dan(np.zeros((4, 3)))




# ===========================================================================
# VIỆC 9 — trích khung dày (phần không cần ffmpeg)
# ===========================================================================

import json as _json  # noqa: E402


def test_khoang_tu_dev_bao_trum_moi_giai_doan_va_co_dem(tmp_path):
    from scripts.trich_khung_day import khoang_tu_dev

    tep = tmp_path / "dev.jsonl"
    tep.write_text(
        "\n".join(
            [
                _json.dumps(
                    {
                        "id": "15",
                        "loai_truy_van": "chuoi_su_kien",
                        "video_id": "L30_V023",
                        "cac_giai_doan": [
                            {"frame_start": 1000, "frame_end": 1010},
                            {"frame_start": 1200, "frame_end": 1215},
                            {"frame_start": 1400, "frame_end": 1408},
                        ],
                    },
                    ensure_ascii=False,
                ),
                _json.dumps(
                    {"id": "1", "loai_truy_van": "mo_ta", "video_id": "L21_V001"},
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    ra = khoang_tu_dev(tep, dem=90)
    assert len(ra) == 1                      # câu mo_ta bị bỏ qua
    vid, dau, cuoi, qid = ra[0]
    assert (vid, qid) == ("L30_V023", "15")
    assert dau == 1000 - 90                  # bao trùm giai đoạn ĐẦU, có đệm
    assert cuoi == 1408 + 90                 # bao trùm giai đoạn CUỐI, có đệm


def test_khoang_khong_bao_gio_am(tmp_path):
    """Sự kiện ở đầu video: đệm không được đẩy khung đầu xuống số âm."""
    from scripts.trich_khung_day import khoang_tu_dev

    tep = tmp_path / "dev.jsonl"
    tep.write_text(
        _json.dumps(
            {
                "id": "1",
                "loai_truy_van": "chuoi_su_kien",
                "video_id": "L23_V025",
                "cac_giai_doan": [{"frame_start": 5, "frame_end": 20}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert khoang_tu_dev(tep, dem=90)[0][1] == 0


def test_cham_diem_trake_dung_cong_thuc_btc():
    from scripts.chay_trake import cham

    q = {
        "video_id": "L23_V025",
        "cac_giai_doan": [
            {"frame_start": 70, "frame_end": 78},
            {"frame_start": 110, "frame_end": 118},
            {"frame_start": 150, "frame_end": 158},
            {"frame_start": 190, "frame_end": 198},
        ],
    }
    assert cham(q, [74, 114, 154, 194]) == 1.0        # trúng cả 4
    assert cham(q, [0, 65, 130, 195]) == 0.25         # trúng 1
    assert cham(q, [0, 0, 0, 0]) == 0.0


def test_moc_phai_dung_THU_TU_moi_tinh_diem():
    """BTC chấm mốc thứ j theo khoảng của sự kiện thứ j — đảo thứ tự là trượt."""
    from scripts.chay_trake import cham

    q = {
        "video_id": "V",
        "cac_giai_doan": [
            {"frame_start": 70, "frame_end": 78},
            {"frame_start": 190, "frame_end": 198},
        ],
    }
    assert cham(q, [74, 194]) == 1.0
    assert cham(q, [194, 74]) == 0.0     # đúng hai số, sai thứ tự -> 0


def test_bo_sinh_nop_goi_ghep_trake_thay_vi_cong_25():
    ma = (_ROOT / "scripts/tao_bo_nop.py").read_text(encoding="utf-8")
    than = "\n".join(ma.split('"""')[::2])
    assert "from aic2026.trake_align import ghep" in than
    assert "ket = ghep(" in than
    assert "goc + j * 25" not in than


def test_chi_tiet_tung_moc_bao_thieu_moc():
    from scripts.chay_trake import chi_tiet_tung_moc

    q = {"cac_giai_doan": [{"frame_start": 10, "frame_end": 20}] * 3}
    ra = chi_tiet_tung_moc(q, [15])
    assert "TRÚNG" in ra[0]
    assert "THIẾU" in ra[1] and "THIẾU" in ra[2]


def test_cat_tien_to_danh_so_kieu_ngoac():
    """Đề THẬT viết "(1) nhấc bánh khỏi rổ" — có ngoặc mở.

    Bản đầu chỉ bắt "1)" nên tiền tố "(1) " còn nguyên và đi thẳng vào phần
    chấm điểm. Đã cắn ở câu 15 và 16 bộ dev.
    """
    from aic2026.trake_align import tach_su_kien

    cum = tach_su_kien(
        "Tìm 3 khoảnh khắc: (1) nhấc bánh khỏi rổ, (2) đặt bánh vào nồi, "
        "(3) đậy nắp nồi"
    )
    assert cum == ["nhấc bánh khỏi rổ", "đặt bánh vào nồi", "đậy nắp nồi"]

    # các kiểu đánh số khác vẫn phải cắt được
    assert tach_su_kien("1) chạy đà, 2. giậm nhảy, [3] tiếp đất") == [
        "chạy đà", "giậm nhảy", "tiếp đất"
    ]


def test_cac_moc_trake_phai_ra_cum_tieng_anh_KHAC_NHAU():
    """Ba mốc rút về cùng một cụm thì ma trận điểm có ba hàng giống hệt nhau,
    và phần ghép chỉ lấy được đỉnh cao nhất với vài khung kề bên.

    Đây chính là thứ làm câu 15 ra 0,000 dù đã được phát sẵn cửa sổ 14 giây
    chắc chắn chứa đáp án.
    """
    from aic2026.query_expand import dich_bang_tu_dien
    from aic2026.trake_align import tach_su_kien

    cau = (
        "Tìm 3 khoảnh khắc: (1) nhấc bánh khỏi rổ, (2) đặt bánh vào nồi, "
        "(3) đậy nắp nồi"
    )
    cum = [dich_bang_tu_dien(sk).cum_chinh for sk in tach_su_kien(cau)]

    assert len(set(cum)) == len(cum), f"có cụm trùng nhau: {cum}"
    for c in cum:
        assert not any("\u00e0" <= ch <= "\u1ef9" for ch in c), (
            f"{c!r} còn chữ tiếng Việt — nhánh CLIP sẽ chạy ở mốc sàn"
        )


def test_dong_tu_duoc_giu_lai():
    """Động từ là thứ phân biệt các mốc — mất nó là mất cả bài toán TRAKE."""
    from aic2026.query_expand import dich_bang_tu_dien

    assert "lifting" in dich_bang_tu_dien("nhấc bánh").cum_chinh
    assert "placing" in dich_bang_tu_dien("đặt bánh").cum_chinh
    assert "lid" in dich_bang_tu_dien("đậy nắp").cum_chinh


def test_tach_su_kien_dinh_dang_THAT_cua_btc():
    """Đề THẬT của BTC đánh dấu E1/E2/E3 ở đầu dòng, dòng đầu là BỐI CẢNH.

    Bản đầu chỉ xử lý định dạng "(1)(2)(3)" một dòng của bộ dev tự soạn, nên
    trên đề thật nó cắt dòng bối cảnh theo dấu phẩy rồi dừng — không bao giờ
    chạm tới E1. TRAKE sẽ hỏng trên đề thật bất kể mật độ khung.
    """
    from aic2026.trake_align import tach_su_kien

    de = (
        "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, "
        "bên cạnh lá cờ trắng viền đỏ.\n"
        "E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng.\n"
        "E2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ.\n"
        "E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân."
    )
    cum = tach_su_kien(de)

    assert len(cum) == 3
    assert cum[0].startswith("Khoảnh khắc đầu tiên xuất hiện")
    assert "kẻng đồng" in cum[2]
    # Dòng bối cảnh KHÔNG được thành một mốc
    assert not any("Đoạn video bắt đầu" in c for c in cum)


def test_tach_su_kien_chiu_duoc_ca_hai_dinh_dang():
    from aic2026.trake_align import tach_su_kien

    assert tach_su_kien("E1: nhấc bánh\nE2: đặt bánh\nE3: đậy nắp") == [
        "nhấc bánh", "đặt bánh", "đậy nắp"
    ]
    assert tach_su_kien("Các mốc: (1) nhấc bánh, (2) đặt bánh") == [
        "nhấc bánh", "đặt bánh"
    ]


def test_su_kien_ngan_khong_bi_xoa():
    """Bộ lọc len>=2 của bản đầu xoá sạch sự kiện một ký tự rồi rơi về nhánh
    dự phòng trả nguyên cả câu lặp N lần — N mốc giống hệt nhau."""
    from aic2026.trake_align import tach_su_kien

    assert tach_su_kien("Các khoảnh khắc: (1) A, (2) B", so_moc=3) == ["A", "B", "B"]
