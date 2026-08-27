"""
Việc 11 — đo đường sinh đáp án Q&A.

Q&A đang 0,2167 trên bộ dev, nhưng con số đó đo bằng ĐÁP ÁN ĐÚNG ĐIỀN SẴN —
chỉ đo tầng TÌM ẢNH. Tầng sinh đáp án chưa từng chạy. Không tách hai tầng thì
không biết nên sửa chỗ nào.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.fixture
def cham(monkeypatch):
    """cham() với so khớp ngữ nghĩa thay bằng so chuỗi — không cần tải mô hình."""
    import aic2026.eval.answer_match as am
    import aic2026.eval.scorer as sc

    khop = lambda a, b: str(a).strip().lower() == str(b).strip().lower()  # noqa: E731
    monkeypatch.setattr(am, "is_semantic_match", khop)
    monkeypatch.setattr(sc, "is_semantic_match", khop)

    from scripts.do_qa import cham as _cham

    return _cham


_Q = {
    "video_id": "L21_V001",
    "frame_start": 100,
    "frame_end": 200,
    "cau_tra_loi": "5",
}


def test_tach_duoc_hai_tang(cham):
    """Ba tình huống phải cho ba kết quả KHÁC NHAU, không thì không tách được
    lỗi tầng tìm ảnh với lỗi tầng đọc đáp án."""
    dung = [{"video_id": "L21_V001", "frame_id": 150, "answer": "5"}]
    sai_dap_an = [{"video_id": "L21_V001", "frame_id": 150, "answer": "9"}]
    sai_anh = [{"video_id": "L21_V001", "frame_id": 9999, "answer": "5"}]

    assert cham(_Q, dung) > 0
    assert cham(_Q, sai_dap_an) == 0      # tầng đọc đáp án làm mất
    assert cham(_Q, sai_anh) == 0         # tầng tìm ảnh làm mất


def test_o_dap_an_rong_bi_cham_0(cham):
    """Vòng p1 một ô trống làm BTC loại nguyên tệp — điểm phải phản ánh điều đó."""
    assert cham(_Q, [{"video_id": "L21_V001", "frame_id": 150, "answer": ""}]) == 0


def test_chi_doc_cau_hoi_dap(tmp_path):
    from scripts.do_qa import doc_cau_qa

    tep = tmp_path / "dev.jsonl"
    tep.write_text(
        "\n".join(
            json.dumps(x, ensure_ascii=False)
            for x in [
                {"id": "1", "loai_truy_van": "hoi_dap", "cau_hoi": "a",
                 "video_id": "V", "frame_start": 1, "frame_end": 2,
                 "cau_tra_loi": "x"},
                {"id": "2", "loai_truy_van": "mo_ta", "cau_hoi": "b",
                 "video_id": "V", "frame_start": 1, "frame_end": 2},
                {"id": "3", "loai_truy_van": "chuoi_su_kien", "cau_hoi": "c",
                 "video_id": "V", "cac_giai_doan": []},
            ]
        ),
        encoding="utf-8",
    )
    assert [q["id"] for q in doc_cau_qa(tep)] == ["1"]


def test_canh_bao_thieu_anh_va_du_phong():
    """Máy thiếu ảnh thì mọi câu rơi về đáp án dự phòng, và con số nói về
    SHARD chứ không nói về thuật toán."""
    ma = (_ROOT / "scripts/do_qa.py").read_text(encoding="utf-8")
    than = "\n".join(ma.split('"""')[::2])

    assert "loc_dev_co_anh" in than, "không chỉ đường lọc bộ dev"
    assert "du_phong" in than, "không đếm câu rơi về đáp án dự phòng"
    assert "CỠ MẪU" in than, "không cảnh báo cỡ mẫu nhỏ"


def test_qa_answer_duoc_goi_that():
    """qa_answer.py từng chưa được gọi ở BẤT CỨ ĐÂU ngoài một thông báo lỗi."""
    ma = (_ROOT / "scripts/do_qa.py").read_text(encoding="utf-8")
    than = "\n".join(ma.split('"""')[::2])

    assert "from aic2026.qa_answer import" in than
    assert "tra_loi_theo_hang(" in than


def test_do_qa_dung_cau_hinh_production_thay_vi_ghi_cung_nhanh_cu():
    """Phép đo Task 11 phải đọc cùng rrf_weights.yaml với bộ sinh bài nộp.

    Bản cũ ghi cứng CLIP B/32 + OCR + ASR và luôn bật Marian, trong khi
    production đã chuyển sang CLIP-L. Test này ngăn hai đường lại lệch nhau.
    """
    ma = (_ROOT / "scripts/do_qa.py").read_text(encoding="utf-8")
    than = "\n".join(ma.split('"""')[::2])

    assert "trong_so_theo_dang(\"qa\")" in than
    assert "dung_clip_l=dung_clip_l" in than
    assert 'default="nguyen_ban"' in than
    assert "nguon_clip_l=nguon_clip_l" in than


# ---------------------------------------------------------------------------
# Tầng sinh đáp án — chỗ làm mất 100% điểm ở lần đo đầu
# ---------------------------------------------------------------------------

def test_cat_ngan_dap_an():
    """Đáp án đúng bộ dev dài 1-4 chữ. Bản đầu trả cả câu ASR 96 ký tự —
    câu 14 tìm ảnh được 1,000 rồi mất sạch ở tầng này."""
    from aic2026.qa_answer import DAI_TOI_DA, cat_ngan

    dai = ("Thì để cái máy đó bên mình sẽ phải nhập khẩu nguyên chiếc về "
           "để mà làm lên những chiếc túi này.")
    ra = cat_ngan(dai)
    assert len(ra) <= DAI_TOI_DA
    assert not ra.endswith(" ")          # cắt tại ranh giới từ
    assert cat_ngan("Jacquemus") == "Jacquemus"


def test_dap_an_luon_bi_cat_ngan_du_di_duong_nao():
    from aic2026.qa_answer import DAI_TOI_DA, DapAn

    d = DapAn("x" * 200, 0.9, "asr")
    assert len(d.van_ban) <= DAI_TOI_DA


@pytest.mark.parametrize(
    "cau_hoi,ung_vien,mong_doi",
    [
        ("Có bao nhiêu tiết mục?",
         ["Sẵn sàng trên bài tiết mục mũi rồng truyền thống", "5 tiết mục"], "5"),
        ("Sự kiện diễn ra năm nào?",
         ["Chương trình được tổ chức long trọng", "2024", "hôm nay"], "2024"),
        ("Dòng chữ trên biển là gì?", ["Online", "EV", "Cây từ vựng"], "Cây từ vựng"),
        ("Áo màu gì?", ["HỂ THAO", "áo trắng của cầu thủ"], "trắng"),
    ],
)
def test_chon_cum_theo_dang_cau_hoi(cau_hoi, ung_vien, mong_doi):
    """Bản đầu lấy hộp OCR conf cao nhất và câu ASR gần mốc nhất — KHÔNG đọc
    câu hỏi. Ra 'EV', 'Online', 'HỂ THAO', 'công'."""
    from aic2026.qa_answer import chon_cum_tot_nhat

    ra = chon_cum_tot_nhat(cau_hoi, ung_vien)
    assert ra and mong_doi.lower() in ra.lower(), f"chọn {ra!r}"


def test_loai_logo_dai():
    """Logo đài đóng trên MỌI khung nên luôn là hộp OCR rõ nhất, và gần như
    không bao giờ là đáp án."""
    from aic2026.qa_answer import chon_cum_tot_nhat

    ra = chon_cum_tot_nhat("Thương hiệu túi là gì?", ["VTV24", "Jacquemus"])
    assert ra == "Jacquemus"


def test_cau_hoi_ten_thuong_hieu_nhan_dung_dang():
    from aic2026.qa_answer import CHU_TREN_HINH, loai_cau_hoi

    for c in ["Thương hiệu túi xách là gì?", "Tên chương trình là gì?",
              "Nhãn hiệu nào xuất hiện?"]:
        assert loai_cau_hoi(c) == CHU_TREN_HINH, c


def test_cau_ten_xa_duoc_nhan_la_dia_diem_khong_phai_chu_tren_hinh():
    from aic2026.qa_answer import DIA_DIEM, chon_cum_tot_nhat, loai_cau_hoi

    cau = "Câu lạc bộ trao quà tại một xã thuộc Khánh Hòa. Xã này có tên là gì?"
    assert loai_cau_hoi(cau) == DIA_DIEM
    assert chon_cum_tot_nhat(cau, ["chương trình từ thiện", "xã Giang Ly"]) == "xã Giang Ly"


@pytest.mark.parametrize(
    "cau_hoi,nguon_mong_doi",
    [
        ("Áo của người phụ nữ màu gì?", "vlm"),
        ("Tiêu đề trên áp phích là gì?", "ocr"),
        ("Xã này có tên là gì?", "ocr"),
        ("Em quyên góp để ủng hộ ai?", "asr"),
    ],
)
def test_tra_loi_dinh_tuyen_nguon_theo_dang_cau(
    monkeypatch, cau_hoi, nguon_mong_doi
):
    import aic2026.qa_answer as qa

    da_goi = []

    def gia(nguon):
        def _f(*args, **kwargs):
            da_goi.append(nguon)
            return qa.DapAn(nguon, 0.8, nguon)
        return _f

    monkeypatch.setattr(qa, "tra_loi_bang_ocr", gia("ocr"))
    monkeypatch.setattr(qa, "tra_loi_bang_asr", gia("asr"))
    monkeypatch.setattr(qa, "tra_loi_bang_vlm", gia("vlm"))

    class Hit:
        video_id = "V"
        n = 1
        frame_idx = 100
        pts_time = 4.0

    ra = qa.tra_loi(cau_hoi, [Hit()], bo_doc_anh=object(), dung_vlm=True)
    assert ra.nguon == nguon_mong_doi
    assert da_goi == [nguon_mong_doi]


def test_ocr_cau_tieu_de_doc_them_hai_keyframe_lan_can(monkeypatch):
    import aic2026.qa_answer as qa

    monkeypatch.setattr(qa, "_doc_ocr_cua_video", lambda video: {
        89: ({"text": "CÂY TỪ VỰNG", "conf": 0.9},),
        91: ({"text": "VTV", "conf": 0.99},),
    })
    d = qa.tra_loi_bang_ocr("Tiêu đề áp phích là gì?", "V", 91)
    assert d is not None and "TỪ VỰNG" in d.van_ban.upper()


def test_ocr_dia_diem_doc_bien_ten_o_noi_khac_trong_video(monkeypatch):
    import aic2026.qa_answer as qa

    monkeypatch.setattr(qa, "_doc_ocr_cua_video", lambda video: {
        9: ({"text": "FANA", "conf": 0.9},),
        34: ({"text": "Xã Giang Ly huyện Khánh Vĩnh", "conf": 0.8},),
    })
    d = qa.tra_loi_bang_ocr("Xã này có tên là gì?", "V", 9)
    assert d is not None and d.van_ban == "Xã Giang Ly"


def test_asr_ung_ho_ai_lay_ve_sau_dong_tu(tmp_path, monkeypatch):
    import aic2026.paths as paths
    import aic2026.qa_answer as qa

    tep = tmp_path / "V.jsonl"
    tep.write_text(json.dumps({
        "start": 39.53, "end": 45.29,
        "text": "Em gom tiền trong hai năm để ủng hộ các bạn thiếu nhi miền Bắc.",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(paths, "asr_file", lambda video: tep)

    d = qa.tra_loi_bang_asr("Em quyên góp để ủng hộ ai?", "V", 42.0)
    assert d is not None
    assert d.van_ban == "ủng hộ các bạn thiếu nhi miền Bắc"


def test_no_lan_can_giu_hit_goc_truoc_va_chen_n_tru_hai(monkeypatch):
    import aic2026.frame_map as fm
    import aic2026.qa_answer as qa

    monkeypatch.setattr(fm, "lookup", lambda video, n: (n * 10, float(n)))

    class Hit:
        video_id, n, score = "V", 91, 1.0
        frame_idx, pts_time, source = 6193, 1.0, "clip_l"

    ra = qa.mo_rong_hit_lan_can([Hit()], so_hit_mo_rong=1)
    assert [h.n for h in ra[:5]] == [91, 89, 90, 92, 93]


def test_tra_loi_theo_hang_giu_dap_an_gan_voi_tung_khung(monkeypatch):
    import aic2026.qa_answer as qa

    monkeypatch.setattr(qa, "mo_rong_hit_lan_can", lambda hits, **kw: list(hits))
    monkeypatch.setattr(
        qa, "tra_loi",
        lambda cau, hits, **kw: qa.DapAn(f"dap-{hits[0].frame_idx}", 1.0, "ocr"),
    )

    class Hit:
        def __init__(self, f):
            self.video_id, self.n, self.frame_idx = "V", f, f
            self.pts_time, self.score, self.source = float(f), 1.0, "x"

    cap = qa.tra_loi_theo_hang("x", [Hit(10), Hit(20)])
    assert [(h.frame_idx, d.van_ban) for h, d in cap] == [(10, "dap-10"), (20, "dap-20")]


def test_tra_loi_theo_hang_co_the_tat_mo_rong_lan_can(monkeypatch):
    import aic2026.qa_answer as qa

    class Hit:
        def __init__(self, frame_idx):
            self.video_id = "V"
            self.frame_idx = frame_idx
            self.n = frame_idx
            self.pts_time = float(frame_idx)
            self.score = 1.0
            self.source = "test"

    monkeypatch.setattr(
        qa, "mo_rong_hit_lan_can",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("không được gọi")),
    )
    monkeypatch.setattr(
        qa, "tra_loi",
        lambda *a, **k: qa.DapAn("x", 1.0, "test"),
    )

    cap = qa.tra_loi_theo_hang(
        "x", [Hit(10), Hit(20)], mo_rong_lan_can=False,
    )
    assert [h.frame_idx for h, _ in cap] == [10, 20]


# ---------------------------------------------------------------------------
# Chiến thuật nộp: rải biến thể đáp án theo THỨ HẠNG
# ---------------------------------------------------------------------------

def test_bien_the_so_va_don_vi():
    """Vòng p1 câu 3 rải bằng tay: '2484' / '2.484' / '2484 kg'."""
    from aic2026.qa_answer import bien_the_dap_an

    bt = bien_the_dap_an("2484 kg")
    assert "2484 kg" in bt and "2484" in bt
    assert any("." in x or "," in x for x in bt), "thiếu dấu phân cách hàng nghìn"


def test_bien_the_bo_tien_to_loai_tu():
    """'đèo Tà Pứa' -> 'Tà Pứa' — biến thể ĐỔI NGHĨA cụm, giá trị cao nhất.

    Biến thể chỉ khác hoa/thường KHÔNG được sinh: dedup_key() hạ chữ thường
    trước khi so nên chúng bị bỏ ở bước ghi, chỉ tổ chiếm chỗ.
    """
    from aic2026.qa_answer import bien_the_dap_an

    bt = bien_the_dap_an("đèo Tà Pứa")
    assert bt[0] == "đèo Tà Pứa"
    assert "Tà Pứa" in bt
    assert not any(
        x != y and x.lower() == y.lower() for x in bt for y in bt
    ), f"còn biến thể chỉ khác hoa/thường: {bt}"


def test_dap_an_rong_van_ra_bien_the():
    from aic2026.qa_answer import DAP_AN_DU_PHONG, bien_the_dap_an

    assert bien_the_dap_an("") == [DAP_AN_DU_PHONG]


def test_khung_tot_nhat_duoc_ghep_MOI_bien_the():
    """Vòng p1 rải xoay vòng nên khung ĐÚNG chỉ được một biến thể — 1/3 cơ hội.

    Điểm cuối là trung bình R@1, R@5, R@20, R@50, R@100 nên hạng đầu đáng giá
    hơn hẳn. Đặt khung tốt nhất × MỌI biến thể ở hạng 1-3 chỉ tốn 3 suất/100.
    """
    from aic2026.qa_answer import bien_the_dap_an, rai_theo_hang

    class _H:
        def __init__(self, f):
            self.video_id, self.frame_idx = "V", f

    hits = [_H(1000 + i) for i in range(100)]
    dong = rai_theo_hang(hits, "2484 kg")

    bt = bien_the_dap_an("2484 kg")
    dau_bang = [a for h, a in dong if h.frame_idx == 1000]
    assert set(dau_bang) == set(bt), "khung hạng 1 phải có ĐỦ mọi biến thể"

    # và chúng phải nằm ở đầu, không rải rác
    assert all(h.frame_idx == 1000 for h, _ in dong[: len(bt)])


def test_rai_khong_trung_dong_va_khong_vuot_han_muc():
    from aic2026.qa_answer import rai_theo_hang

    class _H:
        def __init__(self, f):
            self.video_id, self.frame_idx = "V", f

    dong = rai_theo_hang([_H(1000 + i) for i in range(100)], "6", so_dong=100)
    assert len(dong) <= 100
    khoa = [(h.video_id, h.frame_idx, a) for h, a in dong]
    assert len(khoa) == len(set(khoa)), "có dòng trùng — phí suất"


def test_bo_sinh_tep_nop_giu_dap_an_gan_voi_tung_khung():
    ma = (_ROOT / "scripts/tao_bo_nop.py").read_text(encoding="utf-8")
    than = "\n".join(ma.split('"""')[::2])
    assert "tra_loi_theo_hang" in than
    assert "for h, d in cap" in than
