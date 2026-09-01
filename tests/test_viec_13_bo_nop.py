"""
Việc 13 — sinh tệp nộp cho MỌI truy vấn BTC ra.

Vòng p1: BTC ra 25 truy vấn, nhóm nộp 22. Thiếu câu 8, 14, 25 — mỗi câu 0 điểm
CHẮC CHẮN, tổng 12% điểm mất trắng TRƯỚC KHI thi đấu. Câu 8 và 14 lại có nội
dung y hệt nhau, tức bỏ không hai suất chỉ vì thấy đề lặp lại.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.tao_bo_nop import (  # noqa: E402
    MAU_TEN_DE,
    SO_DONG,
    doc_thu_muc_de,
    dong_doan,
    loc_cua_so_cau_tra_loi,
    so_moc_trake,
    tra_mot_de,
    van_tay,
)


@pytest.fixture(autouse=True)
def _frame_map_gia(monkeypatch):
    """dong_doan() đọc frame_map thật. conftest đặt DATA_ROOT tạm không có tệp
    đó, nên giả lập một bảng nhỏ — phép kiểm ở đây là VỀ SỐ DÒNG, không phải
    về nội dung frame_map."""
    import pandas as pd

    import aic2026.frame_map as fm

    bang = pd.DataFrame(
        {
            "video_id": [f"L21_V{i // 50 + 1:03d}" for i in range(500)],
            "n": [i % 50 + 1 for i in range(500)],
            "pts_time": [i * 2.6 for i in range(500)],
            "fps": [25.0] * 500,
            "frame_idx": [int(i * 2.6 * 25) for i in range(500)],
        }
    )
    monkeypatch.setattr(fm, "load_frame_map", lambda *a, **k: bang)


class _Hit:
    def __init__(self, v, f):
        self.video_id, self.frame_idx = v, f
        self.n, self.pts_time, self.score, self.source = 1, f / 25, 1.0, "clip"


def _de(thu_muc: Path, ten: str, noi_dung: str) -> None:
    (thu_muc / f"{ten}.txt").write_text(noi_dung, encoding="utf-8")


def test_doc_du_moi_de(tmp_path):
    for i, dang in [(1, "kis"), (3, "qa"), (16, "trake"), (25, "kis")]:
        _de(tmp_path, f"query-p1-{i}-{dang}", f"đề số {i}")
    _de(tmp_path, "ghi-chu", "không phải đề")

    de = doc_thu_muc_de(tmp_path)
    assert [d["so"] for d in de] == [1, 3, 16, 25]      # đã sắp theo số
    assert [d["dang"] for d in de] == ["kis", "qa", "trake", "kis"]


def test_ten_tep_de_khong_dung_khuon_bi_bo_qua(tmp_path):
    _de(tmp_path, "query-p1-1-kis", "a")
    _de(tmp_path, "query_p1_2_kis", "b")        # gạch dưới, sai khuôn
    assert len(doc_thu_muc_de(tmp_path)) == 1


def test_phat_hien_de_trung_noi_dung():
    """Câu 8 và 14 vòng p1 có nội dung y hệt nhau."""
    a = "Người đầu bếp lần lượt đặt các miếng nguyên liệu"
    assert van_tay(a) == van_tay(f"  {a.upper()}  ")   # bỏ hoa/thường, khoảng trắng
    assert van_tay(a) != van_tay(a + " thêm chữ")


def test_dem_moc_trake_tu_dinh_dang_BTC():
    de = "Đoạn video mở đầu bằng cảnh chợ.\nE1 khoảnh khắc A\nE2 khoảnh khắc B"
    assert so_moc_trake(de) == 2

    de5 = "nền\n" + "\n".join(f"E{i} mốc {i}" for i in range(1, 6))
    assert so_moc_trake(de5) == 5

    # không đánh dấu gì thì đoán 3 — thà thừa mốc còn hơn thiếu
    assert so_moc_trake("một chuỗi hành động nào đó") == 3


# ---------------------------------------------------------------------------
# Luật cứng: mọi tình huống hỏng vẫn phải ra 100 dòng hợp lệ
# ---------------------------------------------------------------------------

def test_mach_nem_loi_van_ra_du_dong():
    def hong(cau, k):
        raise RuntimeError("FAISS chưa dựng")

    ra, ghi_chu = tra_mot_de({"dang": "kis", "cau_hoi": "x"}, hong)
    assert len(ra) == SO_DONG
    assert "lỗi" in ghi_chu


def test_mach_rong_van_ra_du_dong():
    ra, ghi_chu = tra_mot_de({"dang": "kis", "cau_hoi": "x"}, lambda c, k: [])
    assert len(ra) == SO_DONG
    assert "không trả về gì" in ghi_chu


def test_kis_loc_10_giay_truoc_khi_cat_100():
    """Ba hit gần nhau/video phải còn đúng hit tốt nhất, nhưng vẫn đủ 100."""
    hits = [
        _Hit(f"L21_V{i:03d}", j * 25)
        for i in range(SO_DONG)
        for j in range(3)
    ]
    ra, ghi_chu = tra_mot_de(
        {"dang": "kis", "cau_hoi": "x"}, lambda c, k: hits,
    )

    assert len(ra) == SO_DONG
    assert len({a.video_id for a in ra}) == SO_DONG
    assert "lọc 10.0s 300->100" in ghi_chu


def test_chot_cuoi_loc_ca_dong_qa_va_frame_khong_ton_tai():
    from aic2026.submit import Answer

    answers = [
        Answer("V1", [0], answer="a"),
        Answer("V1", [1], answer="b"),       # cách dòng đầu 9,9 giây
        Answer("V1", [2], answer="c"),       # cách dòng đầu đúng 10 giây
        Answer("V2", [999], answer="d"),     # không có trong frame_map
    ]
    tra = {("V1", 0): 0.0, ("V1", 1): 9.9, ("V1", 2): 10.0}

    ra, bo_gan, bo_khong_tra = loc_cua_so_cau_tra_loi(
        answers, "qa", tra_nguoc=tra, cua_so=10.0,
    )

    assert [a.frame_ids[0] for a in ra] == [0, 2]
    assert bo_gan == 1
    assert bo_khong_tra == 1


def test_chot_cuoi_khong_ap_10_giay_cho_trake():
    from aic2026.submit import Answer

    answers = [Answer("V1", [1, 2, 3]), Answer("V1", [2, 3, 4])]
    ra, bo_gan, bo_khong_tra = loc_cua_so_cau_tra_loi(answers, "trake")
    assert ra == answers
    assert (bo_gan, bo_khong_tra) == (0, 0)


def test_dong_doan_trake_moi_moc_deu_thuoc_cung_video():
    import aic2026.frame_map as fm

    bang = fm.load_frame_map()
    hop_le = {
        (str(video_id), int(frame_idx))
        for video_id, frame_idx in zip(bang["video_id"], bang["frame_idx"])
    }
    ra = dong_doan(so_moc=3)

    assert len(ra) == SO_DONG
    assert all(len(a.frame_ids) == 3 for a in ra)
    assert all(
        (a.video_id, frame_idx) in hop_le
        for a in ra
        for frame_idx in a.frame_ids
    )


def test_validator_qa_khong_nham_dap_an_so_la_frame():
    from scripts.check_submission import (
        cot_frame_cua_dong,
        loai_truy_van_tu_ten,
    )

    assert loai_truy_van_tu_ten(Path("query-p1-3-qa.csv")) == "qa"
    assert cot_frame_cua_dong(["L21_V001", "1234", "5"], "qa") == ["1234"]
    assert cot_frame_cua_dong(["L21_V001", "10", "20", "30"], "trake") == [
        "10", "20", "30"
    ]


def test_qa_khong_bao_gio_de_o_dap_an_rong(monkeypatch):
    """Vòng p1 một ô trống làm BTC loại NGUYÊN TỆP 100 dòng."""
    import aic2026.qa_answer as qa

    monkeypatch.setattr(
        qa, "tra_loi_theo_hang",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("VLM chết")),
    )
    ra, _ = tra_mot_de(
        {"dang": "qa", "cau_hoi": "Có bao nhiêu người?"},
        lambda c, k: [_Hit("L21_V001", i * 100) for i in range(SO_DONG)],
    )
    # Nguồn giả đặt 100 hit trong cùng một video, cách nhau 4 giây. Sau cổng
    # 10 giây, tra_mot_de() chỉ còn 34 ứng viên; main() mới chịu trách nhiệm
    # lấp đủ 100 dòng bằng frame dự phòng không xung đột.
    assert ra
    assert len(ra) < SO_DONG
    assert all(str(a.answer or "").strip() for a in ra)


def test_qa_production_giu_dap_an_cua_tung_khung(monkeypatch):
    import aic2026.qa_answer as qa

    hits = [_Hit("L21_V001", i * 100) for i in range(SO_DONG)]
    monkeypatch.setattr(
        qa,
        "tra_loi_theo_hang",
        lambda *a, **k: [
            (h, qa.DapAn(f"dap-{h.frame_idx}", 1.0, "ocr")) for h in hits
        ],
    )

    ra, ghi_chu = tra_mot_de(
        {"dang": "qa", "cau_hoi": "Dòng chữ là gì?"},
        lambda c, k: hits,
    )
    assert len(ra) == SO_DONG
    assert [a.answer for a in ra[:3]] == ["dap-0", "dap-100", "dap-200"]
    assert "theo từng khung" in ghi_chu


def test_trake_moc_luon_tang_dan(monkeypatch):
    """Vòng p1 nộp 36/100 dòng có mốc KHÔNG tăng dần.

    Sự kiện là chuỗi thời gian nên những dòng đó chắc chắn sai — chỗ trống
    lãng phí.

    Test này chỉ kiểm logic sinh các dòng fallback, không chạy CLIP thật.
    """
    import aic2026.trake_align as ta

    def ghep_that_bai(*args, **kwargs):
        raise RuntimeError("mock: không chạy CLIP trong unit test")

    monkeypatch.setattr(ta, "ghep", ghep_that_bai)

    de = {"dang": "trake", "cau_hoi": "nền\nE1 a\nE2 b\nE3 c"}
    ra, _ = tra_mot_de(
        de,
        lambda c, k: [_Hit("L21_V001", i * 100) for i in range(SO_DONG)],
    )

    assert len(ra) == SO_DONG
    for a in ra:
        assert len(a.frame_ids) == 3
        assert a.frame_ids == sorted(a.frame_ids)
        assert len(set(a.frame_ids)) == 3


def test_trake_production_dat_ket_qua_ghep_len_hang_dau(monkeypatch):
    import aic2026.trake_align as ta

    class Ket:
        video_id = "L21_V009"
        frame_ids = [101, 205, 309]
        du_day = True

    monkeypatch.setattr(ta, "ghep", lambda *a, **k: Ket())

    hits = [_Hit("L21_V001", i * 100) for i in range(SO_DONG)]
    ra, ghi_chu = tra_mot_de(
        {"dang": "trake", "cau_hoi": "E1 a\nE2 b\nE3 c"},
        lambda c, k: hits,
    )
    assert ra[0].video_id == "L21_V009"
    assert ra[0].frame_ids == [101, 205, 309]
    assert "dense" in ghi_chu


def test_ten_de_tach_dung():
    k = MAU_TEN_DE.match("query-p1-16-trake")
    assert (k["dot"], int(k["so"]), k["dang"]) == ("p1", 16, "trake")
    assert MAU_TEN_DE.match("query-p1-16-abc") is None


# ---------------------------------------------------------------------------
# Chốt chặn: mã viết ra phải ĐƯỢC GỌI, không thì cờ bật lên là phép đo giả
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mo_dun,dau_hieu,viec",
    [
        ("aic2026.rerank", "xep_lai", "Việc 4 (rerank)"),
        ("aic2026.query_expand", "tim_ung_vien_clip_mo_rong", "Việc 5 (mở rộng)"),
        ("aic2026.index.objects_index", "khung_chua", "Việc 3 (bộ lọc)"),
        ("aic2026.enrich.caption", "CaptionSearchIndex", "Việc 10 (caption)"),
    ],
)
def test_moi_viec_deu_duoc_noi_vao_phep_do(mo_dun, dau_hieu, viec):
    """Mã viết xong mà không ai gọi thì bật cờ lên vẫn ra số CŨ.

    Đã cắn ba lần: query_expand viết xong nhưng do_trong_so_rrf gọi thẳng
    tim_ung_vien_clip, nên đặt $env:AIC_NGUON_MO_RONG rồi chạy lại ra SỐ GIỐNG
    HỆT. rerank.py và qa_answer.py thì chưa từng được gọi ở đâu cả.
    """
    ma = (_ROOT / "scripts/do_trong_so_rrf.py").read_text(encoding="utf-8")
    than = "\n".join(ma.split('"""')[::2])

    assert mo_dun in than, f"{viec}: {mo_dun} không được nạp trong phép đo"
    assert dau_hieu in than, f"{viec}: {dau_hieu} không được gọi"


def test_moi_cach_bat_deu_co_co_rieng():
    """Mỗi việc một cờ độc lập — không tách được thì không đo riêng được."""
    ma = (_ROOT / "scripts/do_trong_so_rrf.py").read_text(encoding="utf-8")
    for co in ("--nguon-mo-rong", "--loc-vat-the", "--rerank", "--khong-object"):
        assert co in ma, f"thiếu cờ {co}"


def test_rerank_dung_khi_khong_dung_duoc():
    """Rerank hỏng mà chạy tiếp thì ra bảng số giống hệt lượt không rerank —
    người đọc kết luận 'rerank vô dụng' trong khi nó chưa hề chạy."""
    ma = (_ROOT / "scripts/do_trong_so_rrf.py").read_text(encoding="utf-8")
    i = ma.index("RERANK KHÔNG DÙNG ĐƯỢC")
    assert "return 1" in ma[i: i + 600], "phải dừng, không được chạy tiếp"


def test_canh_bao_khi_rerank_thieu_anh():
    """Rerank đẩy ứng viên THIẾU ẢNH xuống cuối, bất kể đúng hay sai.

    Máy chỉ có ảnh cho 8/32 câu dev thì phép đo trở thành 'máy này có ảnh
    không', không phải 'rerank tốt hay xấu'. Đã cắn thật: nhánh CLIP tụt
    0,3667 -> 0,2833 và con số đó vô nghĩa.
    """
    ma = (_ROOT / "scripts/do_trong_so_rrf.py").read_text(encoding="utf-8")
    than = "\n".join(ma.split('"""')[::2])

    assert "THONG_KE_RERANK" in than, "không đếm ứng viên thiếu ảnh"
    assert "so_thieu_anh" in than, "không lấy số thiếu ảnh từ báo cáo rerank"
    assert "ĐỪNG ghi vào sổ điểm" in ma, "không cảnh báo người đọc"


def test_co_cong_cu_loc_dev_theo_shard():
    tep = _ROOT / "scripts/loc_dev_co_anh.py"
    assert tep.exists(), "thiếu công cụ lọc bộ dev xuống câu có ảnh"

    ma = tep.read_text(encoding="utf-8")
    assert "CO_MAU_TOI_THIEU" in ma, "phải chặn khi cỡ mẫu quá nhỏ"


def test_bien_the_khong_bi_khu_trung_lam_mat_dong():
    """SubmissionBudget.dedup_key() hạ chữ thường trước khi so, nên
    'Đèo Tà Pứa' và 'đèo tà pứa' là CÙNG một dòng và bị bỏ.

    Sinh biến thể như vậy là chiếm chỗ trong danh sách rồi mất trắng ở bước
    ghi — đã đo: ba tệp p1 chỉ ra 98/100 dòng.
    """
    from aic2026.qa_answer import bien_the_dap_an
    from aic2026.submit.formatter import Answer

    for goc in ["đèo Tà Pứa", "2484 kg", "70% là đất", "Jacquemus"]:
        bt = bien_the_dap_an(goc)
        khoa = {Answer("V", [1], answer=x).dedup_key() for x in bt}
        assert len(khoa) == len(bt), f"{goc!r}: biến thể {bt} bị khử trùng"


def test_lap_du_100_dong_du_mach_tra_it():
    """Đếm theo số ngân sách THẬT SỰ NHẬN, không đếm danh sách đưa vào —
    ngân sách khử trùng nên một phần dòng có thể bị bỏ mà không ai biết."""
    ma = (_ROOT / "scripts/tao_bo_nop.py").read_text(encoding="utf-8")
    than = "\n".join(ma.split('"""')[::2])

    i = than.index("if len(ngan_sach) < SO_DONG:")
    j = than.index("ngan_sach = SubmissionBudget")
    assert i > j, "phải lấp SAU khi thêm vào ngân sách, không phải trước"
    assert "dong_doan(so_moc, so_dong=so_du_phong)" in than[i: i + 1800]
