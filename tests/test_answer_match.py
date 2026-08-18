"""
Kiểm tra nhánh HỎI–ĐÁP của Task 12 — phần so khớp answer.

Vì sao có tệp này: docstring của tests/test_scoring.py liệt kê điều kiện số 2
là "r_score_qa khớp đúng ví dụ Q&A của BTC", nhưng trong tệp đó KHÔNG có test
nào cho Q&A. Hỏi–đáp chiếm 12/32 câu bộ dev mà toàn bộ answer_match.py không
có một dòng test nào che.

NGUYÊN TẮC CỦA TỆP NÀY: KHÔNG test nào được tải mô hình thật.

`_get_model()` tải paraphrase-multilingual-MiniLM-L12-v2 từ HuggingFace ở lần
gọi đầu tiên. Nếu để test thật gọi vào đó thì:
  - máy chưa tải mô hình sẽ treo vài phút ở lần chạy pytest đầu tiên
  - máy không có mạng sẽ ĐỎ TEST vì lý do chẳng liên quan gì tới đúng/sai
Nên mọi test chạm tới nhánh embedding đều thay `_get_model` bằng mô hình giả
có vector định trước. Test chạy trong mili giây, không cần mạng.

Chạy:

    pytest tests/test_answer_match.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.aic2026.eval import answer_match
from src.aic2026.eval.answer_match import is_semantic_match, normalize_text
from src.aic2026.eval.scorer import r_score_qa


# ---------------------------------------------------------------------------
# Đồ nghề: mô hình giả + dọn cache giữa các test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def don_cache_mo_hinh():
    """
    `answer_match._model_cache` là biến toàn cục. Không dọn thì test chạy
    trước để lại mô hình giả cho test chạy sau — đỏ/xanh phụ thuộc THỨ TỰ
    chạy, loại lỗi mất nhiều giờ nhất để tìm ra.
    """
    answer_match._model_cache = None
    yield
    answer_match._model_cache = None


def vector_lech_goc(do_tuong_dong: float) -> np.ndarray:
    """Vector hợp với [1, 0] một góc có cosine đúng bằng `do_tuong_dong`."""
    return np.array([do_tuong_dong, math.sqrt(max(0.0, 1.0 - do_tuong_dong**2))])


class MoHinhGia:
    """
    Thay cho SentenceTransformer. `encode([a, b])` trả về hai vector có
    cosine ĐÚNG BẰNG con số mình đặt, nên test kiểm được đúng chỗ so ngưỡng
    chứ không phụ thuộc mô hình thật hiểu tiếng Việt tới đâu.
    """

    def __init__(self, do_tuong_dong: float):
        self.do_tuong_dong = do_tuong_dong
        self.so_lan_encode = 0

    def encode(self, cau_list):
        self.so_lan_encode += 1
        return [np.array([1.0, 0.0]), vector_lech_goc(self.do_tuong_dong)]


class MoHinhNo(Exception):
    """Ném ra khi có test nạp mô hình ở nhánh lẽ ra không được nạp."""


def cam_nap_mo_hinh(monkeypatch):
    """Gài bẫy: nạp mô hình ở đây là sai, phải nổ ngay để test bắt được."""

    def _no():
        raise MoHinhNo("Nhánh này KHÔNG được phép nạp mô hình embedding")

    monkeypatch.setattr(answer_match, "_get_model", _no)


def gan_mo_hinh_gia(monkeypatch, do_tuong_dong: float) -> MoHinhGia:
    mo_hinh = MoHinhGia(do_tuong_dong)
    monkeypatch.setattr(answer_match, "_get_model", lambda: mo_hinh)
    return mo_hinh


# ---------------------------------------------------------------------------
# 1. normalize_text
# ---------------------------------------------------------------------------


def test_normalize_bo_hoa_dau_cau_khoang_trang():
    """normalize_text: hạ chữ thường, bỏ dấu câu, gộp khoảng trắng thừa."""
    assert normalize_text("  Màu XANH!!  ") == "màu xanh"
    assert normalize_text("Năm (5) người.") == "năm 5 người"


def test_normalize_chuoi_rong_va_none():
    """normalize_text: None và chuỗi rỗng đều ra chuỗi rỗng, không nổ."""
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_normalize_GIU_dau_tieng_viet():
    """
    normalize_text KHÔNG được bỏ dấu tiếng Việt.

    Bỏ dấu là gộp nhầm hai từ khác nghĩa: "ma" với "má", "bàn" với "bán".
    Đây là quyết định ngược với chỗ FTS (ở đó remove_diacritics 2 là ĐÚNG,
    vì người gõ truy vấn có thể gõ thiếu dấu). Hai chỗ khác nhau có chủ ý —
    ai định "thống nhất" hai chỗ này thì đọc lại đây trước.
    """
    assert normalize_text("má") != normalize_text("ma")
    assert normalize_text("bán") != normalize_text("bàn")


def test_normalize_gop_NFC_hai_cach_go_cung_mot_chu():
    """
    normalize_text: cùng một chữ gõ kiểu tổ hợp (NFD) hay dựng sẵn (NFC)
    phải ra cùng kết quả.

    Bộ dev soạn tay trên bốn máy, gõ bằng Unikey/telex/VNI khác nhau nên
    "ề" có thể nằm ở cả hai dạng. Không chuẩn hoá NFC là chấm sai một
    answer đúng, mà nhìn hai chuỗi trên màn hình thì y hệt nhau.
    """
    dung_san = "về"                  # chữ "ề" dựng sẵn, U+1EC1
    to_hop = "v" + "e\u0302\u0300"   # e + dấu mũ + dấu huyền, ba ký tự

    assert dung_san != to_hop, "hai chuỗi phải khác nhau ở mức ký tự"
    assert normalize_text(dung_san) == normalize_text(to_hop)


# ---------------------------------------------------------------------------
# 2. is_semantic_match — nhánh exact match (KHÔNG được nạp mô hình)
# ---------------------------------------------------------------------------


def test_khop_y_het_khong_nap_mo_hinh(monkeypatch):
    """Answer y hệt sau normalize -> True NGAY, không đụng tới mô hình."""
    cam_nap_mo_hinh(monkeypatch)
    assert is_semantic_match("màu xanh", "Màu Xanh!") is True


def test_answer_rong_tra_false_khong_nap_mo_hinh(monkeypatch):
    """
    Answer rỗng -> False, và KHÔNG nạp mô hình.

    Quan trọng ở chỗ tiết kiệm: một truy vấn có tới 100 dòng nộp, nạp mô
    hình cho dòng rỗng là phí sạch.
    """
    cam_nap_mo_hinh(monkeypatch)
    assert is_semantic_match("màu xanh", "") is False


@pytest.mark.xfail(
    strict=False,
    reason=(
        "CẦN NGUYÊN QUYẾT: gt_answer rỗng + answer nộp rỗng -> hiện trả True "
        "vì normalize('') == normalize(''). Câu hỏi–đáp mà quên điền đáp án "
        "trong bộ dev sẽ tự cho điểm chính nó. Bộ dev 32 câu hiện KHÔNG có "
        "câu nào rỗng nên chưa ảnh hưởng 0.1063, nhưng lúc mở rộng lên 60–80 "
        "câu thì đây là điểm ma."
    ),
)
def test_gt_rong_khong_duoc_tu_cho_diem():
    """gt_answer rỗng thì không câu trả lời nào được coi là khớp."""
    assert is_semantic_match("", "") is False


# ---------------------------------------------------------------------------
# 3. is_semantic_match — nhánh embedding, so ngưỡng
# ---------------------------------------------------------------------------


def test_tren_nguong_thi_khop(monkeypatch):
    """Cosine 0.90 > ngưỡng 0.80 -> khớp."""
    gan_mo_hinh_gia(monkeypatch, 0.90)
    assert is_semantic_match("năm người", "5 người") is True


def test_duoi_nguong_thi_khong_khop(monkeypatch):
    """Cosine 0.50 < ngưỡng 0.80 -> không khớp."""
    gan_mo_hinh_gia(monkeypatch, 0.50)
    assert is_semantic_match("màu xanh", "cái ghế") is False


def test_bang_dung_nguong_thi_khop(monkeypatch):
    """
    Cosine ĐÚNG BẰNG ngưỡng -> khớp (điều kiện là >=, không phải >).

    Chốt luôn chiều của dấu so sánh để sau này ai chỉnh ngưỡng cũng biết
    biên nằm ở đâu.
    """
    nguong = answer_match._nguong_tuong_dong()
    gan_mo_hinh_gia(monkeypatch, nguong)
    assert is_semantic_match("màu xanh", "màu lam") is True


def test_nguong_doc_tu_settings_yaml():
    """
    Ngưỡng phải đọc từ config/settings.yaml mục cham_diem, không phải số
    cứng trong mã. Giá trị đang chốt tạm là 0.80.
    """
    assert answer_match._nguong_tuong_dong() == pytest.approx(0.80)


def test_moi_lan_so_deu_encode_lai(monkeypatch):
    """
    Mỗi lần so khớp không-exact đều phải encode lại cặp câu.

    Không có bộ nhớ đệm kết quả ở tầng này — và cũng KHÔNG cần, vì mô hình
    chạy local, deterministic tuyệt đối. Bản Task 12 đầu tiên dùng LLM trả
    phí nên mới cần cache; cache đó đã bỏ cùng với ANSWER_MATCH_CACHE_PATH.
    """
    mo_hinh = gan_mo_hinh_gia(monkeypatch, 0.95)

    for _ in range(5):
        is_semantic_match("màu xanh", "màu lục")

    assert mo_hinh.so_lan_encode == 5


def test_get_model_chi_dung_mo_hinh_mot_lan(monkeypatch):
    """
    _get_model() chỉ dựng SentenceTransformer ĐÚNG MỘT LẦN, các lần sau
    lấy lại từ _model_cache.

    Dựng lại mỗi lần gọi là chấm 12 câu hỏi–đáp × 100 dòng mất hàng chục
    phút thay vì vài giây. Test tiêm một module sentence_transformers giả
    vào sys.modules nên KHÔNG cần mạng và không tải gì.
    """
    import sys
    import types

    so_lan_dung = {"n": 0}

    class SentenceTransformerGia:
        def __init__(self, ten_mo_hinh):
            so_lan_dung["n"] += 1
            self.ten_mo_hinh = ten_mo_hinh

    module_gia = types.ModuleType("sentence_transformers")
    module_gia.SentenceTransformer = SentenceTransformerGia
    monkeypatch.setitem(sys.modules, "sentence_transformers", module_gia)

    lan_1 = answer_match._get_model()
    lan_2 = answer_match._get_model()

    assert lan_1 is lan_2, "lần gọi thứ hai phải trả về đúng đối tượng cũ"
    assert so_lan_dung["n"] == 1, "chỉ được dựng mô hình một lần"
    assert lan_1.ten_mo_hinh == answer_match.EMBEDDING_MODEL_NAME


def test_thieu_package_bao_loi_ro_rang(monkeypatch):
    """
    Chưa cài sentence-transformers -> RuntimeError với câu hướng dẫn cài,
    không phải ImportError trần trụi.

    Đáng có vì lỗi này nổ GIỮA CHỪNG lần chấm đầu tiên trên máy mới, lúc
    người ngồi máy đang chờ kết quả chứ không đang đọc mã.
    """
    import builtins

    import_that = builtins.__import__

    def import_gia(ten, *args, **kwargs):
        if ten == "sentence_transformers":
            raise ImportError("giả lập máy chưa cài")
        return import_that(ten, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_gia)

    with pytest.raises(RuntimeError, match="sentence-transformers"):
        answer_match._get_model()


# ---------------------------------------------------------------------------
# 4. r_score_qa — điều kiện số 2 mà test_scoring.py hứa nhưng chưa có
# ---------------------------------------------------------------------------


GT_QA = {
    "gt_video_id": "L05_V005",
    "gt_frame_range": [800, 900],
    "gt_answer": "màu xanh",
}


def test_qa_vi_du_btc_dung_het(monkeypatch):
    """Ví dụ BTC mục 2.1.2: L05_V005, 888, màu xanh -> R-Score = 1."""
    cam_nap_mo_hinh(monkeypatch)
    nop = {"video_id": "L05_V005", "frame_id": 888, "answer": "màu xanh"}
    assert r_score_qa(GT_QA, nop) == 1.0


def test_qa_vi_du_btc_sai_answer(monkeypatch):
    """Ví dụ BTC: L05_V005, 888, màu trắng -> sai answer -> R-Score = 0."""
    gan_mo_hinh_gia(monkeypatch, 0.30)
    nop = {"video_id": "L05_V005", "frame_id": 888, "answer": "màu trắng"}
    assert r_score_qa(GT_QA, nop) == 0.0


def test_qa_vi_du_btc_sai_video(monkeypatch):
    """
    Ví dụ BTC: L06_V007, 888, màu xanh -> sai video -> R-Score = 0,
    và KHÔNG được nạp mô hình.

    Thứ tự kiểm phải là video -> frame -> answer. Trong 100 dòng nộp thì
    đa số sai video; chạy embedding cho từng dòng đó là đốt thời gian vô
    ích.
    """
    cam_nap_mo_hinh(monkeypatch)
    nop = {"video_id": "L06_V007", "frame_id": 888, "answer": "màu xanh"}
    assert r_score_qa(GT_QA, nop) == 0.0


def test_qa_sai_frame_khong_nap_mo_hinh(monkeypatch):
    """Đúng video nhưng frame ngoài [800, 900] -> 0, không nạp mô hình."""
    cam_nap_mo_hinh(monkeypatch)
    nop = {"video_id": "L05_V005", "frame_id": 950, "answer": "màu xanh"}
    assert r_score_qa(GT_QA, nop) == 0.0


def test_qa_bien_khoang_frame(monkeypatch):
    """Hai đầu mút 800 và 900 đều tính là ĐÚNG (khoảng đóng)."""
    cam_nap_mo_hinh(monkeypatch)
    for frame_id in (800, 900):
        nop = {"video_id": "L05_V005", "frame_id": frame_id, "answer": "màu xanh"}
        assert r_score_qa(GT_QA, nop) == 1.0, f"frame {frame_id} phải được tính"


def test_qa_thieu_han_khoa_answer(monkeypatch):
    """
    Dòng nộp không có khoá "answer" -> 0, không nổ KeyError.

    Xảy ra thật nếu ai đó ghi tệp nộp Q&A bằng đường KIS. Chấm ra 0 thì
    còn nhìn thấy trong bảng; nổ giữa chừng thì mất luôn 31 câu còn lại.
    """
    cam_nap_mo_hinh(monkeypatch)
    nop = {"video_id": "L05_V005", "frame_id": 888}
    assert r_score_qa(GT_QA, nop) == 0.0


def test_qa_answer_khac_cach_dien_dat_van_khop(monkeypatch):
    """
    Đúng video, đúng frame, answer diễn đạt khác nhưng cùng nghĩa -> 1.

    Đây là lý do tồn tại của cả answer_match.py: BTC chấm "khớp về mặt
    ngữ nghĩa" (mục 2.1.2), không phải so chuỗi y hệt.
    """
    gan_mo_hinh_gia(monkeypatch, 0.92)
    nop = {"video_id": "L05_V005", "frame_id": 888, "answer": "xanh dương"}
    assert r_score_qa(GT_QA, nop) == 1.0
