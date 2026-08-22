"""
Việc 5 (mở rộng truy vấn Việt -> Anh) — Thi.

CẦN gói 01 của Nghi có trước: query_expand dùng BangNhan của objects_index.

Chỉ import mã của Thi — chạy được ngay cả khi các gói khác CHƯA trộn.
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

import pytest

from aic2026 import query_expand  # noqa: E402


def test_cat_chu_thua():
    ra = query_expand.cat_chu_thua(
        "Trong đoạn video, hãy tìm khoảnh khắc ba người đàn ông đội nón đỏ"
    )
    assert "hãy tìm" not in ra
    assert "trong đoạn video" not in ra
    assert "nón đỏ" in ra          # dấu được giữ nguyên


def test_khong_nham_tim_voi_tim_va_trong_voi_trong():
    """Bỏ dấu từng làm 'tìm' hoá 'tím' (purple) và 'trong' hoá 'trống' (Drum)."""
    from aic2026.index.objects_index import BangNhan

    bang = BangNhan({"trống": ["Drum"]})
    assert bang.tim_nhan("một cảnh không có vật thể nào trong bảng") == []
    assert bang.tim_nhan("bộ trống màu đỏ") == ["Drum"]

    kq = query_expand.dich_bang_tu_dien("hãy tìm cảnh trong nhà", bang_nhan=BangNhan({}))
    assert "purple" not in kq.cum_chinh


def test_dich_bang_tu_dien_ra_tieng_anh(tmp_path):
    from aic2026.index.objects_index import BangNhan

    bang = BangNhan({"nón": ["Hat"], "người": ["Person"]})
    kq = query_expand.dich_bang_tu_dien("ba người đội nón đỏ", bang_nhan=bang)
    cum = kq.cum_chinh.lower()
    assert "hat" in cum and "person" in cum
    assert "red" in cum
    assert "three" in cum


def test_khong_nhan_ra_gi_thi_tra_nguyen_ban():
    from aic2026.index.objects_index import BangNhan

    kq = query_expand.dich_bang_tu_dien("qwerty zxcvb", bang_nhan=BangNhan({}))
    assert kq.nguon == "nguyen_ban"
    assert kq.ghi_chu


def test_bo_nho_dem_ghi_va_doc_lai(tmp_path):
    dem = query_expand.BoNhoDem(tmp_path / "dem.json")
    dem.dat("llm", "ba nón đỏ", ["three red hats"])
    dem.ghi()

    dem_2 = query_expand.BoNhoDem(tmp_path / "dem.json")
    assert dem_2.lay("llm", "ba nón đỏ") == ["three red hats"]


def test_mo_rong_dung_lai_ban_da_dem(tmp_path):
    dem = query_expand.BoNhoDem(tmp_path / "dem.json")
    dem.dat("tu_dien", "câu bất kỳ", ["a fixed english phrase"])
    kq = query_expand.mo_rong("câu bất kỳ", nguon="tu_dien", dem=dem)
    assert kq.nguon == "dem"
    assert kq.cum_chinh == "a fixed english phrase"




def test_bien_moi_truong_THANG_settings_yaml(monkeypatch):
    """Cờ dòng lệnh là thứ người gõ NGAY LÚC ĐÓ — phải thắng tệp cấu hình.

    Bản đầu đọc settings.yaml trước, nên --nguon-mo-rong llm bị đè và im lặng
    chạy nhánh tu_dien. Hai lần chạy ra số giống hệt nhau mà người chạy tưởng
    đã thử xong nhánh LLM.
    """
    from aic2026 import query_expand

    monkeypatch.delenv("AIC_NGUON_MO_RONG", raising=False)
    theo_tep = query_expand.nguon_mac_dinh()

    monkeypatch.setenv("AIC_NGUON_MO_RONG", "marian")
    assert query_expand.nguon_mac_dinh() == "marian"

    monkeypatch.setenv("AIC_NGUON_MO_RONG", "llm")
    assert query_expand.nguon_mac_dinh() == "llm"

    monkeypatch.delenv("AIC_NGUON_MO_RONG")
    assert query_expand.nguon_mac_dinh() == theo_tep


def test_bien_rong_khong_de_settings(monkeypatch):
    from aic2026 import query_expand

    monkeypatch.setenv("AIC_NGUON_MO_RONG", "")
    assert query_expand.nguon_mac_dinh() in ("tu_dien", "marian", "llm")


# ---------------------------------------------------------------------------
# Lùi nhánh âm thầm — lỗi đã cắn thật khi đo TRAKE
# ---------------------------------------------------------------------------

def test_da_chi_dinh_nguon_thi_hong_phai_NEM_LOI(tmp_path, monkeypatch):
    """Thiếu khoá API thì nhánh llm lùi về tu_dien. Lùi âm thầm là phép đo giả:
    người chạy thấy dòng 'nguồn mở rộng: llm', thấy số, và tưởng đã đo xong.
    """
    from aic2026 import query_expand

    dem = query_expand.BoNhoDem(tmp_path / "dem.json")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(query_expand.LuiNhanhKhongMongMuon, match="lùi về"):
        query_expand.mo_rong("ba nón đỏ", nguon="llm", dem=dem, bat_buoc=True)


def test_khong_chi_dinh_thi_van_duoc_lui_am_tham(tmp_path, monkeypatch):
    """Không ai chỉ định gì thì lùi về tu_dien là hành vi đúng — vẫn chạy được."""
    from aic2026 import query_expand

    dem = query_expand.BoNhoDem(tmp_path / "dem.json")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    kq = query_expand.mo_rong("ba nón đỏ", nguon="llm", dem=dem, bat_buoc=False)
    assert kq.cum_chinh
    assert kq.ghi_chu                      # lý do vẫn được ghi lại


def test_ket_qua_lui_KHONG_duoc_dem_duoi_ten_nguon_da_yeu_cau(tmp_path, monkeypatch):
    """Đệm kết quả lùi dưới khoá 'llm' là đầu độc đệm: lần sau có khoá API
    thật vẫn đọc lại bản hỏng."""
    from aic2026 import query_expand

    dem = query_expand.BoNhoDem(tmp_path / "dem.json")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    query_expand.mo_rong("ba nón đỏ", nguon="llm", dem=dem, bat_buoc=False)
    assert dem.lay("llm", "ba nón đỏ") is None


def test_llm_chay_duoc_thi_dem_binh_thuong(tmp_path):
    from aic2026 import query_expand

    dem = query_expand.BoNhoDem(tmp_path / "dem.json")
    gia = lambda _: '["three red conical hats", "a photo of three red hats"]'

    kq = query_expand.dich_bang_llm("ba nón đỏ", goi_llm=gia)
    assert kq.nguon == "llm"
    assert kq.cum_chinh == "three red conical hats"

    dem.dat("llm", "ba nón đỏ", kq.cum_tieng_anh)
    dem.ghi()
    assert query_expand.BoNhoDem(tmp_path / "dem.json").lay("llm", "ba nón đỏ")


def test_dem_ban_cu_bi_bo_qua(tmp_path):
    """Đệm v1 chứa kết quả lùi — phải bị bỏ qua thay vì bắt từng người đi xoá."""
    import json

    from aic2026 import query_expand

    tep = tmp_path / "dem.json"
    tep.write_text(
        json.dumps({"llm": {"ba nón đỏ": ["bread cake"]}}), encoding="utf-8"
    )
    assert query_expand.BoNhoDem(tep).lay("llm", "ba nón đỏ") is None


def test_bao_loi_ro_khi_chua_co_khoa_api(monkeypatch):
    """Gõ dấu thăng ở đầu dòng là PowerShell coi cả dòng là chú thích.

    Đã cắn thật: người chạy dán '# thêm ANTHROPIC_API_KEY=... vào .env' vào
    PowerShell, không có gì xảy ra, rồi chạy tiếp và ngạc nhiên.
    """
    from aic2026 import query_expand

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Add-Content"):
        query_expand._goi_anthropic("dịch hộ")


def test_doc_duoc_ca_hai_dang_json():
    """Structured output trả {"cum": [...]}, prompt thường trả [...]."""
    from aic2026.query_expand import _doc_mang_json

    assert _doc_mang_json('{"cum": ["a", "b"]}') == ["a", "b"]
    assert _doc_mang_json('["a", "b"]') == ["a", "b"]
    assert _doc_mang_json('```json\n{"cum": ["x"]}\n```') == ["x"]
    assert _doc_mang_json("không phải json") == []


def test_khong_truyen_temperature_cho_anthropic():
    """Thư viện anthropic 1.0 đã BỎ HẲN temperature — truyền vào là TypeError.

    Tính lặp lại của phép đo do bộ nhớ đệm bảo đảm, không phải do temperature.
    """
    import inspect

    from aic2026 import query_expand

    ma = inspect.getsource(query_expand._goi_anthropic)
    # Bỏ docstring rồi mới soát: chữ "temperature" trong lời chú thích là để
    # GIẢI THÍCH vì sao không dùng, không phải chỗ gọi hàm.
    than_ham = ma.split('"""')[-1]
    assert "temperature" not in than_ham

    # Kiểm bằng đường thứ hai, độc lập với cách viết chú thích: bộ tham số
    # dựng ra phải nằm gọn trong chữ ký thật của thư viện.
    import anthropic

    hop_le = set(
        inspect.signature(anthropic.Anthropic(api_key="x").messages.create).parameters
    )
    for ten in ("model", "max_tokens", "messages"):
        assert ten in hop_le, f"thư viện đã bỏ tham số {ten}"
    assert "temperature" not in hop_le


def test_phan_loai_loi_api_dung_nguyen_nhan():
    """Bản đầu in cùng lời khuyên 'đổi tên mô hình' cho MỌI lỗi, kể cả lỗi
    xác thực — dẫn người đọc đi sai hướng ngay khi họ đang bí."""
    from aic2026.query_expand import _giai_thich_loi_api

    class Gia(Exception):
        def __init__(self, msg, code=None):
            super().__init__(msg)
            self.status_code = code

    khoa = _giai_thich_loi_api(Gia("authentication_error: API key is invalid.", 401), "m")
    assert "KHOÁ API" in khoa
    assert "AIC_MO_HINH_LLM" not in khoa      # KHÔNG khuyên đổi mô hình

    mo_hinh = _giai_thich_loi_api(Gia("model not found", 404), "m")
    assert "AIC_MO_HINH_LLM" in mo_hinh

    toc_do = _giai_thich_loi_api(Gia("rate limit exceeded", 429), "m")
    assert "giới hạn tốc độ" in toc_do

    la = _giai_thich_loi_api(Gia("connection reset"), "m")
    assert "connection reset" in la          # lỗi lạ vẫn hiện nguyên văn


def test_marian_ep_cung_nhanh_dich_khong_phu_thuoc_settings():
    """Bản đầu dựng ClipEncoder() suông, nên nó đọc settings.yaml thấy 'da_ngu'
    và KHÔNG dịch gì — hàm trả rỗng rồi lùi nhánh, dù mô hình dịch có sẵn.
    """
    import inspect

    from aic2026 import query_expand

    def than_ham(f):
        """Bỏ docstring: chữ trong lời chú thích là để GIẢI THÍCH, không phải
        chỗ gọi hàm. Đã cắn hai lần với chính kiểu test này."""
        ma = inspect.getsource(f)
        return ma.split('"""')[-1] if '"""' in ma else ma

    assert 'nhanh_chu="dich"' in than_ham(query_expand._lay_bo_dich)

    # KHÔNG được đổi biến môi trường chung: làm vậy sẽ đổi luôn nhánh chữ của
    # phần tra cứu chính trong cùng tiến trình.
    for f in (query_expand._lay_bo_dich, query_expand.dich_bang_marian):
        assert "AIC_NHANH_CHU" not in than_ham(f)
        assert "environ" not in than_ham(f)


def test_marian_tra_ve_dung_nguon_khi_dich_duoc(monkeypatch):
    from aic2026 import query_expand

    class GiaEncoder:
        def __init__(self):
            self._d = ""

        def encode_text(self, t):
            self._d = "a man lifting a rice cake out of a basket"

        def ban_dich_gan_nhat(self):
            return self._d

    monkeypatch.setattr(query_expand, "_BO_DICH", GiaEncoder())
    kq = query_expand.dich_bang_marian("nhấc bánh khỏi rổ")
    assert kq.nguon == "marian"
    assert "lifting" in kq.cum_chinh


def test_marian_hong_thi_bao_ly_do_chu_khong_im_lang(monkeypatch):
    from aic2026 import query_expand

    class Hong:
        def encode_text(self, t):
            raise RuntimeError("mô hình chưa tải xong")

        def ban_dich_gan_nhat(self):
            return ""

    monkeypatch.setattr(query_expand, "_BO_DICH", Hong())
    kq = query_expand.dich_bang_marian("nhấc bánh khỏi rổ")
    assert kq.nguon == "nguyen_ban"
    assert any("mô hình chưa tải xong" in c for c in kq.ghi_chu)


def test_ban_dich_tra_lai_tieng_viet_bi_coi_la_HONG(monkeypatch):
    """VietAI/envit5 là mô hình HAI CHIỀU và thỉnh thoảng chép lại đầu vào.

    Chuỗi đó KHÔNG rỗng nên phép kiểm 'có trả về gì không' cho qua, rồi cả
    mạch chạy ở mốc sàn. Đã cắn thật: 'một người đàn ông cầm ô' dịch ra chính
    nó, và cả 6 sự kiện TRAKE chạy bằng tiếng Việt.
    """
    from aic2026 import query_expand

    class Echo:
        def encode_text(self, t):
            pass

        def ban_dich_gan_nhat(self):
            return "một người đàn ông cầm ô"

    monkeypatch.setattr(query_expand, "_BO_DICH", Echo())
    kq = query_expand.dich_bang_marian("một người đàn ông cầm ô")
    assert kq.nguon == "nguyen_ban"
    assert any("trả lại tiếng Việt" in c for c in kq.ghi_chu)

    # và khi người chạy ĐÃ chỉ định marian thì phải nổ, không lùi âm thầm
    with pytest.raises(query_expand.LuiNhanhKhongMongMuon):
        query_expand.mo_rong(
            "một người đàn ông cầm ô", nguon="marian", dung_dem=False, bat_buoc=True
        )


def test_bo_tien_to_tac_vu_cua_envit5(monkeypatch):
    from aic2026 import query_expand

    class CoTienTo:
        def encode_text(self, t):
            pass

        def ban_dich_gan_nhat(self):
            return "en: a man holding an umbrella"

    monkeypatch.setattr(query_expand, "_BO_DICH", CoTienTo())
    kq = query_expand.dich_bang_marian("một người đàn ông cầm ô")
    assert kq.nguon == "marian"
    assert kq.cum_chinh == "a man holding an umbrella"


def test_con_tieng_viet():
    from aic2026.query_expand import con_tieng_viet

    assert con_tieng_viet("nhấc bánh khỏi rổ")
    assert con_tieng_viet("Đèo Tà Pứa")
    assert not con_tieng_viet("a man lifting a cake out of a basket")
    assert not con_tieng_viet("")
