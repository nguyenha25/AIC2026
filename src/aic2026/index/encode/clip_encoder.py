"""
CLIP encoder dùng chung cho Việc 3.

NHÁNH ẢNH GIỮ NGUYÊN. NHÁNH CHỮ CHỌN ĐƯỢC.

Vấn đề: BTC phát câu truy vấn bằng tiếng Việt. CLIP ViT-B/32 bản "openai"
học gần như toàn bộ trên chú thích tiếng Anh. Gõ tiếng Việt vẫn chạy, vẫn
trả đủ 100 kết quả, chỉ là vô nghĩa — lỗi không có triệu chứng.

Ràng buộc: KHÔNG mã hoá lại 177.321 ảnh. Vector ảnh do BTC phát, phải giữ
nguyên. Nên chỉ đổi nhánh chữ.

BA NHÁNH CHỮ, chọn bằng tra_cuu.nhanh_ma_hoa_chu trong config/settings.yaml:

    da_ngu  (MẶC ĐỊNH)  sentence-transformers/clip-ViT-B-32-multilingual-v1
                        Nền DistilBERT đa ngữ, chưng cất từ chính text encoder
                        của ViT-B/32 bản openai — cùng không gian vector với
                        vector ảnh BTC phát.
                        Không cần cài thêm gói. Nhanh hơn nhánh gốc.

    dich                Dịch Việt→Anh rồi đưa vào text encoder gốc.
                        Người ngồi máy đọc được bản dịch ở thuộc tính
                        ban_dich_gan_nhat và sửa tay nếu dịch sai tên riêng.
                        Chậm hơn, cần tải thêm mô hình dịch.

    goc                 Đưa thẳng vào text encoder gốc.
                        CHỈ dùng cho câu tiếng Anh. Với tiếng Việt đây là mốc
                        sàn để so, không phải phương án dùng thật.

VÌ SAO CÓ KHOÁ NÀY: vào vòng thi mà một nhánh trục trặc — mô hình chưa tải
về, cache hỏng, máy thiếu RAM — thì đổi một dòng trong settings.yaml là chạy
tiếp được, không phải sửa mã.

VÌ SAO KHÔNG TỰ ĐỘNG CHUYỂN NHÁNH KHI LỖI: tự lùi về "goc" nghĩa là câu tiếng
Việt lặng lẽ cho ra kết quả vô nghĩa mà màn hình vẫn đủ 100 dòng — đúng loại
lỗi không triệu chứng mà cả Việc 3 sinh ra để chặn. Nên ở đây LỖI THÌ DỪNG,
kèm thông báo chỉ rõ phải sửa khoá nào.

PHÉP THỬ CÙNG HỆ KHÔNG BỊ ẢNH HƯỞNG:
encode_image() và encode_image_path() vẫn dùng đúng open_clip ViT-B-32 bản
openai như trước. Đổi nhánh chữ không đụng đến chúng, nên
scripts/verify_clip_encoder.py chạy y nguyên và phải ra đúng con số cũ
(đã đo: cosine trung bình 0,999927 trên 30 ảnh).

CHẠY OFFLINE (địa điểm thi có thể không có mạng):
Tải mô hình về trước một lần, rồi đặt biến môi trường trước khi chạy:
    $env:HF_HUB_OFFLINE = "1"
Mô hình nằm trong C:\\Users\\<tên>\\.cache\\huggingface\\hub — chép cache đó
sang máy khác là dùng được, không cần tải lại.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Đọc khoá cấu hình
#
# Cố ý KHÔNG import từ rank/config.py: tệp đó thuộc nhánh xếp hạng của Ngân,
# còn tệp này thuộc nhánh xương sống. Nhánh dưới không nên phụ thuộc nhánh
# trên — đổi thứ tự import một lần là gãy cả hai.
# ---------------------------------------------------------------------------

NHANH_MAC_DINH = "da_ngu"
CAC_NHANH_HOP_LE = ("da_ngu", "dich", "goc")

_CACHED_SETTINGS: dict | None = None


def _doc_settings() -> dict:
    """Đọc config/settings.yaml một lần rồi giữ trong RAM."""
    global _CACHED_SETTINGS

    if _CACHED_SETTINGS is not None:
        return _CACHED_SETTINGS

    try:
        import yaml

        from ...paths import CONFIG_DIR

        duong_dan = CONFIG_DIR / "settings.yaml"

        if duong_dan.exists():
            with duong_dan.open("r", encoding="utf-8") as f:
                _CACHED_SETTINGS = yaml.safe_load(f) or {}
        else:
            _CACHED_SETTINGS = {}
    except Exception:  # noqa: BLE001 — thiếu cấu hình thì chạy mặc định
        _CACHED_SETTINGS = {}

    return _CACHED_SETTINGS


def nhanh_ma_hoa_chu() -> str:
    """
    Tên nhánh chữ đang dùng.

    Thứ tự ưu tiên: biến môi trường AIC_NHANH_CHU > settings.yaml > mặc định.
    Biến môi trường để thử nhanh trong vòng thi mà không phải sửa tệp:
        $env:AIC_NHANH_CHU = "dich"
    """
    tu_moi_truong = os.getenv("AIC_NHANH_CHU")

    if tu_moi_truong:
        ten = tu_moi_truong.strip().lower()
    else:
        muc = _doc_settings().get("tra_cuu")
        muc = muc if isinstance(muc, dict) else {}
        ten = str(muc.get("nhanh_ma_hoa_chu", NHANH_MAC_DINH)).strip().lower()

    if ten not in CAC_NHANH_HOP_LE:
        raise ValueError(
            f"tra_cuu.nhanh_ma_hoa_chu = '{ten}' không hợp lệ. "
            f"Chọn một trong {CAC_NHANH_HOP_LE}."
        )

    return ten


def xoa_cache_settings() -> None:
    """Gọi sau khi sửa settings.yaml trong cùng một lần chạy."""
    global _CACHED_SETTINGS
    _CACHED_SETTINGS = None


# ---------------------------------------------------------------------------
# Ba nhánh chữ
#
# Cả ba có cùng một hàm: ma_hoa(text) -> vector (512,) float32 CHƯA chuẩn hoá.
# Việc chuẩn hoá làm tập trung ở ClipEncoder để không nhánh nào lỡ làm hai lần.
# ---------------------------------------------------------------------------


class _NhanhGoc:
    """Text encoder gốc của open_clip. Dùng cho câu tiếng Anh."""

    ten = "goc"

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def ma_hoa(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text])

        with torch.no_grad():
            features = self.model.encode_text(tokens)

        return features[0].cpu().numpy().astype(np.float32)


class _NhanhDaNgu:
    """
    Text encoder đa ngữ, cùng không gian vector với ViT-B/32 bản openai.

    Mô hình này được chưng cất TỪ text encoder của chính bản openai, nên vector
    nó sinh ra so được trực tiếp với vector ảnh BTC phát. Không cần mã hoá lại
    ảnh nào.

    Đo trên bộ câu hỏi tiếng Việt của nhóm: hơn nhánh gốc 0,133 điểm theo công
    thức Việc 12, và cosine trung bình của kết quả hạng 1 là 0,312 — nằm trong
    dải lành mạnh 0,25–0,35, xác nhận nó cùng hệ với nhánh ảnh.
    """

    ten = "da_ngu"
    TEN_MO_HINH = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as loi:
            raise ImportError(
                "Nhánh 'da_ngu' cần sentence-transformers (đã có trong "
                "requirements.txt, bản 3.0.1). Cài lại bằng:\n"
                "    python -m pip install -r requirements.txt"
            ) from loi

        try:
            self.model = SentenceTransformer(self.TEN_MO_HINH, device="cpu")
        except Exception as loi:  # noqa: BLE001
            raise RuntimeError(
                f"Không nạp được {self.TEN_MO_HINH}: {loi}\n"
                "Chưa tải mô hình mà đang chạy offline thì bỏ HF_HUB_OFFLINE ra, "
                "chạy một lần cho nó tải về, rồi bật lại.\n"
                "Cần chạy ngay thì đổi tra_cuu.nhanh_ma_hoa_chu sang 'dich' "
                "trong config/settings.yaml."
            ) from loi

    def ma_hoa(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            [text], convert_to_numpy=True, show_progress_bar=False
        )
        return vector[0].astype(np.float32)


class _NhanhDich:
    """
    Dịch Việt→Anh rồi đưa vào text encoder gốc.

    Ưu điểm riêng của nhánh này: bản dịch nằm ở thuộc tính ban_dich_gan_nhat,
    nên giao diện của Việc 7 hiện được cho người ngồi máy đọc. Dịch sai tên
    riêng thì sửa tay rồi tra lại — nhánh đa ngữ không cho làm việc đó vì
    không có bước trung gian nào để nhìn.
    """

    ten = "dich"
    TEN_MO_HINH = "VietAI/envit5-translation"

    def __init__(self, model, tokenizer):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as loi:
            raise ImportError(
                "Nhánh 'dich' cần transformers (đã có trong requirements.txt, "
                "bản 4.41.2)."
            ) from loi

        try:
            self.tokenizer_dich = AutoTokenizer.from_pretrained(self.TEN_MO_HINH)
            self.model_dich = AutoModelForSeq2SeqLM.from_pretrained(self.TEN_MO_HINH)
        except Exception as loi:  # noqa: BLE001
            raise RuntimeError(
                f"Không nạp được mô hình dịch {self.TEN_MO_HINH}: {loi}\n"
                "Mô hình này khoảng 900 MB, phải tải về trước khi chạy offline.\n"
                "Cần chạy ngay thì đổi tra_cuu.nhanh_ma_hoa_chu sang 'da_ngu'."
            ) from loi

        self.model_dich.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.ban_dich_gan_nhat: str = ""

    def dich(self, text: str) -> str:
        """Việt → Anh. envit5 đòi tiền tố 'vi: ' và trả về chuỗi mở đầu 'en: '."""
        goi = self.tokenizer_dich(
            [f"vi: {text}"],
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            dau_ra = self.model_dich.generate(**goi, max_new_tokens=128)

        ket_qua = self.tokenizer_dich.batch_decode(dau_ra, skip_special_tokens=True)[0]
        return ket_qua.removeprefix("en: ").strip()

    def ma_hoa(self, text: str) -> np.ndarray:
        self.ban_dich_gan_nhat = self.dich(text)

        tokens = self.tokenizer([self.ban_dich_gan_nhat])

        with torch.no_grad():
            features = self.model.encode_text(tokens)

        return features[0].cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# ClipEncoder — chữ ký giữ nguyên hoàn toàn
# ---------------------------------------------------------------------------


class ClipEncoder:
    """
    Mã hoá ảnh và văn bản cho hệ tra cứu.

    CHỮ KÝ KHÔNG ĐỔI so với bản trước:
        encode_text(text) -> np.ndarray (512,) float32, đã chuẩn hoá
        encode_image(image) -> np.ndarray (512,) float32, đã chuẩn hoá
        encode_image_path(path) -> như trên
        dimension, model_name, pretrained

    rank/search.py của Ngân và scripts/verify_clip_encoder.py chạy y nguyên,
    không phải sửa một dòng nào.

    NẠP MÔ HÌNH THEO NHU CẦU: mô hình ảnh chỉ nạp khi gọi encode_image lần đầu,
    mô hình chữ chỉ nạp khi gọi encode_text lần đầu. Trước đây __init__ nạp
    open_clip ngay, nên chạy phép thử cùng hệ (chỉ dùng ảnh) vẫn phải chờ nạp
    cả nhánh chữ và ngược lại.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        nhanh_chu: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained

        # Đọc khoá cấu hình NGAY ở __init__ để khoá sai thì lỗi lộ ra lúc dựng
        # đối tượng, không phải lúc gõ câu truy vấn đầu tiên trong vòng thi.
        self.nhanh_chu = nhanh_chu or nhanh_ma_hoa_chu()

        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._nhanh = None

    # -- nhánh ảnh: KHÔNG ĐỔI -----------------------------------------------

    def _nap_open_clip(self):
        """Nạp open_clip đúng một lần. Đây là mô hình của phép thử cùng hệ."""
        if self._model is None:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model.eval()

        return self._model, self._preprocess, self._tokenizer

    @property
    def dimension(self) -> int:
        """Số chiều vector CLIP."""
        return 512

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Mã hoá một ảnh PIL thành vector float32 đã chuẩn hoá.

        Luôn dùng open_clip ViT-B-32 bản openai, bất kể nhánh chữ nào đang bật.
        Đây là chỗ phép thử cùng hệ đo, không được đụng vào.
        """
        model, preprocess, _ = self._nap_open_clip()

        image_input = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            features = model.encode_image(image_input)

        features = features / features.norm(dim=-1, keepdim=True)

        return features[0].cpu().numpy().astype(np.float32)

    def encode_image_path(self, image_path: str | Path) -> np.ndarray:
        """Mã hoá ảnh từ đường dẫn."""
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.encode_image(image)

    # -- nhánh chữ: chọn được ------------------------------------------------

    def _nap_nhanh_chu(self):
        """Dựng nhánh chữ đúng một lần, theo khoá cấu hình."""
        if self._nhanh is not None:
            return self._nhanh

        if self.nhanh_chu == "da_ngu":
            # Không đụng open_clip: nhánh này không cần text encoder gốc.
            self._nhanh = _NhanhDaNgu()
        else:
            model, _, tokenizer = self._nap_open_clip()

            if self.nhanh_chu == "goc":
                self._nhanh = _NhanhGoc(model, tokenizer)
            elif self.nhanh_chu == "dich":
                self._nhanh = _NhanhDich(model, tokenizer)
            else:
                raise ValueError(f"Nhánh chữ lạ: {self.nhanh_chu}")

        return self._nhanh

    def encode_text(self, text: str) -> np.ndarray:
        """
        Mã hoá một câu truy vấn thành vector float32 (512,) đã chuẩn hoá.

        Nhánh nào chạy tuỳ tra_cuu.nhanh_ma_hoa_chu trong settings.yaml.
        Kết quả trả về giống hệt nhau về kiểu và hình dạng ở cả ba nhánh, nên
        phía gọi không cần biết nhánh nào đang bật.
        """
        if not text or not text.strip():
            raise ValueError("Câu truy vấn rỗng.")

        vector = self._nap_nhanh_chu().ma_hoa(text.strip())

        if vector.shape != (512,):
            raise ValueError(
                f"Nhánh '{self.nhanh_chu}' trả vector {vector.shape}, cần (512,). "
                "Vector khác 512 chiều không tra được kho FAISS."
            )

        do_dai = float(np.linalg.norm(vector))

        if do_dai == 0:
            raise ValueError(
                f"Nhánh '{self.nhanh_chu}' trả vector có độ dài 0 cho câu: {text[:60]}"
            )

        return (vector / do_dai).astype(np.float32)

    @property
    def ban_dich_gan_nhat(self) -> str:
        """
        Bản dịch của câu vừa mã hoá — CHỈ có ở nhánh 'dich'.

        Giao diện Việc 7 hiện chuỗi này cho người ngồi máy soát. Ở hai nhánh
        kia trả về chuỗi rỗng vì không có bước dịch nào để mà xem.
        """
        return getattr(self._nhanh, "ban_dich_gan_nhat", "")

    def __repr__(self) -> str:
        return (
            f"<ClipEncoder ảnh={self.model_name}/{self.pretrained} "
            f"chữ={self.nhanh_chu}>"
        )