"""
So khớp NGỮ NGHĨA answer cho truy vấn Q&A — Task 12.
 
BTC chấm: a_i = GT_a "khớp về mặt ngữ nghĩa", không phải so chuỗi y hệt
(mục 2.1.2 tài liệu vòng sơ tuyển). Bộ dev tự soạn không có giám khảo người.
 
CHIẾN LƯỢC — HOÀN TOÀN MIỄN PHÍ, KHÔNG GỌI API TRẢ PHÍ:
 
  1) Chuẩn hoá + so khớp CHÍNH XÁC trước — rẻ, tức thì.
  2) Không khớp -> so độ tương đồng NGỮ NGHĨA bằng mô hình embedding câu
     (sentence-transformers) chạy LOCAL trên máy, không gọi mạng, không
     tốn phí. Package này ĐÃ CÓ SẴN trong requirements.txt (dùng cho việc
     khác ở Giai đoạn 1), không cần thêm gì, không cần báo nhóm.
 
Mô hình tải về máy đúng MỘT LẦN (khi chạy lần đầu, cần mạng để tải từ
HuggingFace Hub), sau đó chạy hoàn toàn offline, không tốn phí mỗi lần gọi.
 
Ngưỡng độ tương đồng (nguong_tuong_dong_qa) đọc từ config/settings.yaml,
mục cham_diem — CẦN NHÓM XÁC NHẬN, xem docs/decisions/002-...md.
"""
 
from __future__ import annotations
 
import re
import unicodedata
 
from ..rank.config import load_settings
 
# Mô hình đa ngôn ngữ (hỗ trợ tiếng Việt), nhẹ, chạy tốt trên CPU không cần
# GPU — hợp với máy nhóm (đúng ràng buộc "không GPU" đã ghi ở Task 13).
# Tải về máy ĐÚNG MỘT LẦN, các lần sau load từ cache local, không cần mạng.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
 
NGUONG_TUONG_DONG_MAC_DINH = 0.80
 
_model_cache = None  # nạp mô hình đúng 1 lần, dùng lại cho mọi lần gọi
 
 
def normalize_text(text: str | None) -> str:
    """Lowercase, bỏ dấu câu, gộp khoảng trắng thừa.
 
    KHÔNG bỏ dấu tiếng Việt — bỏ dấu dễ gộp nhầm hai từ khác nghĩa
    (ví dụ "ma" và "má").
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text.strip().lower())
    text = re.sub(r"[.,!?;:'\"()\[\]{}]", "", text)
    return re.sub(r"\s+", " ", text).strip()
 
 
def _nguong_tuong_dong() -> float:
    gia_tri = load_settings().get("cham_diem", {})
    if not isinstance(gia_tri, dict):
        return NGUONG_TUONG_DONG_MAC_DINH
    return float(gia_tri.get("nguong_tuong_dong_qa", NGUONG_TUONG_DONG_MAC_DINH))
 
 
def _get_model():
    """Nạp mô hình embedding, cache lại trong bộ nhớ — chỉ nạp 1 lần/lần chạy."""
    global _model_cache
    if _model_cache is None:
        try:
            from sentence_transformers import SentenceTransformer  
        except ImportError:
            raise RuntimeError(
                "Chưa cài package 'sentence-transformers'. Package này ĐÃ CÓ "
                "trong requirements.txt — chạy: pip install -r requirements.txt"
            )
        _model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_cache
 
 
def _cosine_similarity(a, b) -> float:
    import numpy as np
 
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
 
 
def is_semantic_match(gt_answer: str, submitted_answer: str) -> bool:
    """Hàm public duy nhất mà scoring.py gọi.
 
    1. Exact match sau normalize -> True ngay, không cần chạy mô hình.
    2. Không khớp -> so độ tương đồng embedding, ngưỡng đọc từ settings.yaml.
 
    KHÔNG gọi mạng trả phí nào. Lần đầu chạy trên máy cần mạng để TẢI mô
    hình (miễn phí, một lần); các lần sau chạy offline hoàn toàn.
    """
    if normalize_text(gt_answer) == normalize_text(submitted_answer):
        return True
 
    if not submitted_answer:
        return False
 
    model = _get_model()
    embeddings = model.encode([gt_answer, submitted_answer])
    do_tuong_dong = _cosine_similarity(embeddings[0], embeddings[1])
 
    return do_tuong_dong >= _nguong_tuong_dong()