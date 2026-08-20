"""
Module OCR Retrieval & Reranking cho Task 10.

BẢN SỬA — bốn thay đổi so với bản trước:

  (3) extract_ocr_keywords() không còn moi ra rác.
      Đo trên bộ dev 32 câu, bản cũ trích nhầm 'vang khac tren nen đen' (câu 03),
      'mau xanh' (câu 09), '2005'/'2014' (câu 29) — toàn mô tả và số năm, không
      phải chữ trên màn hình. Rác này sinh hit rác: câu 29 nằm ở lô L25 (KHÔNG có
      dữ liệu OCR) mà vẫn trả về 3 hit từ video khác.

  (4) Lọc theo độ tin cậy conf khi dựng index.
      EasyOCR trả cả box rác như 'icjugrv' (conf 0.177). Bản cũ nhét tất vào index.
      Nay bỏ box dưới NGUONG_CONF. Khoảng 60% box của bộ dữ liệu nằm dưới 0.40.

  (2) Thêm doc_chu_tai_frame() phục vụ câu hỏi Q&A.
      LƯU Ý GIỚI HẠN: các câu kiểu "gói bột mang tên gì", "thương hiệu nào"
      (câu 05, 13, 14 bộ dev) KHÔNG thể giải bằng search_ocr, vì chữ cần tìm
      chính là ĐÁP ÁN, không nằm trong câu hỏi — không có gì để trích làm từ khoá.
      Hàm này chỉ là HẠ TẦNG: cho phép đọc chữ OCR tại frame đã truy hồi được,
      để nhánh Q&A điền cột đáp án. Việc chọn ra đáp án đúng từ đám chữ đó cần
      VLM/LLM, thuộc Giai đoạn 2 — KHÔNG giải quyết trong tệp này.

  (+) Sửa gốc import: 'from src.aic2026.paths' -> 'from aic2026.paths'.
      Tên gói cài đặt là aic2026 (pyproject: name="aic2026", where=["src"]).
      Dùng 'src.aic2026' làm Python nạp HAI cây module riêng biệt, sinh hai bản
      FrameMap/paths khác nhau — nguồn lỗi ngầm khi so sánh kiểu dữ liệu.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from aic2026.paths import DERIVED_DIR

# Box dưới ngưỡng này bị loại khi dựng index. 0.40 theo khảo sát chất lượng OCR
# của nhóm; chỉnh bằng tham số nguong_conf nếu muốn thử giá trị khác.
NGUONG_CONF = 0.40


def remove_accents(input_str: str) -> str:
    if not input_str:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join(c for c in nfkd_form if not unicodedata.combining(c)).lower()


# --- Bộ lọc cho phần trích từ khoá -----------------------------------------

# Từ mở đầu báo hiệu "đây là MÔ TẢ, không phải chữ trên màn hình".
# 'chữ vàng khắc trên nền đen' -> mô tả. 'bảng tên màu xanh' -> mô tả.
TU_MO_TA = {
    "mau", "vang", "do", "xanh", "trang", "den", "cam", "tim", "nau", "xam",
    "hong", "bac", "sang", "toi", "lon", "nho", "to", "be", "cao", "thap",
    "dai", "ngan", "tron", "vuong", "dep", "cu", "gi", "nao", "bao",
}

# Không dùng làm từ khoá tra OCR dù lọt qua các luật trên.
STOP_WORDS = {
    "trong", "video", "tren", "duoi", "hinh", "anh", "cua", "mot", "cac",
    "nhung", "phia", "ben", "va", "voi", "co", "la", "nay", "do",
}

# Cụm sau các từ mồi này MỚI có khả năng là chữ hiện trên màn hình.
_MOI = (
    r"chữ|dòng chữ|bảng ghi|bảng tên|mang tên|tên là|in chữ|"
    r"thương hiệu|hiệu|băng rôn|khẩu hiệu|biển|biểu ngữ|tiêu đề"
)
_PATTERN_MOI = re.compile(
    rf"(?:{_MOI})\s+([A-ZÀ-Ỹa-zà-ỹ0-9][A-ZÀ-Ỹa-zà-ỹ0-9\s\-]*?)"
    r"(?=,|\.|$|\bđặt\b|\bđứng\b|\bphía\b|\bđang\b|\btreo\b|\bnằm\b|\bbên\b)",
    flags=re.IGNORECASE,
)

# Token IN HOA: phải có ÍT NHẤT 3 CHỮ CÁI HOA thật.
# Bản cũ dùng [A-Z0-9\-]{3,} nên nuốt luôn '2005', '2014', '2024-2025'.
_PATTERN_HOA = re.compile(r"\b(?=(?:[A-Z0-9\-]*[A-Z]){3,})[A-Z0-9\-]{3,}\b")

_PATTERN_NGOAC = re.compile(r"[\"'“”‘’](.*?)[\"'“”‘’]")
_PATTERN_NAM = re.compile(r"\d{4}")

# Token ngắn rút TỪ TRONG cụm đã được từ mồi xác nhận (xem chú thích ở mục 3).
# Cụm năm: '2024-2025', '1655-1735', '2024'.
_PATTERN_NAM_CUM = re.compile(r"\d{4}(?:\s*-\s*\d{4})?")
# Chuỗi tên riêng viết hoa liên tiếp: 'Mạc Cửu', 'Nguyễn Huệ'.
_PATTERN_TEN_RIENG = re.compile(
    r"(?:[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+|$)){2,}|[A-ZÀ-Ỹ][a-zà-ỹ]{3,}"
)


def _dang_chu_man_hinh(cum: str) -> bool:
    """
    Cụm bắt được sau từ mồi có ĐÚNG là chữ trên màn hình không?

    Giữ nếu chứa danh từ riêng (token viết hoa) HOẶC cụm 4 chữ số (năm trên băng
    rôn, bảng ghi). Loại nếu mở đầu bằng từ mô tả.

        'Mạc Cửu 1655-1735'            -> giữ  (viết hoa + số)
        'chào mừng năm học mới 2024-2025' -> giữ  (có 2024)
        'vàng khắc trên nền đen'       -> loại (mở đầu 'vàng', không hoa/số)
        'màu xanh'                     -> loại
    """
    cum = cum.strip()
    if len(cum) < 3:
        return False

    cac_tu = cum.split()
    if not cac_tu:
        return False

    if remove_accents(cac_tu[0]) in TU_MO_TA:
        return False

    co_viet_hoa = any(t[:1].isupper() for t in cac_tu)
    co_nam = bool(_PATTERN_NAM.search(cum))

    return co_viet_hoa or co_nam


def extract_ocr_keywords(query: str) -> list[str]:
    """
    Trích các token có khả năng xuất hiện dưới dạng CHỮ trên khung hình.

    Trả về rỗng khi câu chỉ mô tả cảnh — đó là hành vi ĐÚNG, không phải lỗi:
    thà OCR im còn hơn sinh hit rác kéo tụt kết quả nhánh hình ảnh.
    """
    if not query:
        return []

    keywords: list[str] = []

    # 1. Chữ trong ngoặc kép — tín hiệu mạnh nhất, nhận vô điều kiện.
    keywords.extend(_PATTERN_NGOAC.findall(query))

    # 2. Token IN HOA (tên riêng, thương hiệu): BURBERRY, HONDA...
    keywords.extend(_PATTERN_HOA.findall(query))

    # 3. Cụm sau từ mồi, đã qua kiểm tra _dang_chu_man_hinh().
    #
    #    Thêm cả TOKEN NGẮN đặc trưng rút từ chính cụm đó. Lý do: search_ocr khớp
    #    bằng chuỗi con (kw in text), mà text OCR ghép từ nhiều box theo thứ tự và
    #    khoảng trắng riêng — cụm dài 'chao mung nam hoc moi 2024-2025' hầu như
    #    không bao giờ khớp trọn, chỉ cần OCR sai một ký tự là trượt. Token ngắn
    #    '2024-2025' mới là thứ khớp được.
    #
    #    Token ngắn CHỈ lấy từ cụm đã được từ mồi xác nhận, nên '2005'/'2014' ở câu
    #    hỏi phân tích số liệu (không có từ mồi) vẫn bị loại như mong muốn.
    for m in _PATTERN_MOI.findall(query):
        cum = m.strip()
        if not _dang_chu_man_hinh(cum):
            continue

        keywords.append(cum)
        keywords.extend(_PATTERN_NAM_CUM.findall(cum))
        keywords.extend(_PATTERN_TEN_RIENG.findall(cum))

    norm_keywords = set()
    for kw in keywords:
        norm = remove_accents(kw).strip()
        if len(norm) >= 3 and norm not in STOP_WORDS:
            norm_keywords.add(norm)

    return sorted(norm_keywords)


class OCRReranker:
    def __init__(
        self,
        ocr_dir: Path | None = None,
        nguong_conf: float = NGUONG_CONF,
    ):
        self.ocr_dir = ocr_dir or (DERIVED_DIR / "ocr")
        self.nguong_conf = float(nguong_conf)
        self._index: list[dict] = []
        self.so_box_bo = 0   # thống kê để biết ngưỡng cắt đi bao nhiêu
        self.so_box_giu = 0
        self._build_in_memory_index()

    # -- dựng index ---------------------------------------------------------

    def _text_da_loc(self, item: dict) -> str:
        """
        Ghép lại text từ các box đạt ngưỡng conf.

        Nếu dòng không có 'boxes' (schema cũ/khác) thì lùi về trường 'text' cấp
        dòng — không lọc được thì thà giữ còn hơn mất dữ liệu.
        """
        boxes = item.get("boxes")
        if not isinstance(boxes, list) or not boxes:
            return str(item.get("text", "") or "")

        phan_giu = []
        for b in boxes:
            if not isinstance(b, dict):
                continue
            try:
                conf = float(b.get("conf", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            chu = str(b.get("text", "") or "").strip()
            if not chu:
                continue
            if conf >= self.nguong_conf:
                phan_giu.append(chu)
                self.so_box_giu += 1
            else:
                self.so_box_bo += 1

        return " ".join(phan_giu)

    def _build_in_memory_index(self):
        if not self.ocr_dir.exists():
            return

        for file_path in sorted(self.ocr_dir.glob("*.jsonl")):
            v_id = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    raw_text = self._text_da_loc(item)
                    if not raw_text.strip():
                        continue

                    self._index.append(
                        {
                            "video_id": v_id,
                            # 'n' PHẢI giữ: đổi kết quả OCR thành Hit cần tra
                            # lookup(video_id, n) để lấy pts_time. Bỏ n đi thì
                            # nhánh OCR không cắm vào run_query() được.
                            "n": item.get("n"),
                            "frame_idx": item.get("frame_idx", 0),
                            "norm_text": remove_accents(raw_text),
                            "text": raw_text,
                        }
                    )

    # -- tra cứu ------------------------------------------------------------

    def search_ocr(self, query: str, top_k: int = 100) -> list[dict]:
        """Tìm frame chứa từ khoá trích được từ câu truy vấn."""
        keywords = extract_ocr_keywords(query)
        if not keywords or not self._index:
            return []

        matched_frames = []
        for doc in self._index:
            text = doc["norm_text"]
            match_count = sum(1 for kw in keywords if kw in text)
            if match_count > 0:
                matched_frames.append(
                    {
                        "video_id": doc["video_id"],
                        "n": doc.get("n"),
                        "frame_idx": doc["frame_idx"],
                        "score": match_count * 2.0,
                    }
                )

        matched_frames.sort(key=lambda x: x["score"], reverse=True)
        return matched_frames[:top_k]

    def doc_chu_tai_frame(
        self,
        video_id: str,
        frame_idx: int,
        cua_so_frame: int = 120,
    ) -> list[str]:
        """
        Đọc chữ OCR tại/quanh một frame ĐÃ TRUY HỒI ĐƯỢC.

        Dùng cho câu Q&A dạng "cái này tên gì" — nhánh hình ảnh tìm ra cảnh, hàm
        này lấy chữ trong cảnh đó ra để làm ứng viên đáp án.

        cua_so_frame nới rộng vì keyframe lấy thưa (~75 frame/lần trên bộ dữ liệu
        này), chữ có thể nằm ở keyframe kề bên chứ không đúng frame trúng đích.

        KHÔNG chọn đáp án hộ: trả về danh sách chuỗi thô, việc quyết định chuỗi
        nào là đáp án thuộc về tầng Q&A (VLM/LLM, Giai đoạn 2).
        """
        vid = str(video_id)
        fidx = int(frame_idx)
        ket_qua: list[tuple[int, str]] = []

        for doc in self._index:
            if doc["video_id"] != vid:
                continue
            khoang_cach = abs(int(doc["frame_idx"]) - fidx)
            if khoang_cach <= cua_so_frame:
                chu = doc["text"].strip()
                if chu:
                    ket_qua.append((khoang_cach, chu))

        ket_qua.sort(key=lambda x: x[0])
        return [chu for _, chu in ket_qua]
