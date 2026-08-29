import re
from typing import List
from aic2026.semantic.schema import QueryPlanQA, QueryPlanTRAKE, TrakeEvent


class RuleBasedParser:
    def __init__(self):
        self.qa_rules = [
            # OCR / Text Reading (Ưu tiên cao nhất để không bị đè bởi các rule chung)
            {
                "pattern": r"(con số.*ghi trên.*bao nhiêu|tên gì|tên là gì|tiêu đề|thương hiệu|ghi gì|biển báo|câu thơ|tên của|mang tên|xu hướng gì)",
                "intent": "read_text",
                "answer_type": "short_text",
                "modalities": {"clip_l": 0.2, "ocr": 0.7, "asr": 0.1, "caption": 0.0}
            },
            # Counting (Đếm)
            {
                "pattern": r"(bao nhiêu|mấy)",
                "intent": "count_objects",
                "answer_type": "number",
                # Câu đếm vẫn là truy vấn thị giác. Giữ OCR ở mức phụ trợ
                # nhưng chuẩn hóa tổng trọng số về 1.0 giống các rule khác.
                "modalities": {"clip_l": 0.7, "ocr": 0.1, "asr": 0.0, "caption": 0.2}
            },
            # Speech / Reason
            {
                "pattern": r"(nói gì|theo lời|lý do)",
                "intent": "read_speech",
                "answer_type": "short_text",
                "modalities": {"clip_l": 0.1, "ocr": 0.0, "asr": 0.8, "caption": 0.1}
            },
            # Attribute (Màu sắc, tính chất)
            {
                "pattern": r"(màu gì|màu chủ đạo)",
                "intent": "identify_attribute",
                "answer_type": "short_text",
                "modalities": {"clip_l": 0.7, "ocr": 0.0, "asr": 0.0, "caption": 0.3}
            },
            # Object / Action (Vật, Hành động, Nguyên liệu)
            {
                "pattern": r"(vật gì|hành động|làm gì|cầm gì|cắt thế nào|nguyên liệu gì|là món gì|ủng hộ ai)",
                "intent": "identify_object_action",
                "answer_type": "short_text",
                "modalities": {"clip_l": 0.7, "ocr": 0.0, "asr": 0.1, "caption": 0.2}
            },
            # Temporal
            {
                "pattern": r"(trước khi|sau khi|tiếp theo)",
                "intent": "temporal_reasoning",
                "answer_type": "short_text",
                "modalities": {"clip_l": 0.6, "ocr": 0.1, "asr": 0.1, "caption": 0.2}
            }
        ]

    @staticmethod
    def _clean_entity(raw_entity: str) -> str:
        """Loại phần đuôi câu hỏi khỏi một cụm thực thể ứng viên.

        Hàm chỉ làm sạch theo cấu trúc ngôn ngữ, không phụ thuộc query_id.
        Khi không chắc chắn, trả về chuỗi rỗng tốt hơn việc đưa cả câu hỏi
        vào ``entities`` và làm nhiễu truy vấn ở tầng sau.
        """
        clean_entity = raw_entity.strip()

        # Không để regex hành động ăn sang câu kế tiếp hoặc phần kết luận
        # kiểu "— đây là món gì".
        clean_entity = re.split(r"[.!?]|\s+[—–-]\s+", clean_entity, maxsplit=1)[0]

        garbage_patterns = [
            r"\s+với\s+những\s+nguyên\s+liệu\s+gì\b.*$",
            r"\s+với\s+nguyên\s+liệu\b.*$",
            r"(?:^|\s+)thế\s+nào\b.*$",
            r"(?:^|\s+)là\s+(?:món\s+)?gì\b.*$",
            r"(?:^|\s+)gì\b.*$",
            r"(?:^|\s+)hỏi\s+.*$",
        ]
        for pattern in garbage_patterns:
            clean_entity = re.sub(pattern, "", clean_entity).strip()

        return clean_entity.strip(" \t\r\n,.;:!?—–-")

    @staticmethod
    def _append_unique(values: List[str], value: str) -> None:
        if value and value not in values:
            values.append(value)

    def _extract_entities_actions(self, query: str):
        entities, attributes, actions = [], [], []
        if re.search(r"(màu)", query):
            attributes.append("màu sắc")

        # Đếm: dừng trước mệnh đề bổ nghĩa và luôn bỏ dấu câu cuối.
        if match := re.search(
            r"\bbao nhiêu\s+(.+?)(?=\s+(?:đang|có)\b|[?.!]|$)", query
        ):
            self._append_unique(entities, self._clean_entity(match.group(1)))

        # Mẫu nhận diện món ăn dạng bị động. Quy tắc này tránh hiểu trạng
        # thái "chiên giòn" là một thực thể.
        if match := re.search(
            r"\bmón\s+ăn\s+có\s+(.+?)\s+được\s+(?:cầm|cắt|ướp|chiên)\b",
            query,
        ):
            self._append_unique(entities, self._clean_entity(match.group(1)))

        for match in re.finditer(r"\b(cầm|cắt|ướp|chiên)\s+([^?.!]*)", query):
            action = match.group(1)
            self._append_unique(actions, action)

            clean_entity = self._clean_entity(match.group(2))

            # Sau "chiên", các từ này mô tả trạng thái/cách chế biến chứ
            # không phải vật thể. Thực thể bị động (nếu có) đã được lấy bởi
            # rule phía trên.
            if action == "chiên" and re.match(
                r"^(?:giòn|chín|vàng|xém|sơ)\b", clean_entity
            ):
                clean_entity = ""

            self._append_unique(entities, clean_entity)

        return [e for e in entities if e], attributes, actions

    def parse_qa(self, query_id: str, query_text: str) -> QueryPlanQA:
        query_lower = query_text.lower()
        matched_intent, matched_answer_type = "general_qa", "short_text"
        matched_modalities = {"clip_l": 0.7, "ocr": 0.1, "asr": 0.1, "caption": 0.1}
        
        for rule in self.qa_rules:
            if re.search(rule["pattern"], query_lower):
                matched_intent = rule["intent"]
                matched_answer_type = rule["answer_type"]
                matched_modalities = rule["modalities"]
                break
                
        temporal_rel = "before" if "trước" in query_lower else "after" if "sau" in query_lower else "none"
        entities, attributes, actions = self._extract_entities_actions(query_lower)

        return QueryPlanQA(
            query_id=query_id, task="qa", query_text=query_text,
            intent=matched_intent, answer_type=matched_answer_type,
            entities=entities, attributes=attributes, actions=actions,
            temporal_relation=temporal_rel, preferred_modalities=matched_modalities,
            queries={"clip_l": [query_text], "ocr": [query_text], "asr": [query_text]}
        )

    def parse_trake(self, query_id: str, events: List[str]) -> QueryPlanTRAKE:
        trake_events = [
            TrakeEvent(event_id=f"E{i+1}", text=text, relation="start" if i == 0 else f"after:E{i}")
            for i, text in enumerate(events)
        ]
        return QueryPlanTRAKE(query_id=query_id, task="trake", events=trake_events)
