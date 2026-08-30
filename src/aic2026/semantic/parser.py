import math
import re
from typing import List, Dict
from aic2026.semantic.schema import QueryPlanQA, QueryPlanTRAKE, TrakeEvent

class RuleBasedParser:
    def __init__(self):
        self.qa_rules = [
            # OCR / Text Reading
            {
                "pattern": r"(con số.*ghi trên.*bao nhiêu|tên gì|tên là gì|tiêu đề|thương hiệu|ghi gì|biển báo|câu thơ|tên của|mang tên|xu hướng gì)",
                "intent": "read_text",
                "answer_type": "short_text",
                "modalities": {"clip_l": 0.2, "ocr": 0.7, "asr": 0.1, "caption": 0.0}
            },
            # Counting
            {
                "pattern": r"(bao nhiêu|mấy)",
                "intent": "count_objects",
                "answer_type": "number",
                "modalities": {"clip_l": 0.7, "ocr": 0.1, "asr": 0.0, "caption": 0.2}
            },
            # Speech / Reason
            {
                "pattern": r"(nói gì|theo lời|lý do)",
                "intent": "read_speech",
                "answer_type": "short_text",
                "modalities": {"clip_l": 0.1, "ocr": 0.0, "asr": 0.8, "caption": 0.1}
            },
            # Attribute
            {
                "pattern": r"(màu gì|màu chủ đạo)",
                "intent": "identify_attribute",
                "answer_type": "short_text",
                "modalities": {"clip_l": 0.7, "ocr": 0.0, "asr": 0.0, "caption": 0.3}
            },
            # Object / Action
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
        clean_entity = raw_entity.strip()
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

        if match := re.search(r"\bbao nhiêu\s+(.+?)(?=\s+(?:đang|có)\b|[?.!]|$)", query):
            self._append_unique(entities, self._clean_entity(match.group(1)))

        if match := re.search(r"\bmón\s+ăn\s+có\s+(.+?)\s+được\s+(?:cầm|cắt|ướp|chiên)\b", query):
            self._append_unique(entities, self._clean_entity(match.group(1)))

        for match in re.finditer(r"\b(cầm|cắt|ướp|chiên|ủng hộ)\s+([^?.!]*)", query):
            action = match.group(1)
            self._append_unique(actions, action)
            clean_entity = self._clean_entity(match.group(2))
            if action == "chiên" and re.match(r"^(?:giòn|chín|vàng|xém|sơ)\b", clean_entity):
                clean_entity = ""
            self._append_unique(entities, clean_entity)

        if match := re.search(r"(áp phích|túi mini|con đèo|biển báo|xã|công thức|gói bột|động đất|ủng hộ ai)", query, re.IGNORECASE):
            raw_ent = match.group(1)
            clean_ent = re.sub(r" ai$", "", raw_ent).strip() 
            if not any(clean_ent in e for e in entities):
                self._append_unique(entities, clean_ent)

        return [e for e in entities if e], attributes, actions

    def _generate_expanded_queries(self, query_text: str, entities: List[str], actions: List[str]) -> Dict[str, List[str]]:
        queries = {
            "clip_l": [query_text],
            "ocr": [query_text],
            "asr": [query_text]
        }
        
        core_phrases = []
        if entities and actions:
            for action in actions:
                for entity in entities:
                    core_phrases.append(f"{action} {entity}")
        elif entities:
            core_phrases.extend(entities)
        elif actions:
            core_phrases.extend(actions)
            
        en_dict = {
            "ướp": "marinate", "thịt ếch": "frog", "chiên": "fried", "cua": "crab",
            "cầm": "holding", "công thức": "recipe", "áo đen": "black shirt",
            "màu đỏ": "red", "hành tây": "onion", "cắt": "cut", "biển báo": "sign"
        }
        
        for phrase in core_phrases:
            if phrase not in queries["clip_l"] and len(queries["clip_l"]) < 5:
                queries["clip_l"].append(phrase)
                en_words = [en_dict[w] for w in en_dict if w in phrase.lower()]
                if en_words:
                    en_variant = " ".join(en_words)
                    if en_variant not in queries["clip_l"] and len(queries["clip_l"]) < 5:
                        queries["clip_l"].append(en_variant)
                        
        ocr_keywords = [w for w in ["tên", "tiêu đề", "thương hiệu", "con số"] if w in query_text.lower()]
        
        # Ưu tiên thêm keyword + entity vào nhóm OCR trước
        if ocr_keywords and entities:
            for kw in ocr_keywords:
                for ent in entities:
                    ocr_q = f"{kw} {ent}"
                    if ocr_q not in queries["ocr"] and len(queries["ocr"]) < 5:
                        queries["ocr"].append(ocr_q)

        # Sau đó mới thêm core phrases 
        for phrase in core_phrases:
            if phrase not in queries["ocr"] and len(queries["ocr"]) < 5:
                queries["ocr"].append(phrase)

        clean_text = query_text.strip(" \t\r\n,.;:!?")
        asr_variant = re.sub(r"^(Trong đoạn video|Đoạn phim ghi lại cảnh|Hỏi)\s+", "", clean_text, flags=re.IGNORECASE).strip()
        
        if asr_variant and asr_variant != query_text and asr_variant.lower() != clean_text.lower():
            if asr_variant not in queries["asr"] and len(queries["asr"]) < 5:
                queries["asr"].append(asr_variant)
        elif clean_text != query_text and len(queries["asr"]) < 5:
            queries["asr"].append(clean_text)
            
        return queries

    def _calculate_uncertainty(self, modalities: Dict[str, float]) -> float:
        M = len(modalities)
        if M <= 1:
            return 0.0
        
        entropy = 0.0
        for weight in modalities.values():
            if weight > 0:
                entropy -= weight * math.log2(weight)
                
        max_entropy = math.log2(M)
        normalized_uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return round(normalized_uncertainty, 4)

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
                
        if "trước" in query_lower:
            temporal_rel = "before"
        elif "sau" in query_lower or "tiếp theo" in query_lower:
            temporal_rel = "after"
        else:
            temporal_rel = "none"
            
        entities, attributes, actions = self._extract_entities_actions(query_lower)
        
        expanded_queries = self._generate_expanded_queries(query_text, entities, actions)
        uncertainty_score = self._calculate_uncertainty(matched_modalities)

        return QueryPlanQA(
            query_id=query_id, 
            task="qa", 
            query_text=query_text,
            intent=matched_intent, 
            answer_type=matched_answer_type,
            entities=entities, 
            attributes=attributes, 
            actions=actions,
            temporal_relation=temporal_rel, 
            preferred_modalities=matched_modalities,
            queries=expanded_queries,
            uncertainty=uncertainty_score
        )

    def parse_trake(self, query_id: str, events: List[str]) -> QueryPlanTRAKE:
        trake_events = [
            TrakeEvent(event_id=f"E{i+1}", text=text, relation="start" if i == 0 else f"after:E{i}")
            for i, text in enumerate(events)
        ]
        return QueryPlanTRAKE(query_id=query_id, task="trake", events=trake_events)
    