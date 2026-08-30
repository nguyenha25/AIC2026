import unittest

from aic2026.semantic.parser import RuleBasedParser

class TestN01SemanticParser(unittest.TestCase):
    def setUp(self):
        self.parser = RuleBasedParser()

    def test_mon_an_bi_dong_khong_lay_trang_thai_lam_entity(self):
        plan = self.parser.parse_qa(
            "06",
            (
                "Món ăn có cua được chiên giòn, phủ xốt sệt màu cam đỏ, "
                "rắc ớt sợi, bày trên đĩa cùng rau mầm và cà chua bi — "
                "đây là món gì?"
            ),
        )

        self.assertEqual(plan.intent, "identify_object_action")
        self.assertEqual(plan.entities, ["cua"])
        self.assertEqual(plan.actions, ["chiên"])

    def test_entity_cau_dem_duoc_bo_dau_cau_cuoi(self):
        plan = self.parser.parse_qa(
            "44",
            "Không tính bảng chú giải, có bao nhiêu vị trí ghi nhận động đất cấp độ 4?",
        )

        self.assertEqual(plan.intent, "count_objects")
        self.assertEqual(plan.entities, ["vị trí ghi nhận động đất cấp độ 4"])

    def test_entity_hanh_dong_khong_an_sang_cau_hoi_tiep_theo(self):
        plan = self.parser.parse_qa(
            "72",
            (
                "Trong đoạn video có thể thấy một người đang cầm công thức món ăn "
                "với nguyên liệu chính là 200g thịt nạc xay. "
                "Hỏi tiêu đề của công thức nấu ăn này là gì?"
            ),
        )

        self.assertEqual(plan.intent, "read_text")
        self.assertEqual(plan.entities, ["công thức món ăn"])
        self.assertEqual(plan.actions, ["cầm"])

    def test_entity_nguyen_lieu_cu_giu_nguyen(self):
        plan = self.parser.parse_qa(
            "04",
            "Trong món Lẩu ếch lá lốt, ướp thịt ếch với những nguyên liệu gì?",
        )

        self.assertEqual(plan.entities, ["thịt ếch"])
        self.assertEqual(plan.actions, ["ướp"])

    def test_tu_de_hoi_khong_bi_xem_la_entity(self):
        for query in (
            "Người đàn ông đang cầm gì?",
            "Hành tây được cắt thế nào?",
        ):
            with self.subTest(query=query):
                plan = self.parser.parse_qa("x", query)
                self.assertEqual(plan.entities, [])

    def test_moi_rule_co_tong_trong_so_modality_bang_mot(self):
        for rule in self.parser.qa_rules:
            with self.subTest(intent=rule["intent"]):
                self.assertAlmostEqual(sum(rule["modalities"].values()), 1.0)

    def test_trake_giu_quan_he_tuan_tu(self):
        plan = self.parser.parse_trake(
            "07", ["Sự kiện một", "Sự kiện hai", "Sự kiện ba"]
        )

        self.assertEqual(
            [event.event_id for event in plan.events], ["E1", "E2", "E3"]
        )
        self.assertEqual(
            [event.relation for event in plan.events],
            ["start", "after:E1", "after:E2"],
        )

    def test_calculate_uncertainty_entropy(self):
        plan = self.parser.parse_qa("1", "Câu hỏi chung chung?")
        self.assertGreater(plan.uncertainty, 0.0)
        self.assertLess(plan.uncertainty, 1.0)
        
        zero_uncertainty = self.parser._calculate_uncertainty({"clip_l": 1.0, "ocr": 0.0, "asr": 0.0, "caption": 0.0})
        self.assertEqual(zero_uncertainty, 0.0)

    def test_routing_intent_accuracy_for_gate_1(self):
        test_cases = [
            ("Biển báo ghi gì?", "read_text"),
            ("Người đàn ông nói gì?", "read_speech"),
            ("Có bao nhiêu người?", "count_objects"),
            ("Chiếc áo màu gì?", "identify_attribute"),
            ("Người đó đang cầm gì?", "identify_object_action"),
            ("Tên của con đèo là gì?", "read_text")
        ]
        correct = 0
        for query, expected_intent in test_cases:
            plan = self.parser.parse_qa("x", query)
            if plan.intent == expected_intent:
                correct += 1
        
        accuracy = correct / len(test_cases)
        self.assertGreaterEqual(accuracy, 0.90)

    def test_temporal_tiep_theo(self):
        plan = self.parser.parse_qa("1", "Sự kiện tiếp theo là gì?")
        self.assertEqual(plan.intent, "temporal_reasoning")
        self.assertEqual(plan.temporal_relation, "after")

    def test_modality_routing_dominance(self):
        plan = self.parser.parse_qa("1", "Biển báo ghi gì?")
        self.assertEqual(plan.intent, "read_text")
        self.assertEqual(max(plan.preferred_modalities, key=plan.preferred_modalities.get), "ocr")

        plan = self.parser.parse_qa("2", "Có bao nhiêu người?")
        self.assertEqual(plan.intent, "count_objects")
        self.assertEqual(max(plan.preferred_modalities, key=plan.preferred_modalities.get), "clip_l")

        plan = self.parser.parse_qa("3", "Người đàn ông nói gì?")
        self.assertEqual(plan.intent, "read_speech")
        self.assertEqual(max(plan.preferred_modalities, key=plan.preferred_modalities.get), "asr")
        
        plan = self.parser.parse_qa("4", "Chiếc áo màu gì?")
        self.assertEqual(plan.intent, "identify_attribute")
        self.assertEqual(max(plan.preferred_modalities, key=plan.preferred_modalities.get), "clip_l")

    def test_query_expansion_generates_meaningful_variants(self):
        plan_04 = self.parser.parse_qa(
            "04",
            "Trong món Lẩu ếch lá lốt, ướp thịt ếch với những nguyên liệu gì?",
        )
        self.assertIn("marinate frog", plan_04.queries["clip_l"])
        self.assertTrue(any(q == "Trong món Lẩu ếch lá lốt, ướp thịt ếch với những nguyên liệu gì" for q in plan_04.queries["asr"]))
        
        plan_05 = self.parser.parse_qa(
            "05",
            "Gói bột mang tên gì đang được đâu bếp đổ vào tô để tẩm ướp cá cơm?"
        )
        self.assertIn("tên gói bột", plan_05.queries["ocr"])

if __name__ == "__main__":
    unittest.main()
    