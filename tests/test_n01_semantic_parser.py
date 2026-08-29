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


if __name__ == "__main__":
    unittest.main()
