"""
Unit tests cho Task QA-R1: Multi-stage Retrieval Funnel.

Bao phủ các yêu cầu kỹ thuật:
1. Video-level Weighted RRF: Mỗi video chỉ nhận tối đa 1 contribution từ mỗi source.
2. Handling negative vector IDs (-1 từ FAISS).
3. Stage 1 Output: Top-300 Videos selection.
4. Stage 2 Output: Top-50 Diversity Frame Candidates (mỗi video 1 frame đại diện tốt nhất).
5. Stage 3 Output: Top-12 Reader Candidates.
6. Evaluation Recall: Phân biệt Video Recall@300 và Frame Recall@50/@12.
"""

import sys
import types
import unittest
import numpy as np

# Mock module faiss nếu môi trường test không cài đặt faiss-cpu
try:
    import faiss  # noqa: F401
except ModuleNotFoundError:
    sys.modules["faiss"] = types.ModuleType("faiss")

from scripts.candidate_funnel import (
    RRF_K,
    W_B,
    W_L,
    fuse_video_rrf,
    select_top300_videos,
    extract_top50_coverage_candidates,
    select_top12_reader_candidates,
    video_pool_hits_query,
    frame_candidates_hit_query,
)


class TestQAR1Stage1VideoFusion(unittest.TestCase):

    def test_video_deduplication_and_contribution(self):
        """Mỗi video chỉ được nhận 1 contribution RRF cho mỗi nguồn (CLIP-B / CLIP-L)."""
        b_ids = [("V_1", 1), ("V_1", 2), ("V_2", 1)]
        b_indices = np.array([0, 1, 2], dtype=np.int64)
        b_scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        l_ids = [("V_2", 2), ("V_1", 3)]
        l_indices = np.array([0, 1], dtype=np.int64)
        l_scores = np.array([0.95, 0.85], dtype=np.float32)

        ranked = fuse_video_rrf(
            b_indices=b_indices,
            b_scores=b_scores,
            b_ids=b_ids,
            l_indices=l_indices,
            l_scores=l_scores,
            l_ids=l_ids,
        )

        by_video = dict(ranked)

        # V_1: best B rank = 1, best L rank = 2
        self.assertAlmostEqual(
            by_video["V_1"]["rrf_score"],
            W_B / (RRF_K + 1) + W_L / (RRF_K + 2),
        )

        # V_2: best B rank = 2, best L rank = 1
        self.assertAlmostEqual(
            by_video["V_2"]["rrf_score"],
            W_B / (RRF_K + 2) + W_L / (RRF_K + 1),
        )

        # V_1 gom đủ 3 frames (2 từ B, 1 từ L)
        self.assertEqual(len(by_video["V_1"]["frames"]), 3)

    def test_faiss_negative_ids_ignored(self):
        """FAISS trả về ID negative (-1) phải bị bỏ qua và không tăng video rank."""
        b_ids = [("V_1", 10)]
        b_indices = np.array([-1, 0], dtype=np.int64)
        b_scores = np.array([-1.0, 0.88], dtype=np.float32)

        ranked = fuse_video_rrf(
            b_indices=b_indices,
            b_scores=b_scores,
            b_ids=b_ids,
            l_indices=np.array([], dtype=np.int64),
            l_scores=np.array([], dtype=np.float32),
            l_ids=[],
        )

        self.assertEqual(len(ranked), 1)
        video_id, info = ranked[0]
        self.assertEqual(video_id, "V_1")
        self.assertEqual(info["best_b_rank"], 1)

    def test_select_top300_videos_structure(self):
        """Đảm bảo Top-300 giữ cấu trúc Video Candidate Pool."""
        mock_ranked = [
            (f"V_{i}", {"rrf_score": 1.0 / (i + 1), "best_b_rank": i + 1, "best_l_rank": None, "frames": []})
            for i in range(350)
        ]

        top300 = select_top300_videos(mock_ranked)
        self.assertEqual(len(top300), 300)
        self.assertEqual(top300[0]["video_id"], "V_0")
        self.assertEqual(top300[0]["rank_video"], 1)


class TestQAR1Stage2And3Funnel(unittest.TestCase):

    def setUp(self):
        # Frame lookup giả lập cho 60 videos
        self.frame_lookup = {}
        self.mock_top300_videos = []

        for i in range(60):
            v_id = f"V_{i}"
            n_1, n_2 = 1, 2
            self.frame_lookup[(v_id, n_1)] = {"frame_idx": i * 100 + 10, "pts_time": i * 4.0}
            self.frame_lookup[(v_id, n_2)] = {"frame_idx": i * 100 + 20, "pts_time": i * 4.0 + 0.4}

            self.mock_top300_videos.append(
                {
                    "video_id": v_id,
                    "rank_video": i + 1,
                    "score_fused": 1.0 / (i + 1),
                    "frames": [
                        {"n": n_1, "source": "clip_b", "source_rank": 2, "source_score": 0.8},
                        {"n": n_2, "source": "clip_b", "source_rank": 1, "source_score": 0.9}, # best frame
                    ],
                }
            )

    def test_stage2_extract_top50_coverage_diversity(self):
        """Stage 2 phải lọc đúng 50 frame đại diện từ 50 video khác nhau."""
        candidates_50 = extract_top50_coverage_candidates(
            self.mock_top300_videos, self.frame_lookup
        )

        # Đúng 50 candidate
        self.assertEqual(len(candidates_50), 50)

        # Tất cả video_id đều duy nhất (Deduplication thành công)
        video_ids = [c["video_id"] for c in candidates_50]
        self.assertEqual(len(set(video_ids)), 50)

        # Chọn đúng frame đại diện n=2 có source_rank tốt nhất (rank 1 vs rank 2)
        self.assertEqual(candidates_50[0]["n"], 2)
        self.assertEqual(candidates_50[0]["stage"], "coverage")
        self.assertEqual(candidates_50[0]["rank_final"], 1)

    def test_stage3_select_top12_reader(self):
        """Stage 3 phải bóp phễu lấy đúng Top 12 candidate từ Stage 2."""
        candidates_50 = extract_top50_coverage_candidates(
            self.mock_top300_videos, self.frame_lookup
        )
        candidates_12 = select_top12_reader_candidates(candidates_50)

        self.assertEqual(len(candidates_12), 12)
        self.assertEqual(candidates_12[0]["stage"], "reader")
        self.assertEqual(candidates_12[0]["rank_final"], 1)
        self.assertEqual(candidates_12[11]["rank_final"], 12)


class TestQAR1RecallEvaluation(unittest.TestCase):

    def test_video_recall_300(self):
        """Video Recall@300 chỉ cần khớp video_id."""
        gt = {"gt_video_id": "V_Target", "gt_frame_idx": 500, "raw_item": {"loai_truy_van": "mo_ta"}}
        videos_hit = [{"video_id": "V_Other"}, {"video_id": "V_Target"}]
        videos_miss = [{"video_id": "V_Other"}]

        self.assertTrue(video_pool_hits_query(videos_hit, gt))
        self.assertFalse(video_pool_hits_query(videos_miss, gt))

    def test_frame_recall_single_query(self):
        """Frame Recall đòi hỏi khớp video_id VÀ nằm trong vùng frame_idx tolerance."""
        gt = {
            "gt_video_id": "V_Target",
            "gt_frame_idx": 100,
            "raw_item": {"loai_truy_van": "mo_ta", "frame_start": 100, "frame_end": 105},
        }

        # Frame 108 nằm trong tolerance (100 - 5 <= 108 <= 105 + 5)
        candidates_hit = [{"video_id": "V_Target", "frame_idx": 108}]
        # Frame 130 quá xa tolerance
        candidates_miss = [{"video_id": "V_Target", "frame_idx": 130}]

        self.assertTrue(frame_candidates_hit_query(candidates_hit, gt))
        self.assertFalse(frame_candidates_hit_query(candidates_miss, gt))

    def test_frame_recall_trake_query(self):
        """TRAKE query yêu cầu tìm đủ TẤT CẢ các event."""
        gt = {
            "gt_video_id": "V_Target",
            "gt_frame_idx": 100,
            "raw_item": {
                "loai_truy_van": "chuoi_su_kien",
                "cac_giai_doan": [
                    {"frame_start": 100, "frame_end": 105},
                    {"frame_start": 200, "frame_end": 205},
                ],
            },
        }

        # Tìm đủ 2 event
        full_hit = [
            {"video_id": "V_Target", "frame_idx": 102},
            {"video_id": "V_Target", "frame_idx": 203},
        ]

        # Thiếu event 2
        partial_hit = [
            {"video_id": "V_Target", "frame_idx": 102},
        ]

        self.assertTrue(frame_candidates_hit_query(full_hit, gt))
        self.assertFalse(frame_candidates_hit_query(partial_hit, gt))


if __name__ == "__main__":
    unittest.main()