import sys
import types
import unittest

import numpy as np


# ------------------------------------------------------------
# Cho phép test pure-Python ngay cả khi môi trường test
# không cài faiss-cpu.
# ------------------------------------------------------------

try:
    import faiss  # noqa: F401
except ModuleNotFoundError:
    sys.modules["faiss"] = types.ModuleType("faiss")


from scripts.profile_retrieval import (  # noqa: E402
    RRF_K,
    W_B,
    W_L,
    calculate_recall_at_k,
    fuse_video_rrf,
)


class TestVideoLevelRRF(unittest.TestCase):

    def test_video_nhieu_frame_chi_nhan_mot_contribution_moi_source(
        self,
    ):
        """
        Một video xuất hiện nhiều frame trong cùng source
        chỉ được nhận một contribution RRF.
        """

        b_ids = [
            ("B", 1),
            ("A", 1),
            ("A", 2),
            ("A", 3),
            ("A", 4),
            ("A", 5),
            ("A", 6),
            ("A", 7),
            ("A", 8),
            ("A", 9),
            ("A", 10),
        ]

        b_indices = np.arange(
            len(b_ids),
            dtype=np.int64,
        )

        b_scores = np.linspace(
            1.0,
            0.5,
            len(b_ids),
            dtype=np.float32,
        )

        ranked = fuse_video_rrf(
            b_indices=b_indices,
            b_scores=b_scores,
            b_ids=b_ids,
            l_indices=np.array(
                [],
                dtype=np.int64,
            ),
            l_scores=np.array(
                [],
                dtype=np.float32,
            ),
            l_ids=[],
        )

        by_video = dict(ranked)

        self.assertEqual(
            [video_id for video_id, _ in ranked],
            ["B", "A"],
        )

        self.assertAlmostEqual(
            by_video["B"]["rrf_score"],
            W_B / (RRF_K + 1),
        )

        self.assertAlmostEqual(
            by_video["A"]["rrf_score"],
            W_B / (RRF_K + 2),
        )

        self.assertEqual(
            by_video["B"]["best_b_rank"],
            1,
        )

        self.assertEqual(
            by_video["A"]["best_b_rank"],
            2,
        )

        # Toàn bộ frame của A vẫn phải được giữ.
        self.assertEqual(
            len(by_video["A"]["frames"]),
            10,
        )

    def test_moi_video_nhan_toi_da_mot_contribution_tu_moi_nhanh(
        self,
    ):
        """
        Một video có thể nhận:
            tối đa 1 contribution từ B
            +
            tối đa 1 contribution từ L
        """

        ranked = fuse_video_rrf(
            b_indices=np.array(
                [0, 1, 2],
                dtype=np.int64,
            ),
            b_scores=np.array(
                [0.9, 0.8, 0.7],
                dtype=np.float32,
            ),
            b_ids=[
                ("A", 1),
                ("A", 2),
                ("B", 1),
            ],
            l_indices=np.array(
                [0, 1, 2],
                dtype=np.int64,
            ),
            l_scores=np.array(
                [0.95, 0.85, 0.75],
                dtype=np.float32,
            ),
            l_ids=[
                ("B", 2),
                ("A", 3),
                ("A", 4),
            ],
        )

        by_video = dict(ranked)

        self.assertAlmostEqual(
            by_video["A"]["rrf_score"],
            W_B / (RRF_K + 1)
            + W_L / (RRF_K + 2),
        )

        self.assertAlmostEqual(
            by_video["B"]["rrf_score"],
            W_B / (RRF_K + 2)
            + W_L / (RRF_K + 1),
        )

        self.assertEqual(
            by_video["A"]["best_b_rank"],
            1,
        )

        self.assertEqual(
            by_video["A"]["best_l_rank"],
            2,
        )

        self.assertEqual(
            by_video["B"]["best_b_rank"],
            2,
        )

        self.assertEqual(
            by_video["B"]["best_l_rank"],
            1,
        )

        # A: B có 2 frame + L có 2 frame.
        self.assertEqual(
            len(by_video["A"]["frames"]),
            4,
        )

        # B: B có 1 frame + L có 1 frame.
        self.assertEqual(
            len(by_video["B"]["frames"]),
            2,
        )

    def test_vector_id_am_bi_bo_qua_va_khong_lam_lech_video_rank(
        self,
    ):
        """
        FAISS ID = -1 phải bị bỏ qua và không được làm
        tăng video rank.
        """

        ranked = fuse_video_rrf(
            b_indices=np.array(
                [-1, 0],
                dtype=np.int64,
            ),
            b_scores=np.array(
                [-1.0, 0.8],
                dtype=np.float32,
            ),
            b_ids=[
                ("A", 7),
            ],
            l_indices=np.array(
                [],
                dtype=np.int64,
            ),
            l_scores=np.array(
                [],
                dtype=np.float32,
            ),
            l_ids=[],
        )

        self.assertEqual(
            len(ranked),
            1,
        )

        video_id, info = ranked[0]

        self.assertEqual(
            video_id,
            "A",
        )

        # A phải là video rank 1,
        # không phải rank 2 vì ID -1.
        self.assertEqual(
            info["best_b_rank"],
            1,
        )

        self.assertAlmostEqual(
            info["rrf_score"],
            W_B / (RRF_K + 1),
        )

        # Nhưng frame rank gốc của A vẫn là 2.
        self.assertEqual(
            info["frames"][0]["rank"],
            2,
        )


class TestRecallAtK(unittest.TestCase):

    @staticmethod
    def _candidate(
        video_id,
        *frame_indices,
    ):
        return {
            "video_id": video_id,
            "frames": [
                {
                    "frame_idx": frame_idx
                }
                for frame_idx in frame_indices
            ],
        }

    def test_query_don_can_dung_video_va_frame(
        self,
    ):
        """
        Query đơn chỉ HIT khi:
            đúng video
            +
            đúng vùng frame.
        """

        retrieval = {
            "q1": [
                self._candidate(
                    "OTHER",
                    100,
                ),
                self._candidate(
                    "GT",
                    104,
                ),
            ]
        }

        ground_truths = {
            "q1": {
                "gt_video_id": "GT",
                "gt_frame_idx": 100,
                "raw_item": {
                    "loai_truy_van": "mo_ta",
                    "frame_start": 100,
                    "frame_end": 102,
                },
            }
        }

        # K=1: chưa có đúng video.
        self.assertEqual(
            calculate_recall_at_k(
                retrieval,
                ground_truths,
                1,
            ),
            0.0,
        )

        # K=2: đúng video + frame 104 nằm trong
        # tolerance 5 của GT range 100-102.
        self.assertEqual(
            calculate_recall_at_k(
                retrieval,
                ground_truths,
                2,
            ),
            1.0,
        )

    def test_trake_chi_hit_khi_tim_du_moi_event(
        self,
    ):
        """
        TRAKE chỉ HIT khi tìm đủ tất cả event.
        """

        ground_truths = {
            "q1": {
                "gt_video_id": "GT",
                "gt_frame_idx": 100,
                "raw_item": {
                    "loai_truy_van": "chuoi_su_kien",
                    "cac_giai_doan": [
                        {
                            "frame_start": 100,
                            "frame_end": 105,
                        },
                        {
                            "frame_start": 200,
                            "frame_end": 205,
                        },
                    ],
                },
            }
        }

        full = {
            "q1": [
                self._candidate(
                    "GT",
                    102,
                    203,
                )
            ]
        }

        partial = {
            "q1": [
                self._candidate(
                    "GT",
                    102,
                )
            ]
        }

        self.assertEqual(
            calculate_recall_at_k(
                full,
                ground_truths,
                1,
            ),
            1.0,
        )

        self.assertEqual(
            calculate_recall_at_k(
                partial,
                ground_truths,
                1,
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()