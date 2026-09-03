"""
Unit tests cho Task QA-R1: Multi-stage Retrieval Funnel.

Bao phủ:

1. Video-level Weighted RRF.
2. Mỗi video chỉ nhận tối đa 1 contribution / source.
3. FAISS negative vector IDs (-1) bị bỏ qua.
4. FAISS ID vượt mapping bị reject.
5. Deterministic tie-breaking.
6. Stage 1 Top-300 Video Pool.
7. Stage 2 Top-50 Frame Coverage.
8. Temporal diversity cho query thường.
9. TRAKE giữ nhiều thời điểm trong cùng video.
10. Stage 3 Top-12 Reader.
11. Multi-source provenance.
12. query_id preservation.
13. Missing frame_map fallback.
14. Video Recall@300.
15. Frame Recall@50/@12.
16. TRAKE recall yêu cầu đủ tất cả events.
17. Query embedding <-> query_id alignment.
18. Alignment reject duplicate IDs.
19. Alignment reject missing IDs.
20. Alignment reject extra IDs.
21. Alignment reject row-count mismatch.
22. Alignment reject dimension mismatch.
23. Quality Gate reject zero recall.
24. Quality Gate baseline regression.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


# ============================================================
# MOCK FAISS IF NOT INSTALLED
# ============================================================

try:
    import faiss  # noqa: F401
except ModuleNotFoundError:
    sys.modules["faiss"] = types.ModuleType("faiss")


import scripts.candidate_funnel as candidate_funnel

from scripts.candidate_funnel import (
    RRF_K,
    W_B,
    W_L,
    GroundTruth,
    SourceFrame,
    SourceEvidence,
    MergedFrame,
    VideoCandidate,
    FrameCandidate,
    fuse_video_rrf,
    merge_frame_provenance,
    calculate_frame_evidence,
    build_frame_candidate,
    extract_top50_coverage,
    select_top12_reader,
    frame_candidates_hit,
    video_pool_hit,
    evaluate_quality_gate,
    validate_embedding_alignment,
    build_embedding_row_map,
    calculate_semantic_coverage,
    derive_query_routing,
    frame_to_json,
    run_single_query_benchmark,
)


# ============================================================
# HELPERS
# ============================================================

def make_gt(
    query_id: str = "Q_1",
    video_id: str = "V_Target",
    query_type: str = "mo_ta",
    frame_start: int = 100,
    frame_end: int = 105,
) -> GroundTruth:
    return GroundTruth(
        query_id=query_id,
        video_id=video_id,
        query_type=query_type,
        frame_tolerance=5,
        raw_item={
            "id": query_id,
            "video_id": video_id,
            "loai_truy_van": query_type,
            "frame_start": frame_start,
            "frame_end": frame_end,
        },
    )


def make_video(
    video_id: str,
    rank_video: int,
    score_fused: float = 0.02,
    frames: list[SourceFrame] | None = None,
) -> VideoCandidate:
    return VideoCandidate(
        video_id=video_id,
        rrf_score=score_fused,
        best_b_rank=rank_video,
        best_l_rank=None,
        frames=frames or [],
    )


# ============================================================
# STAGE 1 — VIDEO RRF
# ============================================================

class TestQAR1Stage1VideoFusion(unittest.TestCase):

    def test_video_deduplication_and_contribution(self):
        """
        Mỗi video chỉ nhận 1 contribution RRF từ mỗi source.

        V_1:
            CLIP-B rank 1
            CLIP-L rank 2

        V_2:
            CLIP-B rank 2
            CLIP-L rank 1
        """

        b_ids = [
            ("V_1", 1),
            ("V_1", 2),
            ("V_2", 1),
        ]

        b_indices = np.array(
            [0, 1, 2],
            dtype=np.int64,
        )

        b_scores = np.array(
            [0.9, 0.8, 0.7],
            dtype=np.float32,
        )

        l_ids = [
            ("V_2", 2),
            ("V_1", 3),
        ]

        l_indices = np.array(
            [0, 1],
            dtype=np.int64,
        )

        l_scores = np.array(
            [0.95, 0.85],
            dtype=np.float32,
        )

        ranked = fuse_video_rrf(
            b_indices=b_indices,
            b_scores=b_scores,
            b_ids=b_ids,
            l_indices=l_indices,
            l_scores=l_scores,
            l_ids=l_ids,
        )

        by_video = {
            candidate.video_id: candidate
            for candidate in ranked
        }

        expected_v1 = (
            W_B / (RRF_K + 1)
            + W_L / (RRF_K + 2)
        )

        expected_v2 = (
            W_B / (RRF_K + 2)
            + W_L / (RRF_K + 1)
        )

        self.assertAlmostEqual(
            by_video["V_1"].rrf_score,
            expected_v1,
        )

        self.assertAlmostEqual(
            by_video["V_2"].rrf_score,
            expected_v2,
        )

        # V_1 có 3 frame provenance:
        # B frame 1, B frame 2, L frame 3.
        self.assertEqual(
            len(by_video["V_1"].frames),
            3,
        )

    def test_each_source_contributes_once_per_video(self):
        """
        Một video xuất hiện nhiều frame trong cùng source
        vẫn chỉ được cộng RRF một lần.
        """

        ids = [
            ("V_1", 1),
            ("V_1", 2),
            ("V_1", 3),
        ]

        indices = np.array(
            [0, 1, 2],
            dtype=np.int64,
        )

        scores = np.array(
            [0.9, 0.8, 0.7],
            dtype=np.float32,
        )

        ranked = fuse_video_rrf(
            b_indices=indices,
            b_scores=scores,
            b_ids=ids,
            l_indices=np.array([], dtype=np.int64),
            l_scores=np.array([], dtype=np.float32),
            l_ids=[],
        )

        self.assertEqual(
            len(ranked),
            1,
        )

        candidate = ranked[0]

        self.assertAlmostEqual(
            candidate.rrf_score,
            W_B / (RRF_K + 1),
        )

        self.assertEqual(
            candidate.best_b_rank,
            1,
        )

    def test_faiss_negative_ids_are_ignored(self):
        """
        FAISS có thể trả -1 khi không có đủ kết quả.
        ID này phải bị bỏ qua hoàn toàn.
        """

        b_ids = [
            ("V_1", 10),
        ]

        b_indices = np.array(
            [-1, 0],
            dtype=np.int64,
        )

        b_scores = np.array(
            [-1.0, 0.88],
            dtype=np.float32,
        )

        ranked = fuse_video_rrf(
            b_indices=b_indices,
            b_scores=b_scores,
            b_ids=b_ids,
            l_indices=np.array([], dtype=np.int64),
            l_scores=np.array([], dtype=np.float32),
            l_ids=[],
        )

        self.assertEqual(
            len(ranked),
            1,
        )

        candidate = ranked[0]

        self.assertEqual(
            candidate.video_id,
            "V_1",
        )

        self.assertEqual(
            candidate.best_b_rank,
            1,
        )

    def test_faiss_id_out_of_range_is_rejected(self):
        """
        FAISS ID vượt mapping phải raise thay vì silently sai dữ liệu.
        """

        b_ids = [
            ("V_1", 1),
        ]

        b_indices = np.array(
            [5],
            dtype=np.int64,
        )

        b_scores = np.array(
            [0.9],
            dtype=np.float32,
        )

        with self.assertRaises(IndexError):
            fuse_video_rrf(
                b_indices=b_indices,
                b_scores=b_scores,
                b_ids=b_ids,
                l_indices=np.array([], dtype=np.int64),
                l_scores=np.array([], dtype=np.float32),
                l_ids=[],
            )

    def test_deterministic_tie_breaking(self):
        """
        Nếu RRF score bằng nhau thì video_id phải quyết định thứ tự.

        Test này cố ý tạo một tình huống RRF tie thật sự bằng cách
        tạm thời dùng trọng số hai source bằng nhau.

        Implementation production KHÔNG bị thay đổi.
        """

        b_ids = [
            ("V_B", 1),
            ("V_A", 2),
        ]

        b_indices = np.array(
            [0, 1],
            dtype=np.int64,
        )

        b_scores = np.array(
            [0.9, 0.8],
            dtype=np.float32,
        )

        l_ids = [
            ("V_A", 1),
            ("V_B", 2),
        ]

        l_indices = np.array(
            [0, 1],
            dtype=np.int64,
        )

        l_scores = np.array(
            [0.95, 0.85],
            dtype=np.float32,
        )

        original_w_b = candidate_funnel.W_B
        original_w_l = candidate_funnel.W_L

        try:
            # Với W_B = W_L = 1:
            #
            # V_A = 1/(60+2) + 1/(60+1)
            # V_B = 1/(60+1) + 1/(60+2)
            #
            # => score bằng nhau.
            candidate_funnel.W_B = 1.0
            candidate_funnel.W_L = 1.0

            ranked = fuse_video_rrf(
                b_indices=b_indices,
                b_scores=b_scores,
                b_ids=b_ids,
                l_indices=l_indices,
                l_scores=l_scores,
                l_ids=l_ids,
            )

        finally:
            candidate_funnel.W_B = original_w_b
            candidate_funnel.W_L = original_w_l

        self.assertEqual(
            [x.video_id for x in ranked],
            ["V_A", "V_B"],
        )


# ============================================================
# STAGE 2 — PROVENANCE
# ============================================================

class TestQAR1Stage2Provenance(unittest.TestCase):

    def test_query_plan_changes_clip_l_weight_and_candidate_budget(self):
        routing = derive_query_routing(
            {
                "schema_version": "1.1",
                "preferred_modalities": {
                    "clip_l": 0.7,
                    "ocr": 0.1,
                    "asr": 0.0,
                    "caption": 0.2,
                },
                "semantic_k_hint": 500,
            }
        )

        self.assertEqual(routing["query_plan_schema_version"], "1.1")
        self.assertEqual(routing["semantic_k_hint_applied"], 500)
        self.assertEqual(routing["routing_weights_applied"]["clip_l"], 0.7)
        self.assertEqual(routing["unavailable_modalities"], ["caption", "ocr"])

        low_prior = derive_query_routing(
            {
                "schema_version": "1.1",
                "preferred_modalities": {"clip_l": 0.1, "asr": 0.9},
                "semantic_k_hint": 100,
            }
        )
        self.assertEqual(low_prior["semantic_k_hint_applied"], 300)
        self.assertEqual(low_prior["routing_weights_applied"]["clip_l"], W_L)

    def test_semantic_coverage_uses_real_source_provenance(self):
        modalities = {
            "clip_l": 0.2,
            "ocr": 0.7,
            "asr": 0.1,
            "caption": 0.0,
        }

        self.assertEqual(
            calculate_semantic_coverage({"clip_b"}, modalities),
            0.2,
        )
        self.assertEqual(
            calculate_semantic_coverage({"clip_l", "ocr"}, modalities),
            0.9,
        )

    def test_merge_same_frame_from_multiple_sources(self):
        """
        Cùng frame n từ CLIP-B và CLIP-L phải được merge,
        không tạo thành 2 candidate riêng.
        """

        frames = [
            SourceFrame(
                n=7,
                source="clip_b",
                source_rank=2,
                source_score=0.8,
            ),
            SourceFrame(
                n=7,
                source="clip_l",
                source_rank=1,
                source_score=0.9,
            ),
        ]

        merged = merge_frame_provenance(frames)

        self.assertEqual(
            len(merged),
            1,
        )

        self.assertEqual(
            merged[0].n,
            7,
        )

        self.assertEqual(
            set(merged[0].sources.keys()),
            {"clip_b", "clip_l"},
        )

    def test_duplicate_same_source_keeps_best_evidence(self):
        """
        Nếu cùng source xuất hiện cùng frame nhiều lần,
        giữ rank tốt nhất.
        """

        frames = [
            SourceFrame(
                n=7,
                source="clip_b",
                source_rank=5,
                source_score=0.5,
            ),
            SourceFrame(
                n=7,
                source="clip_b",
                source_rank=2,
                source_score=0.8,
            ),
        ]

        merged = merge_frame_provenance(frames)

        self.assertEqual(
            len(merged),
            1,
        )

        evidence = merged[0].sources["clip_b"]

        self.assertEqual(
            evidence.rank,
            2,
        )

        self.assertAlmostEqual(
            evidence.score,
            0.8,
        )

    def test_frame_evidence_combines_sources(self):
        """
        Evidence score phải cộng contribution của các source.
        """

        sources = {
            "clip_b": type(
                "Evidence",
                (),
                {
                    "rank": 1,
                    "score": 0.9,
                },
            )(),
            "clip_l": type(
                "Evidence",
                (),
                {
                    "rank": 2,
                    "score": 0.8,
                },
            )(),
        }

        expected = (
            W_B / (RRF_K + 1)
            + W_L / (RRF_K + 2)
        )

        actual = calculate_frame_evidence(
            sources
        )

        self.assertAlmostEqual(
            actual,
            expected,
        )

    def test_multi_source_candidate_preserves_provenance(self):
        frames = [
            SourceFrame(
                n=7,
                source="clip_b",
                source_rank=2,
                source_score=0.8,
            ),
            SourceFrame(
                n=7,
                source="clip_l",
                source_rank=1,
                source_score=0.9,
            ),
        ]

        merged = merge_frame_provenance(
            frames
        )

        video = make_video(
            video_id="V_1",
            rank_video=1,
            frames=frames,
        )

        lookup = {
            ("V_1", 7): (
                700,
                7.0,
            )
        }

        candidate = build_frame_candidate(
            query_id="Q_1",
            video=video,
            frame=merged[0],
            frame_lookup=lookup,
        )

        self.assertIsNotNone(candidate)

        assert candidate is not None

        self.assertEqual(
            candidate.query_id,
            "Q_1",
        )

        self.assertEqual(
            candidate.source_hits,
            ("clip_b", "clip_l"),
        )

        self.assertEqual(
            candidate.source_ranks,
            {
                "clip_b": 2,
                "clip_l": 1,
            },
        )

        self.assertEqual(
            candidate.source_scores,
            {
                "clip_b": 0.8,
                "clip_l": 0.9,
            },
        )

    def test_candidate_record_exports_semantic_coverage_and_window(self):
        frame = MergedFrame(
            n=7,
            sources={
                "clip_l": SourceEvidence(rank=2, score=0.8),
            },
        )
        video = make_video("V_1", rank_video=1)
        candidate = build_frame_candidate(
            query_id="Q_1",
            video=video,
            frame=frame,
            frame_lookup={("V_1", 7): (210, 7.0)},
            preferred_modalities={"clip_l": 0.7, "ocr": 0.3},
            source_weights={"clip_b": 1.0, "clip_l": 0.7},
        )

        self.assertIsNotNone(candidate)
        record = frame_to_json(candidate)
        self.assertEqual(record["semantic_coverage"], 0.7)
        self.assertEqual(record["window"], {"start": 4.0, "end": 10.0})
        self.assertEqual(
            record["routing_weights_applied"],
            {"clip_b": 1.0, "clip_l": 0.7},
        )

# ============================================================
# STAGE 2 — TOP 50
# ============================================================

class TestQAR1Stage2Coverage(unittest.TestCase):

    def setUp(self):
        self.frame_lookup: dict[
            tuple[str, int],
            tuple[int, float],
        ] = {}

        self.mock_top300: list[
            VideoCandidate
        ] = []

        for i in range(60):

            video_id = f"V_{i}"

            self.frame_lookup[
                (video_id, 1)
            ] = (
                i * 100 + 10,
                i * 4.0,
            )

            self.frame_lookup[
                (video_id, 2)
            ] = (
                i * 100 + 20,
                i * 4.0 + 0.4,
            )

            frames = [
                SourceFrame(
                    n=1,
                    source="clip_b",
                    source_rank=2,
                    source_score=0.8,
                ),
                SourceFrame(
                    n=2,
                    source="clip_b",
                    source_rank=1,
                    source_score=0.9,
                ),
            ]

            self.mock_top300.append(
                make_video(
                    video_id=video_id,
                    rank_video=i + 1,
                    score_fused=1.0 / (i + 1),
                    frames=frames,
                )
            )

    def test_stage2_returns_top50(self):
        candidates = extract_top50_coverage(
            query_id="Q_1",
            query_type="mo_ta",
            top300=self.mock_top300,
            frame_lookup=self.frame_lookup,
        )

        self.assertEqual(
            len(candidates),
            50,
        )

    def test_stage2_preserves_video_diversity_for_normal_query(self):
        """
        Query thường:
        quota = 1 frame / video.
        """

        candidates = extract_top50_coverage(
            query_id="Q_1",
            query_type="mo_ta",
            top300=self.mock_top300,
            frame_lookup=self.frame_lookup,
        )

        video_ids = [
            candidate.video_id
            for candidate in candidates
        ]

        self.assertEqual(
            len(set(video_ids)),
            50,
        )

    def test_stage2_rank_and_stage_are_correct(self):
        candidates = extract_top50_coverage(
            query_id="Q_1",
            query_type="mo_ta",
            top300=self.mock_top300,
            frame_lookup=self.frame_lookup,
        )

        self.assertEqual(
            candidates[0].stage,
            "coverage",
        )

        self.assertEqual(
            candidates[0].rank_final,
            1,
        )

        self.assertEqual(
            candidates[-1].rank_final,
            50,
        )

    def test_missing_best_frame_falls_back_to_valid_frame(self):
        """
        Frame tốt nhất n=1 không có frame_map.
        n=2 vẫn hợp lệ => video không bị mất.
        """

        top300 = [
            VideoCandidate(
                video_id="V_1",
                rrf_score=0.02,
                best_b_rank=1,
                best_l_rank=None,
                frames=[
                    SourceFrame(
                        n=1,
                        source="clip_b",
                        source_rank=1,
                        source_score=0.9,
                    ),
                    SourceFrame(
                        n=2,
                        source="clip_b",
                        source_rank=2,
                        source_score=0.8,
                    ),
                ],
            )
        ]

        lookup = {
            ("V_1", 2): (
                200,
                2.0,
            )
        }

        candidates = extract_top50_coverage(
            query_id="Q_1",
            query_type="mo_ta",
            top300=top300,
            frame_lookup=lookup,
        )

        self.assertEqual(
            len(candidates),
            1,
        )

        self.assertEqual(
            candidates[0].n,
            2,
        )


# ============================================================
# STAGE 2 — TRAKE
# ============================================================

class TestQAR1TRAKE(unittest.TestCase):

    def test_trake_keeps_multiple_times_same_video(self):
        """
        TRAKE phải giữ được nhiều frame cách nhau đủ xa
        trong cùng video.
        """

        frames = [
            SourceFrame(
                n=1,
                source="clip_b",
                source_rank=1,
                source_score=0.9,
            ),
            SourceFrame(
                n=2,
                source="clip_b",
                source_rank=2,
                source_score=0.8,
            ),
            SourceFrame(
                n=3,
                source="clip_l",
                source_rank=1,
                source_score=0.85,
            ),
        ]

        video = VideoCandidate(
            video_id="V_Target",
            rrf_score=0.02,
            best_b_rank=1,
            best_l_rank=1,
            frames=frames,
        )

        lookup = {
            ("V_Target", 1): (
                102,
                1.0,
            ),
            ("V_Target", 2): (
                203,
                5.0,
            ),
            ("V_Target", 3): (
                300,
                9.0,
            ),
        }

        candidates = extract_top50_coverage(
            query_id="Q_TRAKE",
            query_type="chuoi_su_kien",
            top300=[video],
            frame_lookup=lookup,
        )

        self.assertGreaterEqual(
            len(candidates),
            2,
        )

        self.assertEqual(
            {
                candidate.video_id
                for candidate in candidates
            },
            {"V_Target"},
        )

        self.assertGreaterEqual(
            len(candidates),
            3,
        )

    def test_trake_recall_requires_all_events(self):
        gt = GroundTruth(
            query_id="Q_TRAKE",
            video_id="V_Target",
            query_type="chuoi_su_kien",
            frame_tolerance=5,
            raw_item={
                "id": "Q_TRAKE",
                "video_id": "V_Target",
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
        )

        full_hit = [
            FrameCandidate(
                query_id="Q_TRAKE",
                video_id="V_Target",
                n=1,
                frame_idx=102,
                pts_time=1.0,
                stage="coverage",
                source_hits=("clip_b",),
                source_ranks={"clip_b": 1},
                source_scores={"clip_b": 0.9},
                evidence_score=0.01,
                best_source_rank=1,
                score_fused=0.02,
                rank_video=1,
                rank_final=1,
            ),
            FrameCandidate(
                query_id="Q_TRAKE",
                video_id="V_Target",
                n=2,
                frame_idx=203,
                pts_time=5.0,
                stage="coverage",
                source_hits=("clip_b",),
                source_ranks={"clip_b": 2},
                source_scores={"clip_b": 0.8},
                evidence_score=0.009,
                best_source_rank=2,
                score_fused=0.02,
                rank_video=1,
                rank_final=2,
            ),
        ]

        partial_hit = full_hit[:1]

        self.assertTrue(
            frame_candidates_hit(
                full_hit,
                gt,
            )
        )

        self.assertFalse(
            frame_candidates_hit(
                partial_hit,
                gt,
            )
        )


# ============================================================
# QUERYPLAN INTEGRATION
# ============================================================

class TestQAR1QueryPlanIntegration(unittest.TestCase):

    def test_single_query_record_proves_query_plan_was_applied(self):
        class FakeIndex:
            def search(self, query, k):
                return (
                    np.asarray([[0.9]], dtype=np.float32),
                    np.asarray([[0]], dtype=np.int64),
                )

        gt = make_gt(
            query_id="12",
            video_id="V_Target",
            query_type="hoi_dap",
            frame_start=100,
            frame_end=105,
        )
        query_plan = {
            "schema_version": "1.1",
            "query_id": "12",
            "preferred_modalities": {
                "clip_l": 0.7,
                "ocr": 0.1,
                "asr": 0.0,
                "caption": 0.2,
            },
            "semantic_k_hint": 500,
        }

        _, _, _, records, _ = run_single_query_benchmark(
            index_b=FakeIndex(),
            index_l=FakeIndex(),
            b_ids=[("V_Target", 7)],
            l_ids=[("V_Target", 7)],
            frame_lookup={("V_Target", 7): (102, 3.4)},
            gt_data=[gt],
            query_b=np.zeros((1, 2), dtype=np.float32),
            query_l=np.zeros((1, 2), dtype=np.float32),
            b_row_map={"12": 0},
            l_row_map={"12": 0},
            query_plans={"12": query_plan},
        )

        record = records[0]
        candidate = record["reader_candidates"][0]
        self.assertEqual(record["query_plan_schema_version"], "1.1")
        self.assertEqual(record["semantic_k_hint_applied"], 500)
        self.assertEqual(record["routing_weights_applied"]["clip_l"], 0.7)
        self.assertEqual(candidate["semantic_coverage"], 0.7)
        self.assertEqual(candidate["window"], {"start": 0.4, "end": 6.4})

    def test_query_routing_does_not_change_official_top300_gate(self):
        class FakeIndex:
            def search(self, query, k):
                return (
                    np.asarray([[0.9, 0.8]], dtype=np.float32),
                    np.asarray([[0, 1]], dtype=np.int64),
                )

        original_k_fused = candidate_funnel.K_FUSED
        candidate_funnel.K_FUSED = 1
        try:
            gt = make_gt(
                query_id="12",
                video_id="V_Z",
                query_type="hoi_dap",
                frame_start=100,
                frame_end=105,
            )
            _, _, _, records, _ = run_single_query_benchmark(
                index_b=FakeIndex(),
                index_l=FakeIndex(),
                b_ids=[("V_Z", 1), ("V_A", 1)],
                l_ids=[("V_A", 1), ("V_Z", 1)],
                frame_lookup={
                    ("V_Z", 1): (102, 3.4),
                    ("V_A", 1): (999, 9.0),
                },
                gt_data=[gt],
                query_b=np.zeros((1, 2), dtype=np.float32),
                query_l=np.zeros((1, 2), dtype=np.float32),
                b_row_map={"12": 0},
                l_row_map={"12": 0},
                query_plans={
                    "12": {
                        "schema_version": "1.1",
                        "query_id": "12",
                        "preferred_modalities": {"clip_l": 1.0},
                        "semantic_k_hint": 100,
                    }
                },
            )
        finally:
            candidate_funnel.K_FUSED = original_k_fused

        record = records[0]
        self.assertTrue(record["hit_video_300"])
        self.assertEqual(record["gt_video_rank"], 1)
        self.assertEqual(record["gt_video_rank_routed"], 2)


# ============================================================
# STAGE 3 — TOP 12
# ============================================================

class TestQAR1Stage3Reader(unittest.TestCase):

    def test_stage3_returns_top12(self):

        candidates = []

        for i in range(50):

            candidates.append(
                FrameCandidate(
                    query_id="Q_1",
                    video_id=f"V_{i}",
                    n=i,
                    frame_idx=i * 10,
                    pts_time=float(i),
                    stage="coverage",
                    source_hits=("clip_b",),
                    source_ranks={"clip_b": i + 1},
                    source_scores={"clip_b": 0.9},
                    evidence_score=1.0 / (i + 1),
                    best_source_rank=i + 1,
                    score_fused=0.02,
                    rank_video=i + 1,
                    rank_final=i + 1,
                )
            )

        reader = select_top12_reader(
            candidates
        )

        self.assertEqual(
            len(reader),
            12,
        )

        self.assertEqual(
            reader[0].stage,
            "reader",
        )

        self.assertEqual(
            reader[0].rank_final,
            1,
        )

        self.assertEqual(
            reader[-1].rank_final,
            12,
        )

    def test_stage3_does_not_mutate_beyond_top12(self):

        candidates = []

        for i in range(20):

            candidates.append(
                FrameCandidate(
                    query_id="Q_1",
                    video_id=f"V_{i}",
                    n=i,
                    frame_idx=i,
                    pts_time=float(i),
                    stage="coverage",
                    source_hits=("clip_b",),
                    source_ranks={"clip_b": 1},
                    source_scores={"clip_b": 0.9},
                    evidence_score=1.0,
                    best_source_rank=1,
                    score_fused=0.02,
                    rank_video=1,
                    rank_final=i + 1,
                )
            )

        reader = select_top12_reader(
            candidates
        )

        self.assertEqual(
            len(reader),
            12,
        )

        self.assertEqual(
            reader[-1].rank_final,
            12,
        )

        # Quan trọng:
        # candidate thứ 13 trở đi không bị mutate.
        self.assertEqual(
            candidates[12].stage,
            "coverage",
        )

        self.assertEqual(
            candidates[12].rank_final,
            13,
        )


# ============================================================
# RECALL EVALUATION
# ============================================================

class TestQAR1RecallEvaluation(unittest.TestCase):

    def test_video_recall_300(self):

        gt = make_gt()

        videos_hit = [
            make_video(
                "V_Other",
                1,
            ),
            make_video(
                "V_Target",
                2,
            ),
        ]

        videos_miss = [
            make_video(
                "V_Other",
                1,
            )
        ]

        self.assertTrue(
            video_pool_hit(
                videos_hit,
                gt,
            )
        )

        self.assertFalse(
            video_pool_hit(
                videos_miss,
                gt,
            )
        )

    def test_frame_recall_hit(self):

        gt = make_gt(
            frame_start=100,
            frame_end=105,
        )

        candidate = FrameCandidate(
            query_id="Q_1",
            video_id="V_Target",
            n=1,
            frame_idx=108,
            pts_time=1.0,
            stage="coverage",
            source_hits=("clip_b",),
            source_ranks={"clip_b": 1},
            source_scores={"clip_b": 0.9},
            evidence_score=0.01,
            best_source_rank=1,
            score_fused=0.02,
            rank_video=1,
            rank_final=1,
        )

        self.assertTrue(
            frame_candidates_hit(
                [candidate],
                gt,
            )
        )

    def test_frame_recall_miss(self):

        gt = make_gt(
            frame_start=100,
            frame_end=105,
        )

        candidate = FrameCandidate(
            query_id="Q_1",
            video_id="V_Target",
            n=1,
            frame_idx=130,
            pts_time=1.0,
            stage="coverage",
            source_hits=("clip_b",),
            source_ranks={"clip_b": 1},
            source_scores={"clip_b": 0.9},
            evidence_score=0.01,
            best_source_rank=1,
            score_fused=0.02,
            rank_video=1,
            rank_final=1,
        )

        self.assertFalse(
            frame_candidates_hit(
                [candidate],
                gt,
            )
        )


# ============================================================
# QUERY EMBEDDING ALIGNMENT
# ============================================================

class TestQAR1EmbeddingAlignment(unittest.TestCase):

    def _write_ids(
        self,
        path: Path,
        ids: list[str],
    ) -> None:

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                ids,
                f,
                ensure_ascii=False,
            )

    def test_alignment_success(self):

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = Path(tmp)

            embedding_path = (
                tmp_path / "emb.npy"
            )

            mapping_path = (
                tmp_path / "ids.json"
            )

            embeddings = np.zeros(
                (3, 512),
                dtype=np.float32,
            )

            np.save(
                embedding_path,
                embeddings,
            )

            ids = [
                "Q_1",
                "Q_2",
                "Q_3",
            ]

            self._write_ids(
                mapping_path,
                ids,
            )

            result = validate_embedding_alignment(
                embedding_path=embedding_path,
                mapping_path=mapping_path,
                ground_truth_ids=set(ids),
                expected_dimension=512,
            )

            self.assertEqual(
                result.shape,
                (3, 512),
            )

            # mmap_mode="r" phải trả mmap-backed array.
            self.assertIsInstance(
                result,
                np.memmap,
            )

            # Windows giữ file handle của memmap.
            # Phải giải phóng trước khi TemporaryDirectory
            # cố xóa emb.npy.
            del result

    def test_alignment_rejects_duplicate_query_ids(self):

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = Path(tmp)

            embedding_path = (
                tmp_path / "emb.npy"
            )

            mapping_path = (
                tmp_path / "ids.json"
            )

            np.save(
                embedding_path,
                np.zeros(
                    (2, 512),
                    dtype=np.float32,
                ),
            )

            self._write_ids(
                mapping_path,
                [
                    "Q_1",
                    "Q_1",
                ],
            )

            with self.assertRaises(ValueError):
                validate_embedding_alignment(
                    embedding_path=embedding_path,
                    mapping_path=mapping_path,
                    ground_truth_ids={
                        "Q_1",
                    },
                    expected_dimension=512,
                )

    def test_alignment_rejects_missing_query(self):

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = Path(tmp)

            embedding_path = (
                tmp_path / "emb.npy"
            )

            mapping_path = (
                tmp_path / "ids.json"
            )

            np.save(
                embedding_path,
                np.zeros(
                    (2, 512),
                    dtype=np.float32,
                ),
            )

            self._write_ids(
                mapping_path,
                [
                    "Q_1",
                    "Q_2",
                ],
            )

            with self.assertRaises(ValueError):
                validate_embedding_alignment(
                    embedding_path=embedding_path,
                    mapping_path=mapping_path,
                    ground_truth_ids={
                        "Q_1",
                        "Q_2",
                        "Q_3",
                    },
                    expected_dimension=512,
                )

    def test_alignment_rejects_extra_query(self):

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = Path(tmp)

            embedding_path = (
                tmp_path / "emb.npy"
            )

            mapping_path = (
                tmp_path / "ids.json"
            )

            np.save(
                embedding_path,
                np.zeros(
                    (3, 512),
                    dtype=np.float32,
                ),
            )

            self._write_ids(
                mapping_path,
                [
                    "Q_1",
                    "Q_2",
                    "Q_EXTRA",
                ],
            )

            with self.assertRaises(ValueError):
                validate_embedding_alignment(
                    embedding_path=embedding_path,
                    mapping_path=mapping_path,
                    ground_truth_ids={
                        "Q_1",
                        "Q_2",
                    },
                    expected_dimension=512,
                )

    def test_alignment_rejects_row_count_mismatch(self):

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = Path(tmp)

            embedding_path = (
                tmp_path / "emb.npy"
            )

            mapping_path = (
                tmp_path / "ids.json"
            )

            np.save(
                embedding_path,
                np.zeros(
                    (3, 512),
                    dtype=np.float32,
                ),
            )

            self._write_ids(
                mapping_path,
                [
                    "Q_1",
                    "Q_2",
                ],
            )

            with self.assertRaises(ValueError):
                validate_embedding_alignment(
                    embedding_path=embedding_path,
                    mapping_path=mapping_path,
                    ground_truth_ids={
                        "Q_1",
                        "Q_2",
                    },
                    expected_dimension=512,
                )

    def test_alignment_rejects_dimension_mismatch(self):

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = Path(tmp)

            embedding_path = (
                tmp_path / "emb.npy"
            )

            mapping_path = (
                tmp_path / "ids.json"
            )

            np.save(
                embedding_path,
                np.zeros(
                    (2, 768),
                    dtype=np.float32,
                ),
            )

            self._write_ids(
                mapping_path,
                [
                    "Q_1",
                    "Q_2",
                ],
            )

            with self.assertRaises(ValueError):
                validate_embedding_alignment(
                    embedding_path=embedding_path,
                    mapping_path=mapping_path,
                    ground_truth_ids={
                        "Q_1",
                        "Q_2",
                    },
                    expected_dimension=512,
                )

    def test_embedding_row_map_is_explicit(self):

        ids = [
            "Q_003",
            "Q_001",
            "Q_002",
        ]

        row_map = build_embedding_row_map(
            ids
        )

        self.assertEqual(
            row_map["Q_003"],
            0,
        )

        self.assertEqual(
            row_map["Q_001"],
            1,
        )

        self.assertEqual(
            row_map["Q_002"],
            2,
        )


# ============================================================
# QUALITY GATE
# ============================================================

class TestQAR1QualityGate(unittest.TestCase):

    def test_quality_gate_rejects_zero_recall(self):

        result = evaluate_quality_gate(
            recall_300=0.0,
            recall_50=0.0,
            recall_12=0.0,
            num_queries=76,
        )

        self.assertFalse(
            result["passed"]
        )

        self.assertFalse(
            result["checks"][
                "recall_at_300_videos"
            ]
        )

        self.assertFalse(
            result["checks"][
                "recall_at_50_frames"
            ]
        )

        self.assertFalse(
            result["checks"][
                "recall_at_12_frames"
            ]
        )

    def test_quality_gate_accepts_baseline(self):

        result = evaluate_quality_gate(
            recall_300=56 / 76,
            recall_50=13 / 76,
            recall_12=9 / 76,
            num_queries=76,
        )

        self.assertTrue(
            result["passed"]
        )

    def test_quality_gate_rejects_below_baseline(self):

        result = evaluate_quality_gate(
            recall_300=(56 / 76) - 0.001,
            recall_50=13 / 76,
            recall_12=9 / 76,
            num_queries=76,
        )

        self.assertFalse(
            result["passed"]
        )

    def test_quality_gate_preserves_query_count(self):

        result = evaluate_quality_gate(
            recall_300=56 / 76,
            recall_50=13 / 76,
            recall_12=9 / 76,
            num_queries=32,
        )

        self.assertEqual(
            result["evaluated_queries"],
            32,
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
