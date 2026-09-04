from types import SimpleNamespace

import numpy as np

from aic2026.trake_r2_dp import solve_strict_increasing_path
from aic2026.trake_r2_score import (
    SparseKeyframe,
    solve_sparse_video_paths,
)
from aic2026.trake_r2_windows import (
    chon_video_rrf,
    generate_dense_time_grid,
    gop_cua_so_theo_video,
    rank_video_candidates_rrf,
    windows_from_anchor_times,
)
from aic2026.trake_r2_pipeline import align_trake_query


# ---------------------------------------------------------------------
# DP
# ---------------------------------------------------------------------

def test_dp_picks_best_valid_increasing_sequence():
    S = [[1, 2, 3], [5, 4, 1]]
    chosen, total = solve_strict_increasing_path(S, min_gap=1)
    assert chosen == [0, 1]
    assert total == 5.0  # S[0][0] + S[1][1] = 1 + 4


def test_dp_enforces_strict_increasing_order():
    S = [[9, 1, 1], [1, 1, 9], [1, 9, 1]]
    chosen, _ = solve_strict_increasing_path(S, min_gap=1)
    assert chosen[0] < chosen[1] < chosen[2]


def test_dp_respects_min_gap():
    S = [[1, 1, 1, 1], [1, 1, 1, 1]]
    chosen, _ = solve_strict_increasing_path(S, min_gap=2)
    assert chosen[1] - chosen[0] >= 2


def test_dp_raises_on_infeasible_min_gap():
    S = [[1, 1], [1, 1], [1, 1]]  # 3 event, 2 frame, min_gap=1 -> cần >=3 frame
    try:
        solve_strict_increasing_path(S, min_gap=1)
        assert False, "phải raise ValueError khi infeasible"
    except ValueError:
        pass


def test_dp_raises_on_uneven_rows():
    S = [[1, 2, 3], [1, 2]]
    try:
        solve_strict_increasing_path(S)
        assert False
    except ValueError:
        pass


def test_dp_rejects_non_strict_min_gap():
    try:
        solve_strict_increasing_path([[1.0], [1.0]], min_gap=0)
        assert False
    except ValueError:
        pass


def test_dp_output_order_maps_to_queryplan_order():
    # Event 0 nên map ra frame sớm, event 2 map ra frame trễ, dù không phải
    # event nào cũng có điểm cao nhất ở đúng vị trí "tự nhiên" của nó.
    S = [[5, 1, 1, 1], [1, 5, 1, 1], [1, 1, 1, 5]]
    chosen, _ = solve_strict_increasing_path(S, min_gap=1)
    assert chosen == [0, 1, 3]


# ---------------------------------------------------------------------
# Dense time grid
# ---------------------------------------------------------------------

def test_dense_grid_step_and_bounds():
    grid = generate_dense_time_grid(0.0, 1.0, step=0.16)
    assert grid[0] == 0.0
    assert all(abs((grid[i + 1] - grid[i]) - 0.16) < 1e-9 for i in range(len(grid) - 1))
    assert grid[-1] <= 1.0


def test_dense_grid_rejects_invalid_range():
    try:
        generate_dense_time_grid(5.0, 2.0)
        assert False
    except ValueError:
        pass


def test_dense_grid_rejects_zero_step():
    try:
        generate_dense_time_grid(0.0, 1.0, step=0.0)
        assert False
    except ValueError:
        pass


# ---------------------------------------------------------------------
# Chọn video RRF
# ---------------------------------------------------------------------

def _region(vid, start, end):
    return {"video_id": vid, "start_time": start, "end_time": end}


def test_rrf_prefers_video_consistently_ranked_across_events():
    events_regions = {
        "E1": [_region("V_TARGET", 0, 1), _region("V_NOISE_A", 1, 2)],
        "E2": [_region("V_NOISE_B", 0, 1), _region("V_TARGET", 1, 2)],
        "E3": [_region("V_TARGET", 0, 1), _region("V_NOISE_C", 1, 2)],
    }
    assert chon_video_rrf(events_regions) == "V_TARGET"


def test_rrf_counts_each_video_once_per_event():
    events_regions = {
        "E1": [
            *[_region("V_NOISE", i, i + 1) for i in range(10)],
            _region("V_TARGET", 20, 21),
        ],
        "E2": [_region("V_TARGET", 30, 31)],
        "E3": [_region("V_TARGET", 40, 41)],
    }

    assert chon_video_rrf(events_regions) == "V_TARGET"


def test_rrf_raises_when_no_candidates():
    try:
        chon_video_rrf({})
        assert False
    except ValueError:
        pass


def test_rrf_returns_candidate_beam_in_score_order():
    events_regions = {
        "E1": [_region("V1", 0, 1), _region("V2", 0, 1)],
        "E2": [_region("V2", 1, 2), _region("V3", 1, 2)],
    }

    assert rank_video_candidates_rrf(
        events_regions,
        limit=2,
    ) == ["V2", "V1"]


def test_anchor_windows_merge_and_clamp_at_zero():
    assert windows_from_anchor_times(
        [2.0, 4.0, 20.0],
        padding_seconds=3.0,
    ) == [
        (0.0, 7.0),
        (17.0, 23.0),
    ]


def test_gop_cua_so_merges_only_matching_video():
    events_regions = {
        "E1": [_region("V_TARGET", 10.0, 12.0), _region("V_OTHER", 50.0, 52.0)],
        "E2": [_region("V_TARGET", 30.0, 32.0)],
    }
    start, end = gop_cua_so_theo_video(events_regions, "V_TARGET")
    assert start == 10.0
    assert end == 32.0


def test_gop_cua_so_raises_if_video_not_present():
    events_regions = {"E1": [_region("V_OTHER", 0, 1)]}
    try:
        gop_cua_so_theo_video(events_regions, "V_MISSING")
        assert False
    except ValueError:
        pass


# ---------------------------------------------------------------------
# Video-local sparse DP
# ---------------------------------------------------------------------

def test_sparse_dp_can_override_wrong_rrf_video():
    event_ids = ["E1", "E2", "E3"]

    frames_by_video = {
        "V_NOISE": [
            SparseKeyframe(n=i + 1, frame_idx=i, pts_time=float(i))
            for i in range(4)
        ],
        "V_TARGET": [
            SparseKeyframe(n=i + 1, frame_idx=i, pts_time=float(i))
            for i in range(4)
        ],
    }

    # V_NOISE có peak mạnh riêng lẻ nhưng không thể xếp cả ba peak theo
    # đúng thứ tự. V_TARGET có path E1@0 < E2@1 < E3@2 tốt hơn.
    score_matrices = {
        "V_NOISE": np.asarray(
            [
                [0.1, 0.1, 0.9, 0.1],
                [0.1, 0.9, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1],
            ],
            dtype=np.float32,
        ),
        "V_TARGET": np.asarray(
            [
                [0.8, 0.1, 0.1, 0.1],
                [0.1, 0.8, 0.1, 0.1],
                [0.1, 0.1, 0.8, 0.1],
            ],
            dtype=np.float32,
        ),
    }

    result = solve_sparse_video_paths(
        event_ids,
        frames_by_video,
        score_matrices,
        candidate_order=["V_NOISE", "V_TARGET"],
    )

    assert result["video_id"] == "V_TARGET"
    assert result["chosen_positions"] == [0, 1, 2]
    assert result["chosen_frame_idx"] == [0, 1, 2]
    assert list(result["chosen_times"]) == event_ids


def test_sparse_dp_rejects_non_increasing_frame_metadata():
    frames = {
        "V1": [
            SparseKeyframe(n=1, frame_idx=10, pts_time=2.0),
            SparseKeyframe(n=2, frame_idx=20, pts_time=1.0),
        ]
    }

    try:
        solve_sparse_video_paths(
            ["E1"],
            frames,
            {"V1": np.asarray([[0.1, 0.2]], dtype=np.float32)},
        )
        assert False
    except ValueError:
        pass


def test_sparse_dp_allows_equal_pts_time_when_frame_idx_increases():
    frames = {
        "V1": [
            SparseKeyframe(n=1, frame_idx=10, pts_time=1.0),
            SparseKeyframe(n=2, frame_idx=20, pts_time=1.0),
            SparseKeyframe(n=3, frame_idx=30, pts_time=2.0),
        ]
    }

    result = solve_sparse_video_paths(
        ["E1", "E2"],
        frames,
        {
            "V1": np.asarray(
                [
                    [0.9, 0.1, 0.0],
                    [0.0, 0.1, 0.9],
                ],
                dtype=np.float32,
            )
        },
    )

    assert result["chosen_frame_idx"] == [10, 30]
    assert result["chosen_times"] == {"E1": 1.0, "E2": 2.0}


def test_sparse_selector_rescores_all_video_keyframes(monkeypatch):
    import pandas as pd

    import aic2026.trake_r2_score as score_module
    from aic2026.index import clip_l_index

    ids = pd.DataFrame(
        [
            {
                "video_id": video_id,
                "n": n,
                # Regression thật: hai keyframe khác n/pts_time có thể
                # quy đổi về cùng frame_idx. Selector phải gộp chúng trước
                # DP để output vẫn tăng nghiêm ngặt.
                "frame_idx": (
                    10
                    if video_id == "V_TARGET" and n in (1, 2)
                    else n * 10
                ),
                "pts_time": float(n),
            }
            for video_id in ["V_NOISE", "V_TARGET"]
            for n in [1, 2, 3]
        ]
    )

    features = {
        # E1 peak nằm sau E2 peak nên strict DP không lấy được cả hai.
        "V_NOISE": np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [0.1, 0.1]],
            dtype=np.float32,
        ),
        # E1@frame10 < E2@frame30 là path đúng và mạnh.
        "V_TARGET": np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    fake_model = SimpleNamespace(eval=lambda: None)

    monkeypatch.setattr(
        score_module,
        "_get_trr1_clip_l_runtime",
        lambda: (fake_model, object(), "cpu", object(), ids),
    )
    monkeypatch.setattr(
        score_module,
        "_query_variants",
        lambda text, **kwargs: [text],
    )
    monkeypatch.setattr(
        score_module,
        "_encode_texts_with_runtime",
        lambda *args, **kwargs: FakeTensor(
            [[1.0, 0.0], [0.0, 1.0]]
        ),
    )
    monkeypatch.setattr(
        clip_l_index,
        "doc_dac_trung",
        lambda video_id, so_hang_can=None: features[video_id],
    )

    result = score_module.select_video_by_sparse_dp(
        {"E1": "first", "E2": "second"},
        ["V_NOISE", "V_TARGET"],
        use_query_expansion=False,
    )

    assert result["video_id"] == "V_TARGET"
    assert result["chosen_frame_idx"] == [10, 30]
    # E1 lấy score từ keyframe n=2 (time=2.0), dù n=1 có cùng frame_idx.
    assert result["chosen_times"] == {"E1": 2.0, "E2": 3.0}


# ---------------------------------------------------------------------
# Pipeline end-to-end (score_fn giả lập, không cần model thật)
# ---------------------------------------------------------------------

def test_align_trake_query_end_to_end_with_fake_scorer():
    events_tr_r1 = {
        "E1": {"regions": [_region("V_TARGET", 0.0, 5.0), _region("V_NOISE", 0.0, 5.0)]},
        "E2": {"regions": [_region("V_TARGET", 0.0, 5.0), _region("V_NOISE", 0.0, 5.0)]},
        "E3": {"regions": [_region("V_TARGET", 0.0, 5.0), _region("V_NOISE", 0.0, 5.0)]},
    }
    # "Đúng" thời điểm giả định: E1~1.0s, E2~2.5s, E3~4.0s -> điểm cao quanh đó.
    true_times = {"E1": 1.0, "E2": 2.5, "E3": 4.0}

    def fake_score_fn(event_id, pts_time):
        return -abs(pts_time - true_times[event_id])  # càng gần true_time điểm càng cao

    result = align_trake_query(events_tr_r1, fake_score_fn, step=0.16, min_gap=1)

    assert result["video_id"] == "V_TARGET"
    chosen = result["chosen_times"]
    assert chosen["E1"] < chosen["E2"] < chosen["E3"]
    assert abs(chosen["E1"] - 1.0) < 0.2
    assert abs(chosen["E2"] - 2.5) < 0.2
    assert abs(chosen["E3"] - 4.0) < 0.2


def test_align_trake_query_never_creates_global_edges():
    """
    Đảm bảo pipeline chỉ tính S trên đúng 1 video/cửa sổ đã chọn — không hề
    chạm tới danh sách video khác (kiểm tra gián tiếp: fake_score_fn raise
    nếu bị gọi ngoài phạm vi event_id đã biết, đảm bảo không có "cạnh" nào
    vượt ra ngoài tập event/cửa sổ dự kiến).
    """
    called_event_ids = set()

    def tracking_score_fn(event_id, pts_time):
        called_event_ids.add(event_id)
        return 1.0

    events_tr_r1 = {
        "E1": {"regions": [_region("V1", 0.0, 1.0)]},
        "E2": {"regions": [_region("V1", 0.0, 1.0)]},
    }
    align_trake_query(events_tr_r1, tracking_score_fn, step=0.5, min_gap=1)
    assert called_event_ids == {"E1", "E2"}

def test_run_trake_r2_wires_tr_r1_text_into_dense_scorer(monkeypatch):
    from aic2026.trake_retrieval import TRR1Result, CoarseRegion
    import aic2026.trake_r2_pipeline as pipeline

    tr_r1_results = (
        TRR1Result(
            event_id="E1",
            text="Dao chạm vào cây xả",
            relation="start",
            regions=(
                CoarseRegion(
                    video_id="L26_V315",
                    start_time=1.0,
                    end_time=2.0,
                    score=1.0,
                    hits=(),
                ),
            ),
        ),
        TRR1Result(
            event_id="E2",
            text="Dao cắt cây xả",
            relation="before",
            regions=(
                CoarseRegion(
                    video_id="L26_V315",
                    start_time=2.5,
                    end_time=3.5,
                    score=1.0,
                    hits=(),
                ),
            ),
        ),
    )

    captured = {}

    def fake_select_video_by_sparse_dp(
        event_texts,
        video_ids,
        **kwargs,
    ):
        captured["sparse_event_texts"] = event_texts
        captured["candidate_video_ids"] = video_ids
        captured["sparse_kwargs"] = kwargs

        return {
            "video_id": "L26_V315",
            "beam_rank": 1,
            "num_sparse_frames": 20,
            "chosen_positions": [2, 8],
            "chosen_frame_idx": [25, 75],
            "chosen_times": {
                "E1": 1.0,
                "E2": 3.0,
            },
            "total_score": 1.8,
            "mean_score": 0.9,
            "candidate_scores": [],
        }

    def fake_build_dense_score_fn(
        video_id,
        event_texts,
        *,
        windows=None,
        window_start=None,
        window_end=None,
        batch_size=16,
    ):
        captured["video_id"] = video_id
        captured["event_texts"] = event_texts
        captured["windows"] = windows
        captured["batch_size"] = batch_size

        scorer = SimpleNamespace(
            frames=[
                SimpleNamespace(frame_idx=25, pts_time=1.0),
                SimpleNamespace(frame_idx=50, pts_time=2.0),
                SimpleNamespace(frame_idx=75, pts_time=3.0),
                SimpleNamespace(frame_idx=100, pts_time=4.0),
            ],
            score_matrix=np.asarray(
                [
                    [1.0, 0.5, 0.1, 0.0],
                    [0.0, 0.2, 1.0, 0.5],
                ],
                dtype=np.float32,
            ),
        )

        return scorer, None

    monkeypatch.setattr(
        pipeline,
        "select_video_by_sparse_dp",
        fake_select_video_by_sparse_dp,
    )

    monkeypatch.setattr(
        pipeline,
        "build_dense_score_fn",
        fake_build_dense_score_fn,
    )

    result = pipeline.run_trake_r2(
        tr_r1_results,
        min_gap=1,
        sparse_window_padding_seconds=0.5,
    )

    # -------------------------------------------------------------
    # Verify video selection
    # -------------------------------------------------------------

    assert captured["video_id"] == "L26_V315"
    assert captured["candidate_video_ids"] == ["L26_V315"]

    # -------------------------------------------------------------
    # Verify TR-R1 event text được truyền xuống dense scorer
    # -------------------------------------------------------------

    assert captured["event_texts"] == {
        "E1": "Dao chạm vào cây xả",
        "E2": "Dao cắt cây xả",
    }
    assert captured["sparse_event_texts"] == captured["event_texts"]

    # -------------------------------------------------------------
    # Verify API mới: multi-window
    #
    # Sparse anchors E1=1.0, E2=3.0 với padding=0.5 tạo hai local windows.
    # -------------------------------------------------------------

    assert captured["windows"] == [
        (0.5, 1.5),
        (2.5, 3.5),
    ]

    # -------------------------------------------------------------
    # Verify default batch size
    # -------------------------------------------------------------

    assert captured["batch_size"] == 16

    # -------------------------------------------------------------
    # Verify DP result
    # -------------------------------------------------------------

    assert result["video_id"] == "L26_V315"

    assert result["chosen_times"]["E1"] == 1.0
    assert result["chosen_times"]["E2"] == 3.0
    assert result["chosen_frame_idx"] == [25, 75]
    assert result["sparse_selection"]["video_id"] == "L26_V315"
