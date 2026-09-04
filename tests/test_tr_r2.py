from aic2026.trake_r2_dp import solve_strict_increasing_path
from aic2026.trake_r2_windows import generate_dense_time_grid, chon_video_rrf, gop_cua_so_theo_video
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


def test_rrf_raises_when_no_candidates():
    try:
        chon_video_rrf({})
        assert False
    except ValueError:
        pass


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

    # Fake scores dùng cho DP.
    scores = {
        "E1": {
            1.0: 1.0,
            1.5: 0.5,
            2.0: 0.1,
            2.5: 0.0,
            3.0: 0.0,
            3.5: 0.0,
        },
        "E2": {
            1.0: 0.0,
            1.5: 0.0,
            2.0: 0.0,
            2.5: 0.2,
            3.0: 1.0,
            3.5: 0.5,
        },
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

        def fake_score_fn(event_id, pts_time):
            return scores[event_id][pts_time]

        return object(), fake_score_fn

    monkeypatch.setattr(
        pipeline,
        "build_dense_score_fn",
        fake_build_dense_score_fn,
    )

    result = pipeline.run_trake_r2(
        tr_r1_results,
        step=0.5,
        min_gap=1,
    )

    # -------------------------------------------------------------
    # Verify video selection
    # -------------------------------------------------------------

    assert captured["video_id"] == "L26_V315"

    # -------------------------------------------------------------
    # Verify TR-R1 event text được truyền xuống dense scorer
    # -------------------------------------------------------------

    assert captured["event_texts"] == {
        "E1": "Dao chạm vào cây xả",
        "E2": "Dao cắt cây xả",
    }

    # -------------------------------------------------------------
    # Verify API mới: multi-window
    #
    # Hai region:
    #   E1 -> [1.0, 2.0]
    #   E2 -> [2.5, 3.5]
    #
    # Chúng không overlap/touch nên phải giữ thành
    # hai window riêng biệt.
    # -------------------------------------------------------------

    assert captured["windows"] == [
        (1.0, 2.0),
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