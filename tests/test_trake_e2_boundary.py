import numpy as np
import pytest
from types import SimpleNamespace

from aic2026.trake_e2_boundary import (
    refine_from_dense_scorer,
)

from aic2026.trake_e2_boundary import (
    minmax_normalize,
    optimize_interval,
    refine_event_boundary,
    smooth_scores,
)


def test_minmax_normalize():
    x = minmax_normalize([2.0, 4.0, 6.0])

    assert np.allclose(
        x,
        [0.0, 0.5, 1.0],
    )


def test_minmax_normalize_flat():
    x = minmax_normalize([0.3, 0.3, 0.3])

    assert np.allclose(
        x,
        [0.0, 0.0, 0.0],
    )


def test_smoothing_giam_spike_don():
    raw = [
        0.1,
        0.1,
        1.0,
        0.1,
        0.1,
    ]

    smooth = smooth_scores(
        raw,
        radius=1,
    )

    assert smooth[2] < 1.0
    assert smooth[1] > 0.1
    assert smooth[3] > 0.1


def test_refine_plateau_bao_quanh_vung_manh():
    frames = [
        100,
        104,
        108,
        112,
        116,
        120,
        124,
    ]

    times = [
        4.00,
        4.16,
        4.32,
        4.48,
        4.64,
        4.80,
        4.96,
    ]

    scores = [
        0.1,
        0.2,
        0.8,
        0.9,
        0.85,
        0.2,
        0.1,
    ]

    r = refine_event_boundary(
        event_id="E1",
        frame_idx=frames,
        pts_time=times,
        raw_scores=scores,
        anchor_frame_idx=112,
        smoothing_radius=0,
        peak_search_radius=2,
        boundary_max_radius=4,
        relative_threshold=0.5,
        length_penalty=0.0,
        edge_penalty=0.0,
    )

    assert (
        r.start_frame_idx
        <= r.representative_frame_idx
        <= r.end_frame_idx
    )

    assert r.representative_frame_idx == 112

    assert 0.0 <= r.confidence <= 1.0


def test_anchor_khong_nhay_sang_peak_xa():
    frames = list(
        range(
            100,
            100 + 4 * 15,
            4,
        )
    )

    times = [
        i * 0.16
        for i in range(len(frames))
    ]

    scores = [
        0.1,
        0.2,
        0.8,
        0.7,
        0.2,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        1.0,
        0.9,
        0.1,
        0.1,
    ]

    r = refine_event_boundary(
        event_id="E1",
        frame_idx=frames,
        pts_time=times,
        raw_scores=scores,
        anchor_frame_idx=108,
        smoothing_radius=0,
        peak_search_radius=3,
    )

    # Peak xa hơn ở cuối có score 1.0,
    # nhưng anchor của E1 đang ở vùng đầu.
    assert r.representative_pos < 7


def test_output_dung_frame_idx_that():
    frames = [
        1000,
        1004,
        1008,
        1012,
    ]

    times = [
        40.00,
        40.16,
        40.32,
        40.48,
    ]

    scores = [
        0.1,
        0.7,
        0.9,
        0.2,
    ]

    r = refine_event_boundary(
        event_id="E2",
        frame_idx=frames,
        pts_time=times,
        raw_scores=scores,
        anchor_time=40.30,
        smoothing_radius=0,
    )

    assert (
        r.representative_frame_idx
        in frames
    )

    assert (
        r.start_frame_idx
        in frames
    )

    assert (
        r.end_frame_idx
        in frames
    )


def test_deterministic_khi_tie():
    frames = [
        100,
        104,
        108,
        112,
    ]

    times = [
        0.0,
        0.16,
        0.32,
        0.48,
    ]

    scores = [
        0.1,
        0.9,
        0.9,
        0.1,
    ]

    a = refine_event_boundary(
        event_id="E1",
        frame_idx=frames,
        pts_time=times,
        raw_scores=scores,
        anchor_frame_idx=106,
        smoothing_radius=0,
    )

    b = refine_event_boundary(
        event_id="E1",
        frame_idx=frames,
        pts_time=times,
        raw_scores=scores,
        anchor_frame_idx=106,
        smoothing_radius=0,
    )

    assert a == b


def test_loi_khi_do_dai_khong_khop():
    with pytest.raises(ValueError):
        refine_event_boundary(
            event_id="E1",
            frame_idx=[100, 104],
            pts_time=[1.0],
            raw_scores=[0.2, 0.3],
            anchor_frame_idx=100,
        )


def test_loi_khi_frame_idx_khong_tang():
    with pytest.raises(ValueError):
        refine_event_boundary(
            event_id="E1",
            frame_idx=[100, 96, 104],
            pts_time=[1.0, 1.16, 1.32],
            raw_scores=[0.1, 0.2, 0.3],
            anchor_frame_idx=100,
        )


def test_optimize_interval_luon_chua_peak():
    smooth = [
        0.1,
        0.6,
        1.0,
        0.7,
        0.1,
    ]

    start, end = optimize_interval(
        smooth,
        peak_pos=2,
        proposal_start=0,
        proposal_end=4,
    )

    assert start <= 2 <= end

def test_refine_from_dense_scorer_fake():
    scorer = SimpleNamespace(
        frames=[
            SimpleNamespace(
                frame_idx=100,
                pts_time=4.00,
            ),
            SimpleNamespace(
                frame_idx=104,
                pts_time=4.16,
            ),
            SimpleNamespace(
                frame_idx=108,
                pts_time=4.32,
            ),
            SimpleNamespace(
                frame_idx=112,
                pts_time=4.48,
            ),
            SimpleNamespace(
                frame_idx=116,
                pts_time=4.64,
            ),
        ],
        score_matrix=np.asarray(
            [
                [
                    0.1,
                    0.4,
                    0.9,
                    0.5,
                    0.1,
                ],
                [
                    0.1,
                    0.1,
                    0.3,
                    0.8,
                    0.9,
                ],
            ],
            dtype=np.float64,
        ),
    )

    results = refine_from_dense_scorer(
        scorer=scorer,
        event_ids=[
            "E1",
            "E2",
        ],
        chosen_times={
            "E1": 4.32,
            "E2": 4.64,
        },
        smoothing_radius=0,
        peak_search_radius=2,
        boundary_max_radius=3,
    )

    assert len(results) == 2

    assert results[0].event_id == "E1"
    assert results[1].event_id == "E2"

    assert (
        results[0].representative_frame_idx
        in {100, 104, 108, 112, 116}
    )

    assert (
        results[1].representative_frame_idx
        in {100, 104, 108, 112, 116}
    )

    assert 0.0 <= results[0].confidence <= 1.0
    assert 0.0 <= results[1].confidence <= 1.0

def test_adapter_does_not_jump_across_disjoint_windows():
    from types import SimpleNamespace

    import numpy as np

    frames = [
        SimpleNamespace(
            frame_idx=9032,
            pts_time=361.28,
        ),
        SimpleNamespace(
            frame_idx=9036,
            pts_time=361.44,
        ),
        SimpleNamespace(
            frame_idx=9040,
            pts_time=361.60,
        ),

        # Cửa sổ khác cách gần 195 giây.
        SimpleNamespace(
            frame_idx=13905,
            pts_time=556.20,
        ),
        SimpleNamespace(
            frame_idx=13909,
            pts_time=556.36,
        ),
        SimpleNamespace(
            frame_idx=13912,
            pts_time=556.48,
        ),
    ]

    scorer = SimpleNamespace(
        frames=frames,
        score_matrix=np.asarray(
            [
                [
                    0.20,
                    0.50,
                    0.80,

                    # Peak rất mạnh nhưng thuộc window khác.
                    0.90,
                    0.95,
                    1.00,
                ]
            ],
            dtype=float,
        ),
    )

    results = refine_from_dense_scorer(
        scorer=scorer,
        event_ids=["E1"],
        chosen_times={
            "E1": 361.62,
        },
    )

    result = results[0]

    assert result.representative_time < 362.0
    assert (
        361.28
        <= result.representative_time
        <= 361.60
    )