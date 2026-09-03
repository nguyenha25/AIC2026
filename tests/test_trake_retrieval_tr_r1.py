from __future__ import annotations

from dataclasses import fields

import pytest

from aic2026.trake_retrieval import (
    CoarseRegion,
    TRR1Config,
    TRR1Result,
    _build_test_hit,
    _gom_vung,
    _video_consensus_scores,
    coarse_region_to_dict,
    tim_nhieu_su_kien,
    tim_vung_tho,
    trr1_result_to_dict,
)


def test_queryplan_event_requires_exact_text_field():
    event = {
        "event_id": "E1",
        "text": "a person opens a door",
        "relation": "start",
    }

    result = tim_vung_tho(
        event,
        retriever=lambda text, k: [],
    )

    assert result.event_id == "E1"
    assert result.text == "a person opens a door"
    assert result.relation == "start"


def test_queryplan_event_missing_text_fails():
    event = {
        "event_id": "E1",
        "description": "a person opens a door",
        "relation": "start",
    }

    with pytest.raises(ValueError):
        tim_vung_tho(
            event,
            retriever=lambda text, k: [],
        )


def test_queryplan_event_does_not_stringify_missing_text():
    event = {
        "event_id": "E1",
        "description": {"foo": "bar"},
        "relation": "start",
    }

    with pytest.raises(ValueError):
        tim_vung_tho(
            event,
            retriever=lambda text, k: [],
        )


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        TRR1Config(top_k=0)

    with pytest.raises(ValueError):
        TRR1Config(max_region_duration_seconds=0)

    with pytest.raises(ValueError):
        TRR1Config(region_merge_gap_seconds=-1)

    with pytest.raises(ValueError):
        TRR1Config(max_regions_per_event=0)

    with pytest.raises(ValueError):
        TRR1Config(min_region_duration_seconds=0)

    with pytest.raises(ValueError):
        TRR1Config(
            min_region_duration_seconds=11,
            max_region_duration_seconds=10,
        )

    with pytest.raises(ValueError):
        TRR1Config(video_consensus_weight=1.1)


def test_coarse_region_has_no_best_frame_idx():
    names = {field.name for field in fields(CoarseRegion)}

    assert "best_frame_idx" not in names
    assert "frame_idx" not in names

    assert names == {
        "video_id",
        "start_time",
        "end_time",
        "score",
        "hits",
    }


def test_nearby_hits_merge():
    config = TRR1Config(
        region_merge_gap_seconds=2.0,
        max_region_duration_seconds=10.0,
    )

    hits = [
        _build_test_hit("L23_V001", 10.0, 0.7),
        _build_test_hit("L23_V001", 11.0, 0.8),
        _build_test_hit("L23_V001", 12.5, 0.9),
    ]

    regions = _gom_vung(hits, config=config)

    assert len(regions) == 1

    region = regions[0]

    assert region.video_id == "L23_V001"
    assert region.start_time == 10.0
    assert region.end_time == 12.5

    # New TR-R1 scoring is a normalized combination of:
    # peak relevance + supporting hits + temporal density.
    assert 0.0 <= region.score <= 1.0

    # The strongest hit must still be represented in the region.
    assert max(hit["score"] for hit in region.hits) == 0.9


def test_far_hits_do_not_merge():
    config = TRR1Config(
        region_merge_gap_seconds=2.0,
        max_region_duration_seconds=10.0,
    )

    hits = [
        _build_test_hit("L23_V001", 10.0, 0.9),
        _build_test_hit("L23_V001", 20.0, 0.8),
    ]

    regions = _gom_vung(hits, config=config)

    assert len(regions) == 2


def test_chain_merge_is_bounded_by_absolute_duration():
    config = TRR1Config(
        region_merge_gap_seconds=2.0,
        max_region_duration_seconds=10.0,
    )

    # Nếu chỉ kiểm tra gap với hit trước:
    #
    # 0 -> 2 -> 4 -> 6 -> ... -> 20
    #
    # sẽ bị merge thành một region khổng lồ.
    #
    # Hard duration cap phải chặn việc này.

    hits = [
        _build_test_hit(
            "L23_V001",
            float(timestamp),
            1.0,
        )
        for timestamp in range(0, 22, 2)
    ]

    regions = _gom_vung(hits, config=config)

    assert len(regions) >= 2

    for region in regions:
        assert (
            region.end_time - region.start_time
            <= config.max_region_duration_seconds
        )


def test_different_videos_never_merge():
    config = TRR1Config(
        region_merge_gap_seconds=10.0,
        max_region_duration_seconds=20.0,
    )

    hits = [
        _build_test_hit("L23_V001", 10.0, 0.9),
        _build_test_hit("L23_V002", 10.5, 0.8),
    ]

    regions = _gom_vung(hits, config=config)

    assert len(regions) == 2
    assert {region.video_id for region in regions} == {
        "L23_V001",
        "L23_V002",
    }


def test_region_score_uses_more_than_peak_hit_only():
    config = TRR1Config()

    # Hai region có cùng peak score nhưng region đầu có
    # thêm evidence hỗ trợ. Với scoring mới, score không
    # được phép chỉ phụ thuộc vào max hit.
    hits = [
        _build_test_hit("L23_V001", 10.0, 0.9),
        _build_test_hit("L23_V001", 11.0, 0.8),
        _build_test_hit("L23_V001", 12.0, 0.7),
        _build_test_hit("L23_V002", 20.0, 0.9),
    ]

    regions = _gom_vung(hits, config=config)

    assert len(regions) == 2

    by_video = {region.video_id: region for region in regions}

    region_v1 = by_video["L23_V001"]
    region_v2 = by_video["L23_V002"]

    assert 0.0 <= region_v1.score <= 1.0
    assert 0.0 <= region_v2.score <= 1.0

    # V1 has the same peak relevance as V2 but more temporal
    # supporting evidence, so it should rank at least as high.
    assert region_v1.score >= region_v2.score


def test_regions_are_sorted_by_score():
    config = TRR1Config(
        region_merge_gap_seconds=0.5,
    )

    hits = [
        _build_test_hit("L23_V001", 10.0, 0.4),
        _build_test_hit("L23_V001", 20.0, 0.9),
        _build_test_hit("L23_V001", 30.0, 0.7),
    ]

    regions = _gom_vung(hits, config=config)

    assert len(regions) == 3

    scores = [region.score for region in regions]

    # Region ranking must be deterministic and descending.
    assert scores == sorted(scores, reverse=True)

    # Every score must remain a valid normalized score.
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_max_regions_per_event_is_enforced():
    config = TRR1Config(
        region_merge_gap_seconds=0.1,
        max_regions_per_event=2,
    )

    hits = [
        _build_test_hit("L23_V001", 0.0, 0.5),
        _build_test_hit("L23_V001", 10.0, 0.8),
        _build_test_hit("L23_V001", 20.0, 0.7),
        _build_test_hit("L23_V001", 30.0, 0.9),
    ]

    regions = _gom_vung(hits, config=config)

    assert len(regions) == 2

    scores = [region.score for region in regions]

    # Only the two highest-ranked regions survive.
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)

    # The strongest peak region must be retained.
    strongest_region = max(
        regions,
        key=lambda region: max(hit["score"] for hit in region.hits),
    )
    assert max(hit["score"] for hit in strongest_region.hits) == 0.9


def test_empty_hits_return_empty_regions():
    config = TRR1Config()

    regions = _gom_vung([], config=config)

    assert regions == []


def test_public_api_uses_supplied_retriever():
    calls = []

    def retriever(text, top_k):
        calls.append((text, top_k))

        return [
            _build_test_hit(
                "L23_V001",
                12.0,
                0.95,
            )
        ]

    event = {
        "event_id": "E1",
        "text": "a person opens a door",
        "relation": "start",
    }

    result = tim_vung_tho(
        event,
        config=TRR1Config(top_k=123),
        retriever=retriever,
    )

    assert calls == [
        ("a person opens a door", 123),
    ]

    assert len(result.regions) == 1
    assert result.regions[0].video_id == "L23_V001"
    assert result.regions[0].start_time == 11.75
    assert result.regions[0].end_time == 12.25


def test_singleton_hit_becomes_non_degenerate_region():
    regions = _gom_vung(
        [_build_test_hit("L23_V001", 12.0, 0.95)],
        config=TRR1Config(min_region_duration_seconds=0.5),
    )

    assert len(regions) == 1
    assert regions[0].start_time < 12.0 < regions[0].end_time
    assert regions[0].end_time - regions[0].start_time == pytest.approx(0.5)


def test_timestamp_comes_from_pts_time():
    config = TRR1Config()

    hits = [
        {
            "video_id": "L23_V001",
            "n": 999,
            "frame_idx": 12345,
            "pts_time": 42.5,
            "score": 0.9,
        }
    ]

    regions = _gom_vung(hits, config=config)

    assert len(regions) == 1
    assert regions[0].start_time == 42.25
    assert regions[0].end_time == 42.75


def test_video_consensus_is_length_invariant():
    scores = _video_consensus_scores(
        [
            [
                _build_test_hit("shared", 1.0, 0.8),
                _build_test_hit("long_wrong", 2.0, 0.7),
                _build_test_hit("long_wrong", 3.0, 0.6),
            ],
            [_build_test_hit("shared", 4.0, 0.8)],
        ]
    )

    assert scores["shared"] > scores["long_wrong"]


def test_multiple_events_use_shared_video_consensus():
    events = [
        {"event_id": "E1", "text": "first event"},
        {"event_id": "E2", "text": "second event"},
    ]

    def retriever(text, _top_k):
        wrong_video = "wrong_a" if text == "first event" else "wrong_b"
        return [
            _build_test_hit(wrong_video, 1.0, 0.99),
            _build_test_hit("shared_video", 2.0, 0.80),
        ]

    results = tim_nhieu_su_kien(
        events,
        config=TRR1Config(
            max_regions_per_event=1,
            video_consensus_weight=0.8,
        ),
        retriever=retriever,
    )

    assert [r.regions[0].video_id for r in results] == [
        "shared_video",
        "shared_video",
    ]


def test_single_event_keeps_local_ranking_even_with_consensus_enabled():
    results = tim_nhieu_su_kien(
        [{"event_id": "E1", "text": "event"}],
        config=TRR1Config(
            max_regions_per_event=1,
            video_consensus_weight=0.8,
        ),
        retriever=lambda _text, _top_k: [
            _build_test_hit("strong", 1.0, 0.99),
            _build_test_hit("weak", 2.0, 0.50),
        ],
    )

    assert results[0].regions[0].video_id == "strong"


def test_default_retriever_honors_query_expansion_config(monkeypatch):
    calls = []

    def fake_tim_hit(text, top_k, *, use_query_expansion, max_query_variants):
        calls.append((text, top_k, use_query_expansion, max_query_variants))
        return []

    monkeypatch.setattr(
        "aic2026.trake_retrieval.tim_hit_clip_l",
        fake_tim_hit,
    )

    tim_vung_tho(
        {"event_id": "E1", "text": "event"},
        config=TRR1Config(
            top_k=123,
            use_query_expansion=False,
            max_query_variants=1,
        ),
    )

    assert calls == [("event", 123, False, 1)]


def test_serialization_has_only_coarse_region_fields():
    region = CoarseRegion(
        video_id="L23_V001",
        start_time=10.0,
        end_time=12.0,
        score=0.9,
        hits=(
            {
                "video_id": "L23_V001",
                "pts_time": 10.0,
                "score": 0.9,
            },
        ),
    )

    data = coarse_region_to_dict(region)

    assert data["video_id"] == "L23_V001"
    assert data["start_time"] == 10.0
    assert data["end_time"] == 12.0
    assert data["score"] == 0.9

    assert "best_frame_idx" not in data
    assert "frame_idx" not in data


def test_result_serialization():
    result = TRR1Result(
        event_id="E1",
        text="person opens door",
        relation="start",
        regions=(
            CoarseRegion(
                video_id="L23_V001",
                start_time=10.0,
                end_time=12.0,
                score=0.9,
                hits=(),
            ),
        ),
    )

    data = trr1_result_to_dict(result)

    assert data == {
        "event_id": "E1",
        "text": "person opens door",
        "relation": "start",
        "regions": [
            {
                "video_id": "L23_V001",
                "start_time": 10.0,
                "end_time": 12.0,
                "score": 0.9,
                "hits": [],
            }
        ],
    }


def test_hard_duration_cap():
    config = TRR1Config(
        region_merge_gap_seconds=100.0,
        max_region_duration_seconds=5.0,
    )

    hits = [
        _build_test_hit("L23_V001", 0.0, 1.0),
        _build_test_hit("L23_V001", 4.0, 0.9),
        _build_test_hit("L23_V001", 5.0, 0.8),
        _build_test_hit("L23_V001", 10.0, 0.7),
    ]

    regions = _gom_vung(hits, config=config)

    for region in regions:
        assert (
            region.end_time - region.start_time
            <= 5.0
        )


def test_multiple_events_keep_queryplan_order():
    events = [
        {
            "event_id": "E1",
            "text": "person enters",
            "relation": "start",
        },
        {
            "event_id": "E2",
            "text": "person sits",
            "relation": "during",
        },
        {
            "event_id": "E3",
            "text": "person leaves",
            "relation": "after",
        },
    ]

    def retriever(text, top_k):
        return [
            _build_test_hit(
                "L23_V001",
                float(len(text)),
                0.8,
            )
        ]

    results = tim_nhieu_su_kien(
        events,
        retriever=retriever,
    )

    assert [result.event_id for result in results] == [
        "E1",
        "E2",
        "E3",
    ]


def test_result_is_deterministic():
    config = TRR1Config()

    hits = [
        _build_test_hit("L23_V001", 12.0, 0.7),
        _build_test_hit("L23_V001", 10.0, 0.9),
        _build_test_hit("L23_V001", 11.0, 0.8),
        _build_test_hit("L23_V002", 5.0, 0.95),
    ]

    result_a = _gom_vung(hits, config=config)
    result_b = _gom_vung(list(reversed(hits)), config=config)

    assert result_a == result_b


def test_trr1_result_has_expected_fields():
    names = {field.name for field in fields(TRR1Result)}

    assert names == {
        "event_id",
        "text",
        "relation",
        "regions",
    }
