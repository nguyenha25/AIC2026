from scripts.export_trake_ranked import candidate_answer, export_one


def test_candidate_answer_keeps_real_increasing_frame_idx():
    answer = candidate_answer(
        {
            "video_id": "L26_V315",
            "frame_idx": [100, 104, 109],
        },
        expected_events=3,
    )

    assert answer is not None
    assert answer.video_id == "L26_V315"
    assert answer.frame_ids == [100, 104, 109]


def test_candidate_answer_rejects_duplicate_or_wrong_event_count():
    assert candidate_answer(
        {"video_id": "L26_V315", "frame_idx": [100, 100, 109]},
        expected_events=3,
    ) is None
    assert candidate_answer(
        {"video_id": "L26_V315", "frame_idx": [100, 109]},
        expected_events=3,
    ) is None


def test_export_keeps_two_distinct_profiles_from_same_video(tmp_path):
    path, rows, rejected = export_one(
        {
            "query_id": "32",
            "events": [{"event_id": "E1"}, {"event_id": "E2"}],
            "ranked_candidates": [
                {
                    "video_id": "L26_V091",
                    "frame_idx": [100, 200],
                    "source": "dense_profile_wide_forward",
                },
                {
                    "video_id": "L26_V091",
                    "frame_idx": [110, 210],
                    "source": "dense_profile_narrow_forward",
                },
            ],
        },
        tmp_path,
        max_answers=20,
    )

    assert rows == 2
    assert rejected == 0
    assert path.read_text(encoding="utf-8").splitlines() == [
        "L26_V091,100,200",
        "L26_V091,110,210",
    ]
