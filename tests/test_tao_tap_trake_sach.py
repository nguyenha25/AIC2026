from pathlib import Path

import scripts.tao_tap_trake_sach as m


def test_khung_dense_bo_qua_ten_khong_phai_so(tmp_path, monkeypatch):
    video_id = "L99_V001"
    thu_muc = tmp_path / video_id
    thu_muc.mkdir(parents=True)

    (thu_muc / "000100.jpg").write_bytes(b"x")
    (thu_muc / "000104.jpg").write_bytes(b"x")
    (thu_muc / "check.jpg").write_bytes(b"x")

    monkeypatch.setattr(m, "FRAMES_DENSE_DIR", tmp_path)

    assert m.khung_dense(video_id) == [100, 104]


def test_kiem_mot_cau_clean_khi_moi_event_co_dense(tmp_path, monkeypatch):
    video_id = "L99_V001"
    thu_muc = tmp_path / video_id
    thu_muc.mkdir(parents=True)

    for f in [100, 104, 108, 112]:
        (thu_muc / f"{f:06d}.jpg").write_bytes(b"x")

    monkeypatch.setattr(m, "FRAMES_DENSE_DIR", tmp_path)

    q = {
        "id": "01",
        "loai_truy_van": "chuoi_su_kien",
        "video_id": video_id,
        "cac_giai_doan": [
            {
                "su_kien": "A",
                "frame_start": 99,
                "frame_end": 105,
            },
            {
                "su_kien": "B",
                "frame_start": 107,
                "frame_end": 113,
            },
        ],
    }

    kq = m.kiem_mot_cau(q)

    assert kq["status"] == "clean"
    assert kq["event_count"] == 2
    assert kq["covered_event_count"] == 2
    assert kq["missing"] == []


def test_kiem_mot_cau_missing_khi_event_khong_co_dense(tmp_path, monkeypatch):
    video_id = "L99_V002"
    thu_muc = tmp_path / video_id
    thu_muc.mkdir(parents=True)

    (thu_muc / "000100.jpg").write_bytes(b"x")

    monkeypatch.setattr(m, "FRAMES_DENSE_DIR", tmp_path)

    q = {
        "id": "02",
        "loai_truy_van": "chuoi_su_kien",
        "video_id": video_id,
        "cac_giai_doan": [
            {
                "su_kien": "A",
                "frame_start": 100,
                "frame_end": 100,
            },
            {
                "su_kien": "B",
                "frame_start": 200,
                "frame_end": 210,
            },
        ],
    }

    kq = m.kiem_mot_cau(q)

    assert kq["status"] == "missing"
    assert kq["covered_event_count"] == 1
    assert "event_2_khong_co_dense" in kq["missing"]


def test_kiem_mot_cau_khong_can_keyframe_goc(tmp_path, monkeypatch):
    video_id = "L99_V003"
    thu_muc = tmp_path / video_id
    thu_muc.mkdir(parents=True)

    (thu_muc / "000050.jpg").write_bytes(b"x")

    monkeypatch.setattr(m, "FRAMES_DENSE_DIR", tmp_path)

    q = {
        "id": "03",
        "loai_truy_van": "chuoi_su_kien",
        "video_id": video_id,
        "cac_giai_doan": [
            {
                "su_kien": "A",
                "frame_start": 40,
                "frame_end": 60,
            }
        ],
    }

    kq = m.kiem_mot_cau(q)

    assert kq["status"] == "clean"
    assert "keyframe_goc" not in kq["missing"]


def test_kiem_mot_cau_bao_loi_khi_start_lon_hon_end(tmp_path, monkeypatch):
    video_id = "L99_V004"
    thu_muc = tmp_path / video_id
    thu_muc.mkdir(parents=True)

    (thu_muc / "000100.jpg").write_bytes(b"x")

    monkeypatch.setattr(m, "FRAMES_DENSE_DIR", tmp_path)

    q = {
        "id": "04",
        "loai_truy_van": "chuoi_su_kien",
        "video_id": video_id,
        "cac_giai_doan": [
            {
                "su_kien": "A",
                "frame_start": 120,
                "frame_end": 100,
            }
        ],
    }

    kq = m.kiem_mot_cau(q)

    assert kq["status"] == "missing"
    assert "event_1_frame_start_lon_hon_frame_end" in kq["missing"]


def test_doc_dev_chi_lay_chuoi_su_kien(tmp_path):
    p = tmp_path / "dev.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"id":"01","loai_truy_van":"mo_ta","video_id":"A"}',
                '{"id":"02","loai_truy_van":"hoi_dap","video_id":"B"}',
                '{"id":"03","loai_truy_van":"chuoi_su_kien","video_id":"C","cac_giai_doan":[]}',
            ]
        ),
        encoding="utf-8",
    )

    kq = m.doc_dev(p)

    assert len(kq) == 1
    assert kq[0]["id"] == "03"