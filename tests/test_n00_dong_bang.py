"""N-00 — schema, evidence sạch và split không leakage."""

from __future__ import annotations

import json

import pytest

from aic2026.n00_freeze import (
    LoiDongBang,
    chia_theo_video,
    dem_theo_loai,
    doc_jsonl,
    kiem_schema,
    tach_tap_sach,
)


def _qa(i: int, video: str) -> dict:
    return {
        "id": str(i),
        "loai_truy_van": "hoi_dap",
        "cau_hoi": "Biển hiệu ghi gì?",
        "cau_tra_loi": "Xã Giang Ly",
        "video_id": video,
        "frame_start": 100,
        "frame_end": 120,
    }


def _kis(i: int, video: str) -> dict:
    return {
        "id": str(i),
        "loai_truy_van": "mo_ta",
        "cau_hoi": "Một người đi xe đạp",
        "video_id": video,
        "frame_start": 200,
        "frame_end": 250,
    }


def _trake(i: int, video: str) -> dict:
    return {
        "id": str(i),
        "loai_truy_van": "chuoi_su_kien",
        "cau_hoi": "chạy đà, giậm nhảy, tiếp đất",
        "video_id": video,
        "cac_giai_doan": [
            {"su_kien": "chạy đà", "frame_start": 10, "frame_end": 20},
            {"su_kien": "giậm nhảy", "frame_start": 30, "frame_end": 40},
        ],
    }


def test_doc_jsonl_bao_dung_dong_hong(tmp_path):
    tep = tmp_path / "dev.jsonl"
    tep.write_text(json.dumps(_qa(1, "L00_V001")) + "\n{hong", encoding="utf-8")
    with pytest.raises(LoiDongBang, match="dòng 2"):
        doc_jsonl(tep)


def test_schema_tu_choi_qa_khong_co_dap_an():
    q = _qa(1, "L00_V001")
    q["cau_tra_loi"] = ""
    with pytest.raises(LoiDongBang, match="cau_tra_loi rỗng"):
        kiem_schema([q])


def test_schema_tu_choi_event_nguoc():
    q = _trake(1, "L00_V001")
    q["cac_giai_doan"][0]["frame_start"] = 99
    with pytest.raises(LoiDongBang, match="frame_start > frame_end"):
        kiem_schema([q])


def test_tap_sach_luon_ghi_ly_do_cau_bi_loai():
    cau = [_qa(1, "L00_V001"), _qa(2, "L00_V002")]

    def kiem(q):
        return (True, "ok") if q["video_id"] == "L00_V001" else (False, "thieu_anh")

    sach, loai = tach_tap_sach(cau, kiem)
    assert [q["id"] for q in sach] == ["1"]
    assert loai == [
        {
            "id": "2",
            "loai_truy_van": "hoi_dap",
            "video_id": "L00_V002",
            "reason": "thieu_anh",
        }
    ]


def test_split_tat_dinh_va_khong_ro_ri_video():
    cau = [
        _qa(1, "L00_V001"),
        _qa(2, "L00_V002"),
        _qa(3, "L00_V003"),
        _kis(4, "L00_V001"),
        _kis(5, "L00_V004"),
        _trake(6, "L00_V005"),
        _trake(7, "L00_V006"),
    ]
    tune_1, hold_1 = chia_theo_video(cau, ti_le_holdout=0.3, seed="fixed")
    tune_2, hold_2 = chia_theo_video(cau, ti_le_holdout=0.3, seed="fixed")
    assert [q["id"] for q in tune_1] == [q["id"] for q in tune_2]
    assert [q["id"] for q in hold_1] == [q["id"] for q in hold_2]
    assert {q["video_id"] for q in tune_1}.isdisjoint(
        {q["video_id"] for q in hold_1}
    )


def test_loai_co_hai_video_xuat_hien_o_ca_tune_va_holdout():
    cau = [
        _qa(1, "L00_V001"),
        _qa(2, "L00_V002"),
        _kis(3, "L00_V001"),
        _kis(4, "L00_V003"),
        _trake(5, "L00_V004"),
        _trake(6, "L00_V005"),
    ]
    tune, hold = chia_theo_video(cau, ti_le_holdout=0.4, seed="fixed")
    for loai in ("hoi_dap", "mo_ta", "chuoi_su_kien"):
        assert any(q["loai_truy_van"] == loai for q in tune)
        assert any(q["loai_truy_van"] == loai for q in hold)


def test_dem_theo_loai_luon_co_du_ba_khoa():
    assert dem_theo_loai([_qa(1, "L00_V001")]) == {
        "chuoi_su_kien": 0,
        "hoi_dap": 1,
        "mo_ta": 0,
    }
