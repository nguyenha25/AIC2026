from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import kiem_ke_evidence as audit


def _fake_question(
    *,
    query_id: str = "1",
    loai_truy_van: str = "hoi_dap",
    video_id: str = "L00_V001",
    cac_giai_doan: list[dict] | None = None,
) -> dict:
    q = {
        "id": query_id,
        "loai_truy_van": loai_truy_van,
        "video_id": video_id,
        "cau_hoi": "Câu hỏi kiểm thử",
    }
    if cac_giai_doan is not None:
        q["cac_giai_doan"] = cac_giai_doan
    return q


def test_doc_dev_tho_giu_nguyen_dong_va_parse_json(tmp_path: Path) -> None:
    tep = tmp_path / "dev_questions.jsonl"
    dong_1 = '{"id":"1","loai_truy_van":"hoi_dap","video_id":"L00_V001"}'
    dong_2 = '{"id":"2","loai_truy_van":"mo_ta","video_id":"L00_V002"}'
    tep.write_text(f"{dong_1}\n\n{dong_2}\n", encoding="utf-8")

    ket_qua = audit.doc_dev_tho(tep)

    assert len(ket_qua) == 2
    assert ket_qua[0][0] == dong_1
    assert ket_qua[0][1]["id"] == "1"
    assert ket_qua[1][0] == dong_2
    assert ket_qua[1][1]["video_id"] == "L00_V002"


def test_doc_dev_tho_bao_loi_khi_json_hong(tmp_path: Path) -> None:
    tep = tmp_path / "dev_questions.jsonl"
    tep.write_text('{"id":"1"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="không phải JSON hợp lệ"):
        audit.doc_dev_tho(tep)


def test_qa_ok_khi_co_keyframe_va_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "kiem_frame_map", lambda video_id: True)
    monkeypatch.setattr(audit, "co_keyframe_goc", lambda video_id: True)
    monkeypatch.setattr(audit, "co_ocr", lambda video_id: True)
    monkeypatch.setattr(audit, "co_asr", lambda video_id: False)

    q = _fake_question(loai_truy_van="hoi_dap")
    ket_qua = audit.kiem_ke_mot_cau(q)

    assert ket_qua["task"] == "qa"
    assert ket_qua["status"] == "ok"
    assert ket_qua["missing"] == []
    assert ket_qua["evidence"]["ocr"] is True
    assert ket_qua["evidence"]["asr"] is False


def test_qa_ok_khi_co_keyframe_va_asr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "kiem_frame_map", lambda video_id: True)
    monkeypatch.setattr(audit, "co_keyframe_goc", lambda video_id: True)
    monkeypatch.setattr(audit, "co_ocr", lambda video_id: False)
    monkeypatch.setattr(audit, "co_asr", lambda video_id: True)

    q = _fake_question(loai_truy_van="hoi_dap")
    ket_qua = audit.kiem_ke_mot_cau(q)

    assert ket_qua["status"] == "ok"
    assert ket_qua["missing"] == []


def test_qa_missing_khi_khong_co_ocr_lan_asr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "kiem_frame_map", lambda video_id: True)
    monkeypatch.setattr(audit, "co_keyframe_goc", lambda video_id: True)
    monkeypatch.setattr(audit, "co_ocr", lambda video_id: False)
    monkeypatch.setattr(audit, "co_asr", lambda video_id: False)

    q = _fake_question(loai_truy_van="hoi_dap")
    ket_qua = audit.kiem_ke_mot_cau(q)

    assert ket_qua["status"] == "missing"
    assert ket_qua["missing"] == ["ocr_hoac_asr"]


def test_qa_missing_frame_map_va_keyframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "kiem_frame_map", lambda video_id: False)
    monkeypatch.setattr(audit, "co_keyframe_goc", lambda video_id: False)
    monkeypatch.setattr(audit, "co_ocr", lambda video_id: True)
    monkeypatch.setattr(audit, "co_asr", lambda video_id: False)

    q = _fake_question(loai_truy_van="hoi_dap")
    ket_qua = audit.kiem_ke_mot_cau(q)

    assert ket_qua["status"] == "missing"
    assert "frame_map" in ket_qua["missing"]
    assert "keyframe_goc" in ket_qua["missing"]
    assert "ocr_hoac_asr" not in ket_qua["missing"]


def test_kis_khong_bat_buoc_ocr_hoac_asr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "kiem_frame_map", lambda video_id: True)
    monkeypatch.setattr(audit, "co_keyframe_goc", lambda video_id: True)
    monkeypatch.setattr(audit, "co_ocr", lambda video_id: False)
    monkeypatch.setattr(audit, "co_asr", lambda video_id: False)

    q = _fake_question(loai_truy_van="mo_ta")
    ket_qua = audit.kiem_ke_mot_cau(q)

    assert ket_qua["task"] == "kis"
    assert ket_qua["status"] == "ok"
    assert ket_qua["missing"] == []


def test_dense_phu_su_kien_danh_dau_tung_giai_doan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "_khung_dense_co_san",
        lambda video_id: {100, 150, 250, 400},
    )

    cac_giai_doan = [
        {
            "su_kien": "E1",
            "frame_start": 90,
            "frame_end": 110,
        },
        {
            "su_kien": "E2",
            "frame_start": 200,
            "frame_end": 260,
        },
        {
            "su_kien": "E3",
            "frame_start": 300,
            "frame_end": 350,
        },
    ]

    ket_qua = audit.dense_phu_su_kien("L00_V001", cac_giai_doan)

    assert [x["co_dense"] for x in ket_qua] == [True, True, False]
    assert ket_qua[0]["su_kien_thu"] == 0
    assert ket_qua[2]["frame_start"] == 300
    assert ket_qua[2]["frame_end"] == 350


def test_trake_ok_khi_moi_su_kien_deu_co_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "kiem_frame_map", lambda video_id: True)
    monkeypatch.setattr(audit, "co_keyframe_goc", lambda video_id: True)
    monkeypatch.setattr(audit, "co_ocr", lambda video_id: False)
    monkeypatch.setattr(audit, "co_asr", lambda video_id: False)
    monkeypatch.setattr(
        audit,
        "dense_phu_su_kien",
        lambda video_id, cac_giai_doan: [
            {
                "su_kien_thu": 0,
                "su_kien": "E1",
                "frame_start": 100,
                "frame_end": 120,
                "co_dense": True,
            },
            {
                "su_kien_thu": 1,
                "su_kien": "E2",
                "frame_start": 200,
                "frame_end": 220,
                "co_dense": True,
            },
        ],
    )

    q = _fake_question(
        loai_truy_van="chuoi_su_kien",
        cac_giai_doan=[
            {"su_kien": "E1", "frame_start": 100, "frame_end": 120},
            {"su_kien": "E2", "frame_start": 200, "frame_end": 220},
        ],
    )
    ket_qua = audit.kiem_ke_mot_cau(q)

    assert ket_qua["task"] == "trake"
    assert ket_qua["status"] == "ok"
    assert ket_qua["missing"] == []
    assert ket_qua["evidence"]["dense_du"] is True


def test_trake_missing_neu_chi_mot_su_kien_thieu_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "kiem_frame_map", lambda video_id: True)
    monkeypatch.setattr(audit, "co_keyframe_goc", lambda video_id: True)
    monkeypatch.setattr(audit, "co_ocr", lambda video_id: True)
    monkeypatch.setattr(audit, "co_asr", lambda video_id: True)
    monkeypatch.setattr(
        audit,
        "dense_phu_su_kien",
        lambda video_id, cac_giai_doan: [
            {
                "su_kien_thu": 0,
                "su_kien": "E1",
                "frame_start": 100,
                "frame_end": 120,
                "co_dense": True,
            },
            {
                "su_kien_thu": 1,
                "su_kien": "E2",
                "frame_start": 200,
                "frame_end": 220,
                "co_dense": False,
            },
        ],
    )

    q = _fake_question(
        loai_truy_van="chuoi_su_kien",
        cac_giai_doan=[
            {"su_kien": "E1", "frame_start": 100, "frame_end": 120},
            {"su_kien": "E2", "frame_start": 200, "frame_end": 220},
        ],
    )
    ket_qua = audit.kiem_ke_mot_cau(q)

    assert ket_qua["status"] == "missing"
    assert "dense" in ket_qua["missing"]
    assert ket_qua["evidence"]["dense_du"] is False


def test_trake_khong_co_giai_doan_thi_missing_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "kiem_frame_map", lambda video_id: True)
    monkeypatch.setattr(audit, "co_keyframe_goc", lambda video_id: True)
    monkeypatch.setattr(audit, "co_ocr", lambda video_id: False)
    monkeypatch.setattr(audit, "co_asr", lambda video_id: False)

    q = _fake_question(
        loai_truy_van="chuoi_su_kien",
        cac_giai_doan=[],
    )
    ket_qua = audit.kiem_ke_mot_cau(q)

    assert ket_qua["status"] == "missing"
    assert ket_qua["missing"] == ["dense"]
    assert ket_qua["evidence"]["dense_du"] is False


def test_schema_version_va_cac_truong_chinh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "kiem_frame_map", lambda video_id: True)
    monkeypatch.setattr(audit, "co_keyframe_goc", lambda video_id: True)
    monkeypatch.setattr(audit, "co_ocr", lambda video_id: True)
    monkeypatch.setattr(audit, "co_asr", lambda video_id: False)

    q = _fake_question(query_id="12", loai_truy_van="hoi_dap", video_id="L23_V024")
    ket_qua = audit.kiem_ke_mot_cau(q)

    assert ket_qua["schema_version"] == "1.1"
    assert ket_qua["query_id"] == "12"
    assert ket_qua["task"] == "qa"
    assert ket_qua["video_id"] == "L23_V024"
    assert ket_qua["error"] is None