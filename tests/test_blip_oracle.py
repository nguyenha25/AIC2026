from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.blip_oracle as m


def test_doc_qa_chi_lay_hoi_dap(tmp_path: Path):
    p = tmp_path / "dev.jsonl"
    rows = [
        {"id": "01", "loai_truy_van": "mo_ta"},
        {"id": "02", "loai_truy_van": "hoi_dap", "cau_hoi": "x"},
        {"id": "03", "loai_truy_van": "chuoi_su_kien"},
    ]
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    assert [x["id"] for x in m.doc_qa(p)] == ["02"]


def test_doc_qa_json_loi(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": 1}\n{bad json}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        m.doc_qa(p)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("NÆ°á»›c nghá»‡ tÆ°Æ¡i!", "nÆ°á»›c nghá»‡ tÆ°Æ¡i"),
        ("  Bá»™t   canh ", "bá»™t canh"),
        ("ABC", "abc"),
    ],
)
def test_exact_match_chuan_hoa(a: str, b: str):
    assert m.exact_match(a, b)


def test_exact_match_khong_dich():
    assert not m.exact_match("NÆ°á»›c nghá»‡ tÆ°Æ¡i", "turmeric water")


def test_chon_frame_oracle_nearest_nam_trong_range():
    row = SimpleNamespace(video_id="L26_V495", n=97, pts_time=148.8, fps=25.0, frame_idx=3720)

    class FM:
        def nearest_by_time(self, pts):
            return row

    q = {"frame_start": 3675, "frame_end": 3806, "pts_time": 147.0}
    assert m.chon_frame_oracle(q, FM()) is row


def test_chon_frame_oracle_fallback_midpoint():
    out = SimpleNamespace(video_id="V", n=10, pts_time=4.0, fps=25.0, frame_idx=100)
    inside = SimpleNamespace(video_id="V", n=11, pts_time=6.0, fps=25.0, frame_idx=150)

    class FM:
        def __init__(self):
            self.calls = 0

        def nearest_by_time(self, pts):
            self.calls += 1
            return out if self.calls == 1 else inside

    q = {"frame_start": 140, "frame_end": 160, "pts_time": 4.0}
    assert m.chon_frame_oracle(q, FM()) is inside


def test_chon_frame_oracle_none_neu_khong_co_frame_gt():
    out = SimpleNamespace(video_id="V", n=10, pts_time=4.0, fps=25.0, frame_idx=100)

    class FM:
        def nearest_by_time(self, pts):
            return out

    q = {"frame_start": 140, "frame_end": 160, "pts_time": 4.0}
    assert m.chon_frame_oracle(q, FM()) is None


def test_tim_anh_theo_n_3_chu_so(tmp_path: Path):
    d = tmp_path / "Keyframes_L26_e" / "L26_V495"
    d.mkdir(parents=True)
    p = d / "097.jpg"
    p.write_bytes(b"x")
    assert m.tim_anh_theo_n("L26_V495", 97, tmp_path) == p


def test_tim_anh_theo_n_4_chu_so(tmp_path: Path):
    d = tmp_path / "L26_V495"
    d.mkdir(parents=True)
    p = d / "0097.jpg"
    p.write_bytes(b"x")
    assert m.tim_anh_theo_n("L26_V495", 97, tmp_path) == p


def test_tong_hop():
    rows = [
        {"status": "ok", "intent": "khac", "exact_match": False, "semantic_match": True, "latency_ms": 100.0},
        {"status": "ok", "intent": "khac", "exact_match": True, "semantic_match": True, "latency_ms": 300.0},
        {"status": "missing", "reason": "keyframe_image"},
    ]
    s = m.tong_hop(rows)
    assert s["total_records"] == 3
    assert s["ok"] == 2
    assert s["overall"]["exact_rate"] == 0.5
    assert s["overall"]["semantic_rate"] == 1.0
    assert s["overall"]["avg_latency_ms"] == 200.0
    assert s["non_ok_reasons"] == {"keyframe_image": 1}

