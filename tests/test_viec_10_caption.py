"""Việc 10 — caption đầy đủ, chạy tiếp được và tra cứu như nhánh độc lập."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aic2026.enrich import caption as cap
from scripts import verify_task10_acceptance as verify


def _moc(i: int):
    return SimpleNamespace(
        video_id="L00_V001",
        i=i,
        n=i + 1,
        frame_idx=100 + i * 25,
        pts_time=float(i),
    )


class _BoSinhGia:
    ten_mo_hinh = "fake/blip"
    kich_thuoc_lo = 2

    def __init__(self):
        self.cac_lo = []

    def sinh(self, duong_dan):
        self.cac_lo.append([p.name for p in duong_dan])
        return [f"caption {p.stem}" if p.exists() else "" for p in duong_dan]


def _gan_duong_dan(monkeypatch, tmp_path, so_anh=3):
    import aic2026.kf_index as kf
    import aic2026.paths as paths

    anh = tmp_path / "images"
    anh.mkdir()
    for n in range(1, so_anh + 1):
        (anh / f"{n}.jpg").write_bytes(b"fake")
    captions = tmp_path / "captions"
    monkeypatch.setattr(kf, "so_hang", lambda _v: 3)
    monkeypatch.setattr(kf, "hang_sang_moc", lambda _v, i: _moc(i))
    monkeypatch.setattr(paths, "captions_file", lambda v: captions / f"{v}.jsonl")
    monkeypatch.setattr(paths, "keyframe_image", lambda _v, n: anh / f"{n}.jpg")
    return captions


def test_doc_partial_bo_qua_dong_json_bi_dut(tmp_path):
    tep = tmp_path / "x.partial"
    tep.write_text('{"n": 1, "caption": "ok"}\n{"n":', encoding="utf-8")
    assert cap._doc_ban_ghi_theo_n(tep) == {1: {"n": 1, "caption": "ok"}}


def test_sinh_caption_chay_tiep_chi_lam_phan_thieu(tmp_path, monkeypatch):
    captions = _gan_duong_dan(monkeypatch, tmp_path)
    captions.mkdir()
    tam = captions / "L00_V001.jsonl.partial"
    tam.write_text(
        json.dumps(cap._tao_ban_ghi("L00_V001", _moc(0), "old caption", "fake/blip", "ok"))
        + "\n",
        encoding="utf-8",
    )
    bo = _BoSinhGia()

    kq = cap.sinh_cho_video("L00_V001", bo, in_tien_do=False)

    assert kq["tiep_tuc_tu"] == 1
    assert sum(len(x) for x in bo.cac_lo) == 2
    dong = [json.loads(x) for x in (captions / "L00_V001.jsonl").read_text().splitlines()]
    assert [x["n"] for x in dong] == [1, 2, 3]
    assert dong[0]["caption"] == "old caption"
    assert all(x["status"] == "ok" for x in dong)
    assert not tam.exists()


def test_moi_keyframe_co_mot_ban_ghi_ke_ca_anh_thieu(tmp_path, monkeypatch):
    captions = _gan_duong_dan(monkeypatch, tmp_path, so_anh=2)
    bo = _BoSinhGia()
    kq = cap.sinh_cho_video("L00_V001", bo, in_tien_do=False)
    dong = [json.loads(x) for x in (captions / "L00_V001.jsonl").read_text().splitlines()]

    assert len(dong) == 3
    assert kq["so_thieu_anh"] == 1
    assert dong[2]["caption"] == ""
    assert dong[2]["status"] == "missing_image"


def test_nap_fts_bo_caption_rong_va_tra_or(tmp_path):
    thu_muc = tmp_path / "captions"
    thu_muc.mkdir()
    (thu_muc / "L00_V001.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"video_id": "L00_V001", "n": 1, "frame_idx": 100,
                            "pts_time": 1.0, "caption": "a man holds a red umbrella"}),
                json.dumps({"video_id": "L00_V001", "n": 2, "frame_idx": 125,
                            "pts_time": 2.0, "caption": ""}),
            ]
        ),
        encoding="utf-8",
    )
    kho = cap.CaptionSearchIndex(tmp_path / "fts.sqlite", tu_dich_cau_hoi=False)
    assert kho.nap_tu_thu_muc(thu_muc)["so_dong_nap"] == 1
    ra = kho.tra_cuu("the umbrella, in rain", top_k=10)
    assert len(ra) == 1
    assert ra[0]["video_id"] == "L00_V001"
    assert ra[0]["source"] == "caption"


def test_bat_buoc_dich_khong_nuot_loi(tmp_path, monkeypatch):
    import aic2026.query_expand as qe

    def hong(*_args, **_kwargs):
        raise RuntimeError("dịch hỏng")

    monkeypatch.setattr(qe, "mo_rong", hong)
    kho = cap.CaptionSearchIndex(
        tmp_path / "fts.sqlite", nguon_dich="marian", bat_buoc_dich=True
    )
    with pytest.raises(RuntimeError, match="dịch hỏng"):
        kho.tra_cuu("một người cầm ô")


def test_nghiem_thu_phat_hien_trung_n(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "CAPTIONS_DIR", tmp_path)
    monkeypatch.setattr(verify, "so_hang", lambda _v: 3)
    monkeypatch.setattr(verify, "hang_sang_moc", lambda _v, i: _moc(i))
    ban_ghi = [cap._tao_ban_ghi("L00_V001", _moc(i), "caption", "fake", "ok") for i in range(3)]
    ban_ghi.append(dict(ban_ghi[0]))
    (tmp_path / "L00_V001.jsonl").write_text(
        "\n".join(json.dumps(x) for x in ban_ghi), encoding="utf-8"
    )
    kq = verify.kiem_video("L00_V001")
    assert not kq["dat"]
    assert any("trùng n" in x for x in kq["loi"])
