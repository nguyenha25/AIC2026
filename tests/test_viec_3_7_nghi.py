"""
Việc 3 (bảng tra ngược vật thể) và Việc 7 (khoá gộp RRF) — Nghi.

Chỉ import mã của Nghi — chạy được ngay cả khi các gói khác CHƯA trộn.
Đây là lý do bộ test được tách theo chủ: một tệp test chung sẽ nạp cả 11 việc
ở cấp module, nên pytest hỏng ngay lúc thu thập nếu thiếu bất kỳ gói nào,
và người mới trộn một nhánh không chạy nổi test của chính mình.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json
import sqlite3

import pytest

# ===========================================================================
# VIỆC 7 — khoá gộp RRF
# ===========================================================================

from aic2026.rank.fuse import khoa_gop, reciprocal_rank_fusion  # noqa: E402


def _muc(vid, n, frame_idx, pts_time, score=0.0):
    return {
        "video_id": vid,
        "n": n,
        "frame_idx": frame_idx,
        "pts_time": pts_time,
        "score": score,
    }


def test_khoa_gop_dung_pts_time_khong_dung_frame_idx():
    a = _muc("L21_V001", 10, 500, 20.020)
    b = _muc("L21_V001", 11, 500, 20.520)   # CÙNG frame_idx, KHÁC pts_time
    assert khoa_gop(a) != khoa_gop(b)


def test_hai_keyframe_trung_frame_idx_khong_bi_gop_lam_mot():
    """Đây chính là lỗi Việc 7 phải sửa.

    Hai tấm ảnh khác nhau làm tròn về cùng frame_idx. Bản cũ gộp chúng thành
    một dòng và tấm sau biến mất. Bản mới phải giữ đủ HAI dòng.
    """
    ket_qua = reciprocal_rank_fusion(
        {
            "clip": [
                _muc("L21_V001", 10, 500, 20.020),
                _muc("L21_V001", 11, 500, 20.520),
            ]
        }
    )
    assert len(ket_qua) == 2
    assert {r["n"] for r in ket_qua} == {10, 11}


def test_cung_mot_tam_o_hai_nhanh_thi_gop_lam_mot():
    ket_qua = reciprocal_rank_fusion(
        {
            "clip": [_muc("L21_V001", 10, 500, 20.020)],
            "ocr": [_muc("L21_V001", 10, 500, 20.020)],
        }
    )
    assert len(ket_qua) == 1
    assert ket_qua[0]["ranks"] == {"clip": 1, "ocr": 1}


def test_lech_chu_so_cuoi_van_gop_lam_mot():
    """float64 từ hai đường đọc khác nhau lệch ở chữ số cuối."""
    ket_qua = reciprocal_rank_fusion(
        {
            "clip": [_muc("L21_V001", 10, 500, 20.02)],
            "ocr": [_muc("L21_V001", 10, 500, 20.020000000000003)],
        }
    )
    assert len(ket_qua) == 1


def test_khu_trung_trong_cung_mot_nhanh():
    """Một nhánh trả cùng một tấm hai lần chỉ được cộng điểm MỘT lần."""
    mot_lan = reciprocal_rank_fusion(
        {"asr": [_muc("L21_V001", 10, 500, 20.02)]}
    )
    hai_lan = reciprocal_rank_fusion(
        {
            "asr": [
                _muc("L21_V001", 10, 500, 20.02),
                _muc("L21_V001", 10, 500, 20.02),
            ]
        }
    )
    assert mot_lan[0]["score"] == pytest.approx(hai_lan[0]["score"])


def test_giu_lai_n_va_pts_time_cho_buoc_loc_trung():
    ket_qua = reciprocal_rank_fusion({"clip": [_muc("L21_V001", 10, 500, 20.02)]})
    assert ket_qua[0]["n"] == 10
    assert ket_qua[0]["pts_time"] == pytest.approx(20.02)
    assert ket_qua[0]["frame_idx"] == 500


def test_thieu_pts_time_thi_lui_ve_khoa_cu_va_bao_cao():
    from aic2026.rank.fuse import CANH_BAO_THIEU_PTS

    ket_qua = reciprocal_rank_fusion(
        {"ocr": [{"video_id": "L21_V001", "frame_idx": 500}]}
    )
    assert len(ket_qua) == 1
    assert CANH_BAO_THIEU_PTS == {"ocr": 1}


def test_thu_tu_tat_dinh_khi_diem_bang_nhau():
    dau_vao = {
        "clip": [
            _muc("L22_V002", 1, 100, 4.0),
            _muc("L21_V001", 1, 100, 4.0),
        ]
    }
    lan_1 = [(r["video_id"], r["pts_time"]) for r in reciprocal_rank_fusion(dau_vao)]
    lan_2 = [(r["video_id"], r["pts_time"]) for r in reciprocal_rank_fusion(dau_vao)]
    assert lan_1 == lan_2


def test_trong_so_lam_doi_thu_hang():
    dau_vao = {
        "clip": [_muc("A", 1, 1, 1.0), _muc("B", 1, 2, 2.0)],
        "ocr": [_muc("B", 1, 2, 2.0), _muc("A", 1, 1, 1.0)],
    }
    nghieng_clip = reciprocal_rank_fusion(dau_vao, weights={"clip": 1.0, "ocr": 0.1})
    nghieng_ocr = reciprocal_rank_fusion(dau_vao, weights={"clip": 0.1, "ocr": 1.0})
    assert nghieng_clip[0]["video_id"] == "A"
    assert nghieng_ocr[0]["video_id"] == "B"



# ===========================================================================
# VIỆC 3 — bảng tra ngược vật thể
# ===========================================================================

from aic2026.index import objects_index  # noqa: E402


def test_doc_mot_tep_loc_theo_nguong(tmp_path):
    tep = tmp_path / "0001.json"
    tep.write_text(
        json.dumps(
            {
                "detection_class_entities": ["Person", "Hat", "Piano"],
                "detection_scores": [0.95, 0.30, 0.55],
            }
        ),
        encoding="utf-8",
    )
    nhan, dem = objects_index.doc_mot_tep(tep, nguong=0.4)
    assert nhan == ["Person", "Piano"]        # Hat 0.30 bị loại
    assert dem == {"Person": 1, "Piano": 1}


def test_doc_mot_tep_hong_khong_lam_dung_ca_me(tmp_path):
    tep = tmp_path / "0002.json"
    tep.write_text("{ khong phai json", encoding="utf-8")
    assert objects_index.doc_mot_tep(tep) == ([], {})


def test_cum_dai_thang_cum_ngan_va_an_luon_doan_da_khop():
    """"bánh mì" khớp xong thì "bánh" KHÔNG được khớp lại trên cùng đoạn chữ.

    Nhãn thừa làm truy vấn AND không khung nào thoả, buộc phải lùi về OR —
    chất lượng tụt mà không có dấu hiệu gì.
    """
    bang = objects_index.BangNhan({"bánh": ["Cake"], "bánh mì": ["Bread"]})
    assert bang.tim_nhan("BA Ổ BÁNH MÌ") == ["Bread"]
    assert bang.tim_nhan("một cái bánh sinh nhật") == ["Cake"]
    assert bang.tim_nhan("bánh mì và bánh ngọt") == ["Bread", "Cake"]
    assert bang.tim_nhan("không có gì") == []


def test_ranh_gioi_tu_khong_khop_chuoi_con():
    bang = objects_index.BangNhan({"cam": ["Orange"]})
    assert bang.tim_nhan("một chiếc camera") == []
    assert bang.tim_nhan("quả cam") == ["Orange"]


def test_nap_va_tra_cuu_objects_fts(tmp_path, monkeypatch):
    db = tmp_path / "text.sqlite"
    kho = objects_index.ObjectSearchIndex(
        db, bang_nhan=objects_index.BangNhan({"trống": ["Drum"], "đàn piano": ["Piano"]})
    )

    with sqlite3.connect(db) as conn:
        conn.executemany(
            f"INSERT INTO {objects_index.BANG_OBJECT} "
            "(video_id, n, frame_idx, pts_time, dem_json, nhan) VALUES (?,?,?,?,?,?)",
            [
                ("L21_V001", 1, 100, 4.0, '{"Drum": 1, "Piano": 1}', "Drum Piano"),
                ("L21_V001", 2, 200, 8.0, '{"Drum": 1}', "Drum"),
                ("L22_V002", 1, 50, 2.0, '{"Person": 3}', "Person Person Person"),
            ],
        )
        conn.commit()

    ca_hai = kho.tra_theo_nhan(["Drum", "Piano"], bat_buoc_du=True)
    assert len(ca_hai) == 1
    assert ca_hai[0]["n"] == 1
    assert ca_hai[0]["frame_idx"] == 100
    assert ca_hai[0]["source"] == "object"

    tu_cau_viet = kho.tra_bang_cau_viet("có bộ trống và đàn piano")
    assert [r["n"] for r in tu_cau_viet] == [1]

    assert kho.tra_bang_cau_viet("một cảnh không có vật thể nào trong bảng") == []


def test_nhan_ban_theo_so_luong_nang_hang_khung_nhieu_vat(tmp_path):
    """Khung có 3 người phải xếp trên khung có 1 người."""
    db = tmp_path / "text.sqlite"
    kho = objects_index.ObjectSearchIndex(db, bang_nhan=objects_index.BangNhan({}))
    with sqlite3.connect(db) as conn:
        conn.executemany(
            f"INSERT INTO {objects_index.BANG_OBJECT} "
            "(video_id, n, frame_idx, pts_time, dem_json, nhan) VALUES (?,?,?,?,?,?)",
            [
                ("A", 1, 10, 1.0, '{"Person": 1}', "Person Car Tree Building Sky"),
                ("A", 2, 20, 2.0, '{"Person": 3}', "Person Person Person"),
            ],
        )
        conn.commit()
    ket_qua = kho.tra_theo_nhan(["Person"])
    assert ket_qua[0]["n"] == 2


def test_n_tu_ten_tep():
    assert objects_index._n_tu_ten_tep(Path("0047.json")) == 47
    assert objects_index._n_tu_ten_tep(Path("47.json")) == 47


# ---------------------------------------------------------------------------
# Cụm phải gạch trước khi tra nhãn — phát hiện từ lần chạy thật 873 video
# ---------------------------------------------------------------------------

def test_bo_qua_cum_gay_hieu_nham():
    """Ranh giới từ KHÔNG cứu được: "nhà" trong "trong nhà" là từ trọn vẹn."""
    bang = objects_index.BangNhan(
        {"nhà": ["House"], "kéo": ["Scissors"], "kính": ["Glasses"]},
        bo_qua=["trong nhà", "nhà báo", "kéo dài", "kính thưa"],
    )
    assert bang.tim_nhan("một cảnh quay trong nhà lúc trời mưa") == []
    assert bang.tim_nhan("một nhà báo phỏng vấn") == []
    assert bang.tim_nhan("buổi lễ kéo dài ba tiếng") == []
    assert bang.tim_nhan("kính thưa quý vị") == []

    # nhưng nghĩa vật thể thật thì VẪN phải nhận ra
    assert bang.tim_nhan("ngôi nhà mái đỏ") == ["House"]
    assert bang.tim_nhan("một cái kéo màu đen") == ["Scissors"]


def test_bang_nhan_that_doc_duoc_ca_hai_muc():
    """config/nhan_vat_the.yaml phải có cả anh_xa lẫn bo_qua."""
    bang = objects_index.BangNhan.nap()
    assert len(bang) > 100
    assert bang.tim_nhan("biển số xe") == ["Vehicle registration plate"]
    assert bang.tim_nhan("cảnh quay trong nhà") == []
