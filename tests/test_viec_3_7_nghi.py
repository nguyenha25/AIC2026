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
    # Không lọc bỏ nữa — xếp theo độ phủ. Khung phủ ĐỦ phải đứng đầu.
    assert ca_hai[0]["n"] == 1 and ca_hai[0]["du_nhan"]
    assert ca_hai[0]["frame_idx"] == 100
    assert ca_hai[0]["source"] == "object"

    tu_cau_viet = kho.tra_bang_cau_viet("có bộ trống và đàn piano")
    # Khung phủ ĐỦ đứng đầu; khung phủ một phần vẫn được trả về phía sau
    assert tu_cau_viet[0]["n"] == 1 and tu_cau_viet[0]["du_nhan"]
    assert all(not r["du_nhan"] for r in tu_cau_viet[1:])

    assert kho.tra_bang_cau_viet("một cảnh không có vật thể nào trong bảng") == []


def test_nhan_ban_theo_so_luong_nang_hang_khung_nhieu_vat(tmp_path):
    """Với MỘT nhãn, khung có 3 người phải xếp trên khung có 1 người."""
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


# ---------------------------------------------------------------------------
# Việc 6 — trọng số tách theo dạng câu, nối vào mạch chính
# ---------------------------------------------------------------------------

def test_trong_so_doc_duoc_theo_dang(tmp_path, monkeypatch):
    """config/rrf_weights.yaml phải THẬT SỰ tới được chỗ gộp RRF.

    Trước đây trong_so_theo_dang() có tồn tại nhưng hop_nhat.tim_ung_vien_gop()
    vẫn dùng bảng gõ cứng — viết tệp yaml xong không có gì thay đổi.
    """
    import yaml

    from aic2026.rank import config as cfg

    tep = tmp_path / "rrf_weights.yaml"
    tep.write_text(
        yaml.safe_dump(
            {
                "mac_dinh": {"clip": 1.0, "ocr_fts": 1.0, "asr": 0.0},
                "theo_dang": {
                    "kis": {"clip": 1.0, "ocr_fts": 1.0, "asr": 0.0},
                    "qa": {"clip": 1.0, "ocr_fts": 0.0, "asr": 1.0},
                    "trake": None,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)

    kis = cfg.trong_so_theo_dang("kis")
    qa = cfg.trong_so_theo_dang("qa")

    # ASR chạy một mình được KIS 0,0000 nhưng QA 0,1667 — trọng số phải khác nhau
    assert kis["asr"] == 0.0 and qa["asr"] == 1.0
    # OCR ngược lại: KIS 0,3000, QA 0,0000
    assert kis["ocr_fts"] == 1.0 and qa["ocr_fts"] == 0.0

    # dạng chưa đo được thì lùi về bộ mặc định, KHÔNG bịa số
    assert cfg.trong_so_theo_dang("trake")["clip"] == 1.0


def test_khong_co_tep_thi_lui_ve_settings(tmp_path, monkeypatch):
    from aic2026.rank import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)     # thư mục rỗng
    ts = cfg.trong_so_theo_dang("kis")
    assert ts == cfg.trong_so_nguon()


def test_tim_ung_vien_gop_nhan_dang_cau():
    """Chữ ký phải có dang_cau, không thì mạch chính không truyền được dạng."""
    import inspect

    from aic2026.rank.hop_nhat import tim_ung_vien_gop

    assert "dang_cau" in inspect.signature(tim_ung_vien_gop).parameters


def test_khong_de_cung_trong_so_len_rrf_weights():
    """Truyền cứng trong_so=TRONG_SO_MAC_DINH là đè lên config/rrf_weights.yaml,
    khiến mọi con số đo được ở Việc 6 không tới được mạch chạy thật."""
    from pathlib import Path

    goc = Path(__file__).resolve().parents[1]
    for ten in ("scripts/benchmark_rrf.py", "src/aic2026/ui/app.py"):
        tep = goc / ten
        if not tep.exists():
            continue
        # Bỏ dòng chú thích trước khi soát: chữ trong chú thích là để GIẢI
        # THÍCH vì sao không dùng, không phải chỗ gọi hàm. Đã cắn bốn lần với
        # đúng kiểu test này.
        ma = "\n".join(
            d for d in tep.read_text(encoding="utf-8").splitlines()
            if not d.lstrip().startswith("#")
        )
        assert "trong_so=TRONG_SO_MAC_DINH" not in ma, (
            f"{ten} còn đè cứng trọng số lên rrf_weights.yaml"
        )


def test_cay_dan_khong_bi_hieu_thanh_cai_cay():
    """Đề thật p1 câu 25: "một CÂY ĐÀN piano" rút ra cả Tree/Houseplant/Plant.

    Một nhãn thừa làm truy vấn AND rỗng, hệ thống lùi về OR, và không dòng nào
    trong 20 kết quả có Drum — mà Drum mới là thứ phân biệt câu này.
    """
    bang = objects_index.BangNhan.nap()

    nhan = bang.tim_nhan("bộ trống cơ màu đỏ và một cây đàn piano")
    assert "Drum" in nhan and "Piano" in nhan
    assert not {"Tree", "Houseplant", "Plant"} & set(nhan)

    # nghĩa "cái cây" thật thì VẪN phải nhận ra
    assert "Tree" in bang.tim_nhan("hàng cây xanh hai bên đường")


def test_bao_khi_phai_lui_ve_OR(tmp_path):
    """Lùi âm thầm từ AND sang OR làm kết quả chỉ khớp một phần nhãn mà nhìn
    vào danh sách không phân biệt được."""
    import sqlite3

    db = tmp_path / "text.sqlite"
    kho = objects_index.ObjectSearchIndex(db, bang_nhan=objects_index.BangNhan({}))
    with sqlite3.connect(db) as conn:
        conn.execute(
            f"INSERT INTO {objects_index.BANG_OBJECT} "
            "(video_id, n, frame_idx, pts_time, dem_json, nhan) VALUES (?,?,?,?,?,?)",
            ("A", 1, 10, 1.0, '{"Piano": 1}', "Piano"),
        )
        conn.commit()

    kho.tra_theo_nhan(["Piano"])
    assert objects_index.DA_LUI_VE_OR is False

    kho.tra_theo_nhan(["Piano", "Drum"])          # không khung nào có đủ hai
    assert objects_index.DA_LUI_VE_OR is True


def test_xep_theo_DO_PHU_NHAN_khong_theo_tan_suat(tmp_path):
    """Đề p1 câu 25: trong 164.812 khung, KHÔNG khung nào có cả Piano lẫn Drum.

    AND rỗng tuyệt đối. Lùi về OR thì BM25 xếp khung có BA cái piano lên trên
    khung có MỘT piano và MỘT trống — đúng ngược thứ câu hỏi cần.
    """
    import sqlite3

    db = tmp_path / "text.sqlite"
    kho = objects_index.ObjectSearchIndex(db, bang_nhan=objects_index.BangNhan({}))
    with sqlite3.connect(db) as conn:
        conn.executemany(
            f"INSERT INTO {objects_index.BANG_OBJECT} "
            "(video_id, n, frame_idx, pts_time, dem_json, nhan) VALUES (?,?,?,?,?,?)",
            [
                ("nhieu_piano", 1, 10, 1.0, '{"Piano": 3}', "Piano Piano Piano"),
                ("du_ca_hai", 1, 20, 2.0, '{"Piano": 1, "Drum": 1}', "Piano Drum"),
                ("chi_trong", 1, 30, 3.0, '{"Drum": 1}', "Drum"),
            ],
        )
        conn.commit()

    ra = kho.tra_theo_nhan(["Piano", "Drum"])
    assert ra[0]["video_id"] == "du_ca_hai", "khung phủ đủ phải đứng đầu"
    assert ra[0]["so_nhan_khop"] == 2 and ra[0]["du_nhan"]
    assert all(r["so_nhan_khop"] == 1 for r in ra[1:])

    # phủ một phần vẫn được trả về, không bị lọc bỏ như bản AND cũ
    assert len(ra) == 3


def test_do_phu_mot_phan_van_xep_dung_thu_tu(tmp_path):
    import sqlite3

    db = tmp_path / "text.sqlite"
    kho = objects_index.ObjectSearchIndex(db, bang_nhan=objects_index.BangNhan({}))
    with sqlite3.connect(db) as conn:
        conn.executemany(
            f"INSERT INTO {objects_index.BANG_OBJECT} "
            "(video_id, n, frame_idx, pts_time, dem_json, nhan) VALUES (?,?,?,?,?,?)",
            [
                ("hai_tren_ba", 1, 10, 1.0, "{}", "Piano Drum"),
                ("mot_tren_ba", 1, 20, 2.0, "{}", "Piano Piano Piano Piano"),
            ],
        )
        conn.commit()

    ra = kho.tra_theo_nhan(["Piano", "Drum", "Guitar"])
    assert [r["video_id"] for r in ra] == ["hai_tren_ba", "mot_tren_ba"]
    assert objects_index.DA_LUI_VE_OR is True     # không khung nào phủ đủ


def test_do_phu_dem_theo_KHAI_NIEM_khong_theo_nhan():
    """"đàn piano" trỏ tới CẢ Piano LẪN Musical keyboard — bộ nhận dạng gán hai
    nhãn cho CÙNG MỘT vật thể.

    Đếm theo nhãn tiếng Anh thì khung chỉ có cây piano đạt 2/3 còn khung có cái
    trống chỉ được 1/3, nên piano luôn thắng dù câu hỏi đòi cả hai. Đề p1 câu 25
    dính đúng chuyện này: 20/20 kết quả đầu đều là piano, không có trống nào.
    """
    bang = objects_index.BangNhan.nap()
    nhom = bang.tim_nhom("bộ trống cơ màu đỏ và một cây đàn piano")

    ten = [t for t, _ in nhom]
    assert len(nhom) == 2, f"phải ra ĐÚNG hai khái niệm, ra {ten}"
    assert any("trống" in t for t in ten)
    assert any("piano" in t for t in ten)

    # Piano và Musical keyboard phải nằm CÙNG một khái niệm
    nhom_piano = next(ds for t, ds in nhom if "piano" in t)
    assert {"Piano", "Musical keyboard"} <= set(nhom_piano)


def test_khung_du_hai_khai_niem_thang_khung_nhieu_piano(tmp_path):
    import sqlite3

    db = tmp_path / "text.sqlite"
    kho = objects_index.ObjectSearchIndex(db, bang_nhan=objects_index.BangNhan.nap())
    with sqlite3.connect(db) as conn:
        conn.executemany(
            f"INSERT INTO {objects_index.BANG_OBJECT} "
            "(video_id, n, frame_idx, pts_time, dem_json, nhan) VALUES (?,?,?,?,?,?)",
            [
                ("chi_piano", 1, 10, 1.0, "{}",
                 "Piano Piano Piano Musical keyboard Musical keyboard"),
                ("trong_va_piano", 1, 20, 2.0, "{}", "Piano Drum"),
                ("chi_trong", 1, 30, 3.0, "{}", "Drum Drum"),
            ],
        )
        conn.commit()

    ra = kho.tra_bang_cau_viet("bộ trống cơ màu đỏ và một cây đàn piano")
    assert ra[0]["video_id"] == "trong_va_piano"
    assert ra[0]["so_nhan_khop"] == 2 and ra[0]["du_nhan"]
    assert all(r["so_nhan_khop"] == 1 for r in ra[1:])


# ---------------------------------------------------------------------------
# Việc 3 — BỘ LỌC (đúng kiến trúc checklist yêu cầu)
# ---------------------------------------------------------------------------

def _kho_gia(tmp_path, so_khung=5000):
    """Kho mô phỏng phân bố thật: Person phổ biến, Drum hiếm."""
    import json
    import random
    import sqlite3

    db = tmp_path / "text.sqlite"
    kho = objects_index.ObjectSearchIndex(db, bang_nhan=objects_index.BangNhan.nap())

    rng = random.Random(7)
    lo = []
    for i in range(so_khung):
        d = {}
        if rng.random() < 0.42:
            d["Person"] = rng.randint(1, 3)
        if rng.random() < 0.30:
            d["Human face"] = 1
        if rng.random() < 0.02:
            d.update({"Piano": 1, "Musical keyboard": 1})
        if rng.random() < 0.01:
            d["Drum"] = 1
        nhan = " ".join(k for k, v in d.items() for _ in range(v))
        lo.append(
            (f"V{i // 200:03d}", i % 200 + 1, i * 30, round(i * 1.2, 3),
             json.dumps(d), nhan)
        )
    with sqlite3.connect(db) as conn:
        conn.executemany(
            f"INSERT INTO {objects_index.BANG_OBJECT} "
            "(video_id, n, frame_idx, pts_time, dem_json, nhan) VALUES (?,?,?,?,?,?)",
            lo,
        )
        conn.commit()
    return kho


def test_tan_suat_KHONG_cat_nat_nhan_nhieu_chu(tmp_path):
    """Cột `nhan` ngăn nhãn bằng dấu cách, nên .split() cắt "Musical keyboard"
    thành "Musical" và "keyboard".

    Sai với mọi nhãn nhiều chữ: Human face, Sun hat, Vehicle registration
    plate. Hệ quả nặng hơn: phép đếm ĐỘ PHỦ chưa bao giờ khớp được nhãn nhiều
    chữ, vì nó so cả cụm với từng từ rời.
    """
    ts = _kho_gia(tmp_path).tan_suat_nhan()

    assert "Musical keyboard" in ts and "Musical" not in ts and "keyboard" not in ts
    assert "Human face" in ts and "face" not in ts


def test_loc_thu_hep_manh_voi_khai_niem_hiem(tmp_path):
    kho = _kho_gia(tmp_path)
    gt = kho.giai_thich_loc("một bộ trống và cây đàn piano", toi_thieu_giu_lai=50)

    assert gt["co_loc"]
    assert gt["ti_le_giu_lai"] < 0.10, "khái niệm hiếm phải thu hẹp mạnh"


def test_KHONG_loc_khi_moi_khai_niem_deu_pho_bien(tmp_path):
    """`người`, `áo`, `mặt` có ở gần nửa kho — lọc theo chúng vô nghĩa.

    Thà mất cơ hội thu hẹp còn hơn giết mất đáp án đúng.
    """
    kho = _kho_gia(tmp_path)
    assert kho.khung_chua("người đàn ông mặc áo trắng") is None


def test_KHONG_loc_khi_cau_khong_nhac_vat_the(tmp_path):
    kho = _kho_gia(tmp_path)
    assert kho.khung_chua("cảnh quay lúc hoàng hôn trên biển") is None


def test_KHONG_loc_khi_tap_qua_hep(tmp_path):
    """Bộ nhận dạng bỏ sót nhiều — tập lọc quá hẹp thì rủi ro lớn hơn lợi ích."""
    kho = _kho_gia(tmp_path)
    assert kho.khung_chua("một bộ trống", toi_thieu_giu_lai=10**9) is None


def test_hop_khong_phai_giao(tmp_path):
    """Khung chỉ cần có MỘT khái niệm hiếm là được giữ.

    Giao là quá chặt: đo thật trên kho 164.812 khung, KHÔNG khung nào có cả
    Piano lẫn Drum. Dùng giao thì bộ lọc trả về tập rỗng.
    """
    kho = _kho_gia(tmp_path)
    chi_trong = kho.khung_chua("một bộ trống", toi_thieu_giu_lai=1)
    ca_hai = kho.khung_chua("một bộ trống và cây đàn piano", toi_thieu_giu_lai=1)

    assert chi_trong and ca_hai
    assert len(ca_hai) > len(chi_trong), "phép HỢP phải cho tập LỚN hơn"


def test_tim_ung_vien_gop_co_co_loc_vat_the():
    import inspect

    from aic2026.rank.hop_nhat import tim_ung_vien_gop

    tham_so = inspect.signature(tim_ung_vien_gop).parameters
    assert "loc_vat_the" in tham_so
    assert "dung_object" in tham_so, "hai cờ phải ĐỘC LẬP để đo riêng từng cái"
