"""
BỘ KIỂM TRA TỰ ĐỘNG — TÁM MỤC (Task 6).

Chạy:  pytest -q
Riêng tệp này tám mục. Cả thư mục tests/ hiện là 18 passed.
Bốn máy đều phải ra cùng con số. Ai không pass thì BÁO NHÓM, đừng tự cài thêm gói.

Tám mục này KHÔNG cần dữ liệu BTC — chạy được ngay sau khi clone.
Đó là chủ ý: bộ test kiểm BỘ KHUNG, không kiểm dữ liệu. Dữ liệu do
scripts/peek_data.py và scripts/check_phase0_gates.py lo.
"""

import csv
from pathlib import Path

import pytest

from aic2026 import paths
from aic2026.frame_map import SOURCE_COLUMNS, FrameMap, KeyframeRow
from aic2026.submit import KIS, QA, TRAKE, Answer, SubmissionBudget, submission_filename


# ---------------------------------------------------------------------------
# Dữ liệu giả: một bảng map-keyframes ba dòng, 25 hình/giây
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_csv(tmp_path: Path) -> Path:
    path = tmp_path / "L21_V001.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        # SOURCE_COLUMNS = bốn cột của CSV BTC (không có video_id — cột đó
        # lấy từ TÊN TỆP và đã truyền vào qua FrameMap.load(video_id, ...)).
        # EXPECTED_COLUMNS = năm cột của frame_map.parquet.
        # Bản cũ ghi EXPECTED_COLUMNS, tức là dùng hằng số của ĐẦU RA để
        # dựng tệp ĐẦU VÀO: header năm tên, dòng dữ liệu bốn ô, ô cuối
        # thành None.
        w.writerow(SOURCE_COLUMNS)
        w.writerow([1, 0.0, 25.0, 0])
        w.writerow([2, 4.0, 25.0, 100])
        w.writerow([3, 12.52, 25.0, 313])
    return path


# --- MỤC 1 -----------------------------------------------------------------
def test_1_khong_ai_viet_duong_dan_rieng():
    """paths.py là nơi duy nhất biết đường dẫn; các module khác không hardcode."""
    assert paths.DATA_ROOT.is_absolute()
    assert paths.RAW_DIR.parent == paths.DATA_ROOT
    assert paths.DERIVED_DIR.parent == paths.DATA_ROOT
    assert paths.INDEX_DIR.parent == paths.DATA_ROOT
    # Gốc dữ liệu KHÔNG được nằm trong gốc mã nguồn
    assert paths.PROJECT_ROOT.resolve() not in paths.DATA_ROOT.resolve().parents


# --- MỤC 2 -----------------------------------------------------------------
def test_2_du_hai_muoi_thu_muc_chuan():
    """Danh sách thư mục chuẩn đủ và không trùng nhau."""
    assert len(paths.REQUIRED_DIRS) == 21
    assert len(set(paths.REQUIRED_DIRS)) == len(paths.REQUIRED_DIRS)
    # frame_map.parquet là TỆP, tuyệt đối không nằm trong danh sách thư mục
    assert paths.FRAME_MAP_PARQUET not in paths.REQUIRED_DIRS
    assert paths.FRAME_MAP_PARQUET.suffix == ".parquet"


# --- MỤC 3 -----------------------------------------------------------------
def test_3_ten_tep_giu_dung_bon_chu_so():
    """Quy tắc đặt tên số 2: không được cắt số 0 ở đầu."""
    # BTC đệm BA chữ số: raw/objects/L21_V001/047.json. Bản cũ khẳng định
    # "0047.json" và chỉ pass trên máy CHƯA tải objects — lúc đó hàm rơi về
    # đệm bốn chữ số. Máy có dữ liệu thật thì test đỏ.
    #
    # Quy tắc "không cắt số 0 ở đầu" áp cho tệp NHÓM TỰ TẠO, không áp cho tệp
    # BTC phát. Với tệp BTC, việc phải làm là tra ra ĐÚNG tệp có thật.
    ten = paths.objects_file("L21_V001", 47).name
    assert ten.endswith(".json")
    assert int(ten.removesuffix(".json").lstrip("0") or "0") == 47
    assert paths.keyframe_image("L21_V001", 0).name == "0000.jpg"
    assert paths.thumbnail_image("L21_V001", 1234).name == "1234.jpg"
    assert paths.map_keyframes_file("L21_V001").name == "L21_V001.csv"
    assert paths.clip_features_file("L21_V001").name == "L21_V001.npy"


# --- MỤC 4 -----------------------------------------------------------------
def test_4_doc_duoc_bang_doi_chieu(fake_csv):
    """Đọc map-keyframes ra đúng bốn cột, đúng kiểu số."""
    fm = FrameMap.load("L21_V001", fake_csv)
    assert len(fm) == 3
    r = fm.row_of(2)
    assert isinstance(r, KeyframeRow)
    assert r.pts_time == 4.0 and r.fps == 25.0 and r.frame_idx == 100
    assert r.image_name == "0002.jpg"


# --- MỤC 5 -----------------------------------------------------------------
def test_5_n_khac_frame_idx(fake_csv):
    """MỤC QUAN TRỌNG NHẤT: số thứ tự ảnh KHÁC vị trí khung hình."""
    fm = FrameMap.load("L21_V001", fake_csv)
    assert fm.frame_idx_of(2) == 100      # ảnh thứ 2 nằm ở khung hình 100
    assert fm.frame_idx_of(3) == 313
    assert fm.frame_idx_of(2) != 2        # nộp nhầm số 2 là 0 điểm
    # Phép tính kiểm chứng: vị trí ≈ giây × số hình mỗi giây
    assert fm.max_drift() <= 1
    with pytest.raises(KeyError):
        fm.frame_idx_of(999)


# --- MỤC 6 -----------------------------------------------------------------
def test_6_tep_nop_dung_dinh_dang():
    """Ba dạng truy vấn ra đúng số cột."""
    assert Answer("L21_V001", [1500]).to_row(KIS) == ["L21_V001", "1500"]
    assert Answer("L21_V001", [3450], "màu xanh").to_row(QA) == ["L21_V001", "3450", "màu xanh"]
    assert Answer("L21_V001", [10, 20, 30, 40]).to_row(TRAKE) == ["L21_V001", "10", "20", "30", "40"]
    assert submission_filename("7", QA) == "query-7-qa.csv"
    with pytest.raises(ValueError):
        Answer("L21_V001", [100]).to_row(QA)      # Q&A thiếu answer


# --- MỤC 7 -----------------------------------------------------------------
def test_7_bo_trung_va_cat_o_100_dong():
    """Trùng nhau chiếm suất mà không thêm điểm; quá 100 dòng bị BTC bỏ."""
    b = SubmissionBudget(task=KIS)
    assert b.add(Answer("L21_V001", [500])) is True
    assert b.add(Answer("L21_V001", [500])) is False   # trùng y hệt
    assert b.dropped_duplicate == 1
    assert len(b) == 1

    b2 = SubmissionBudget(task=KIS)
    for i in range(150):
        b2.add(Answer("L21_V001", [i]))
    assert len(b2) == 100
    assert b2.is_full()
    assert b2.dropped_overflow == 50


# --- MỤC 8 -----------------------------------------------------------------
def test_8_ghi_tep_nop_khong_dong_tieu_de(tmp_path):
    """Tệp nộp: không dòng tiêu đề, giữ nguyên thứ tự, tiếng Việt có dấu."""
    b = SubmissionBudget(task=QA)
    b.add(Answer("L21_V001", [500], "màu xanh"))
    b.add(Answer("L22_V003", [880], "năm người"))
    out = b.write(tmp_path / "query-demo-qa.csv")

    with out.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["L21_V001", "500", "màu xanh"]   # dòng đầu là dữ liệu
    assert rows[1][2] == "năm người"                    # dấu tiếng Việt còn nguyên
    assert len(rows) == 2
    assert not list(tmp_path.glob("*.partial"))         # tệp tạm đã đổi tên