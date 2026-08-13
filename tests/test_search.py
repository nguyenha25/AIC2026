"""
Bộ test Việc 4 — mạch tìm kiếm đầu–cuối.

Không cần faiss, không cần open-clip-torch, không cần dữ liệu thật. Nguồn ứng
viên được truyền vào qua tham số `tim_ung_vien` của run_query(), nên bộ test
này kiểm ĐÚNG phần Ngân chịu trách nhiệm: lọc trùng, cắt 100, ghi tệp — chứ
không kiểm chất lượng của kho FAISS (phần đó là Việc 2 của Thi).

Chạy:  pytest tests/test_search.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import pytest

from src.aic2026.rank.search import run_query
from src.aic2026.submit import KIS, QA, TRAKE


FPS = 25.0


@dataclass(frozen=True)
class HitGia:
    """Bản sao tối giản của index.faiss_index.Hit — đủ sáu thuộc tính."""

    video_id: str
    n: int
    score: float
    frame_idx: int
    pts_time: float
    source: str = "clip"


def tao_hit(video_id: str, n: int, pts_time: float, score: float) -> HitGia:
    """
    n và frame_idx CỐ Ý khác nhau rất xa.

    Đây là điểm mấu chốt của cả bộ test: nếu mạch tìm kiếm lỡ ghi `n` vào tệp
    nộp thay vì `frame_idx`, bài nộp vẫn hợp lệ về định dạng và không có triệu
    chứng nào — chỉ là không được điểm. Cho hai số lệch hẳn nhau thì phép so
    trong test bắt được ngay.
    """
    return HitGia(
        video_id=video_id,
        n=n,
        score=score,
        frame_idx=int(round(pts_time * FPS)),
        pts_time=pts_time,
        source="clip",
    )


def nguon_gia(so_video: int, so_anh: int, buoc_giay: float):
    """
    Sinh một hàm tìm ứng viên giả, xếp hạng sẵn theo score giảm dần.

    Mỗi video có `so_anh` ảnh, cách nhau `buoc_giay` giây.
    """
    hits: list[HitGia] = []
    diem = 1.0

    for k in range(so_anh):
        for v in range(so_video):
            diem -= 0.0001
            hits.append(
                tao_hit(
                    video_id=f"L21_V{v:03d}",
                    n=k + 1,
                    pts_time=k * buoc_giay,
                    score=diem,
                )
            )

    def tim(cau_hoi: str, so_ung_vien: int):
        return hits[:so_ung_vien]

    return tim


# ---------------------------------------------------------------------------


def test_mach_chay_du_nam_buoc(tmp_path):
    """Mạch đầu–cuối chạy hết và ghi ra tệp nộp"""
    kq = run_query(
        cau_hoi="người dẫn chương trình trong trường quay",
        query_id="test-1",
        task=KIS,
        thu_muc_nop=tmp_path,
        tim_ung_vien=nguon_gia(so_video=60, so_anh=10, buoc_giay=2.0),
    )

    assert kq.duong_dan_nop is not None
    assert kq.duong_dan_nop.exists()
    assert kq.duong_dan_nop.name == "query-test-1-kis.csv"
    assert kq.so_ung_vien == 500          # lấy dư theo settings, không lấy 100
    assert kq.so_dong_nop == 100


def test_tep_nop_ghi_frame_idx_khong_phai_n(tmp_path):
    """Tệp nộp ghi frame_idx, không phải n"""
    tim = nguon_gia(so_video=60, so_anh=10, buoc_giay=2.0)

    kq = run_query(
        cau_hoi="bất kỳ",
        query_id="test-2",
        task=KIS,
        thu_muc_nop=tmp_path,
        tim_ung_vien=tim,
    )

    with kq.duong_dan_nop.open(encoding="utf-8") as f:
        dong = list(csv.reader(f))

    frame_idx_hop_le = {h.frame_idx for h in tim("", 500)}
    n_hop_le = {h.n for h in tim("", 500)}

    for video_id, so in dong:
        gia_tri = int(so)
        assert gia_tri in frame_idx_hop_le
        # n chạy 1..10, frame_idx chạy 0..450 — chồng nhau ở vài giá trị nhỏ,
        # nên kiểm bằng cách khác: số ghi ra phải khớp frame_idx của ĐÚNG
        # dòng đó, tra ngược qua danh sách kết quả.
    for hit, (video_id, so) in zip(kq.ket_qua, dong):
        assert video_id == hit.video_id
        assert int(so) == hit.frame_idx
        if hit.n != hit.frame_idx:
            assert int(so) != hit.n

    assert n_hop_le  # danh sách n không rỗng, phép so trên có ý nghĩa


def test_tep_nop_khong_co_dong_tieu_de(tmp_path):
    """Tệp nộp không có dòng tiêu đề"""
    kq = run_query(
        cau_hoi="bất kỳ",
        query_id="test-3",
        task=KIS,
        thu_muc_nop=tmp_path,
        tim_ung_vien=nguon_gia(60, 10, 2.0),
    )

    dong_dau = kq.duong_dan_nop.read_text(encoding="utf-8").splitlines()[0]
    video_id, so = dong_dau.split(",")

    assert video_id.startswith("L")
    assert so.isdigit()


def test_cong_thoat_so_5(tmp_path):
    """Cổng thoát 5: không có hai dòng cùng video cách nhau dưới 10 giây"""
    kq = run_query(
        cau_hoi="bất kỳ",
        query_id="test-4",
        task=KIS,
        thu_muc_nop=tmp_path,
        # Ảnh cách nhau 2 giây: nguồn giả CỐ Ý vi phạm luật lọc trùng.
        tim_ung_vien=nguon_gia(60, 10, 2.0),
    )

    assert kq.dat_cong_5, kq.vi_pham_cong_5
    assert kq.bao_cao_loc_trung.bo_vi_gan > 0    # có bỏ thật, không phải may


def test_cat_dung_100_dong(tmp_path):
    """Cắt đúng 100 dòng dù nguồn cho nhiều hơn"""
    kq = run_query(
        cau_hoi="bất kỳ",
        query_id="test-5",
        task=KIS,
        thu_muc_nop=tmp_path,
        # Cách nhau 30 giây: không ảnh nào bị lọc, 500 ứng viên đều hợp lệ.
        tim_ung_vien=nguon_gia(60, 10, 30.0),
    )

    so_dong = len(kq.duong_dan_nop.read_text(encoding="utf-8").splitlines())

    assert so_dong == 100
    assert kq.so_dong_nop == 100


def test_qa_thieu_dap_an_thi_dung_ngay():
    """Dạng Q&A thiếu đáp án thì dừng ngay, không ghi tệp"""
    with pytest.raises(ValueError, match="đáp án"):
        run_query(
            cau_hoi="bất kỳ",
            query_id="test-6",
            task=QA,
            tim_ung_vien=nguon_gia(10, 5, 30.0),
        )


def test_qa_ghi_du_ba_cot(tmp_path):
    """Dạng Q&A ghi đủ ba cột: video, frame, đáp án"""
    kq = run_query(
        cau_hoi="có bao nhiêu người lên sân khấu",
        query_id="test-7",
        task=QA,
        dap_an="5",
        thu_muc_nop=tmp_path,
        tim_ung_vien=nguon_gia(60, 10, 30.0),
    )

    with kq.duong_dan_nop.open(encoding="utf-8") as f:
        dong = list(csv.reader(f))

    assert all(len(d) == 3 for d in dong)
    assert all(d[2] == "5" for d in dong)


def test_trake_moi_dong_du_moc_va_tang_dan_theo_thoi_gian(tmp_path):
    """TRAKE: mỗi dòng đủ số mốc và xếp tăng dần theo thời gian"""
    kq = run_query(
        cau_hoi="vận động viên nhảy cao",
        query_id="test-8",
        task=TRAKE,
        so_moc_trake=4,
        thu_muc_nop=tmp_path,
        tim_ung_vien=nguon_gia(60, 6, 12.0),
    )

    with kq.duong_dan_nop.open(encoding="utf-8") as f:
        dong = list(csv.reader(f))

    assert dong, "TRAKE không ra dòng nào"

    for d in dong:
        moc = [int(x) for x in d[1:]]
        assert len(moc) == 4
        assert moc == sorted(moc)


def test_khong_ghi_thi_khong_dung_dia(tmp_path):
    """ghi_tep=False thì chạy hết mạch nhưng không tạo tệp"""
    kq = run_query(
        cau_hoi="bất kỳ",
        query_id="test-9",
        task=KIS,
        ghi_tep=False,
        thu_muc_nop=tmp_path,
        tim_ung_vien=nguon_gia(60, 10, 30.0),
    )

    assert kq.duong_dan_nop is None
    assert kq.so_dong_nop == 100
    assert list(tmp_path.iterdir()) == []


def test_canh_bao_khi_khong_du_100_dong(tmp_path):
    """Không đủ 100 dòng thì phải có cảnh báo, không im lặng"""
    kq = run_query(
        cau_hoi="bất kỳ",
        query_id="test-10",
        task=KIS,
        thu_muc_nop=tmp_path,
        # 5 video × 4 ảnh, ảnh cách nhau 2 giây → lọc trùng còn 5 dòng.
        tim_ung_vien=nguon_gia(5, 4, 2.0),
    )

    assert kq.so_dong_nop < 100
    assert any("dòng" in c for c in kq.canh_bao)