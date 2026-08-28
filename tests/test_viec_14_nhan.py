"""
Việc 14 (ô đáp án Q&A không bao giờ rỗng) — Nhàn.

Chỉ import mã của Nhàn — chạy được ngay cả khi các gói khác CHƯA trộn.
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

import pytest

from aic2026.submit.formatter import QA, Answer, SubmissionBudget  # noqa: E402


@pytest.mark.parametrize("dap_an", [None, "", "   ", "\t\n"])
def test_o_dap_an_qa_rong_bi_chan(dap_an):
    a = Answer(video_id="L21_V001", frame_ids=[500], answer=dap_an)
    with pytest.raises(ValueError, match="rỗng"):
        a.to_row(QA)


def test_khong_ghi_tep_khi_con_o_rong(tmp_path):
    ngan_sach = SubmissionBudget(task=QA)
    ngan_sach.add(Answer("L21_V001", [500], "5"))
    ngan_sach.add(Answer("L21_V001", [600], ""))

    assert ngan_sach.o_dap_an_rong() == [2]

    dich = tmp_path / "query-1-qa.csv"
    with pytest.raises(ValueError, match="rỗng"):
        ngan_sach.write(dich)
    assert not dich.exists()      # không để lại tệp cụt


def test_dap_an_hop_le_van_ghi_binh_thuong(tmp_path):
    ngan_sach = SubmissionBudget(task=QA)
    ngan_sach.add(Answer("L21_V001", [500], " 5 "))
    dich = ngan_sach.write(tmp_path / "query-1-qa.csv")
    assert dich.read_text(encoding="utf-8").strip() == "L21_V001,500,5"


def test_kis_khong_bi_luat_qa_anh_huong(tmp_path):
    from aic2026.submit.formatter import KIS

    ngan_sach = SubmissionBudget(task=KIS)
    ngan_sach.add(Answer("L21_V001", [500]))
    assert ngan_sach.o_dap_an_rong() == []
    ngan_sach.write(tmp_path / "query-1-kis.csv")


