"""
Xuất tệp nộp bài.

Ba dạng truy vấn, ba định dạng dòng (theo tài liệu vòng sơ tuyển):

    KIS    : <video_id>,<frame_id>
    Q&A    : <video_id>,<frame_id>,<answer>
    TRAKE  : <video_id>,<frame_id_1>,<frame_id_2>,...,<frame_id_n>

Ba luật bắt buộc:

1. TỐI ĐA 100 DÒNG một truy vấn. Dòng thứ 101 trở đi bị bỏ.
2. KHÔNG TRÙNG. Điểm cuối lấy `max` ở từng mốc thứ hạng, nên hai dòng trỏ
   vào cùng một chỗ chỉ ăn điểm một lần mà lại chiếm mất một suất.
3. TỆP KHÔNG CÓ DÒNG TIÊU ĐỀ. Thứ tự dòng chính là thứ hạng.

LƯU Ý: quy ước ĐẶT TÊN TỆP dưới đây là do nhóm tự chọn, BTC chưa công bố
chính thức. Khi có thông báo thì sửa hàm submission_filename(), chỗ khác
không phải đụng tới.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

MAX_ANSWERS_PER_QUERY = 100

KIS = "kis"
QA = "qa"
TRAKE = "trake"


@dataclass
class Answer:
    """Một câu trả lời = một dòng trong tệp nộp."""

    video_id: str
    frame_ids: list[int]          # KIS/QA: một phần tử. TRAKE: n phần tử.
    answer: str | None = None     # chỉ Q&A mới có

    def __post_init__(self):
        if not self.video_id:
            raise ValueError("video_id rỗng")
        if not self.frame_ids:
            raise ValueError("Phải có ít nhất một frame_id")
        self.frame_ids = [int(x) for x in self.frame_ids]
        if any(x < 0 for x in self.frame_ids):
            raise ValueError(f"frame_id âm: {self.frame_ids}")

    def dedup_key(self) -> tuple:
        """Hai câu trả lời cùng khóa này là TRÙNG NHAU."""
        ans = self.answer.strip().lower() if self.answer else None
        return (self.video_id, tuple(self.frame_ids), ans)

    def to_row(self, task: str) -> list[str]:
        if task == KIS:
            return [self.video_id, str(self.frame_ids[0])]
        if task == QA:
            # VIỆC 14 — LUẬT CỨNG. Vòng p1: MỘT ô đáp án để trống làm BTC loại
            # NGUYÊN TỆP 100 dòng. Không phải chấm 0 điểm câu đó — mất trắng cả
            # câu. Điền bừa vẫn giữ tệp hợp lệ và không mất gì thêm.
            #
            # Chặn ở đây chứ không chỉ ở script sinh tệp: đây là chỗ DUY NHẤT
            # mọi đường ghi tệp nộp đều đi qua, nên không đường nào lách được.
            if self.answer is None or not str(self.answer).strip():
                raise ValueError(
                    f"Ô đáp án Q&A rỗng ({self.video_id}, "
                    f"frame {self.frame_ids[0]}). "
                    "Vòng p1 một ô trống đã làm BTC loại nguyên tệp 100 dòng. "
                    "Chưa biết đáp án thì ĐIỀN BỪA — xem "
                    "aic2026.qa_answer.doan_du_phong()."
                )
            return [self.video_id, str(self.frame_ids[0]), str(self.answer).strip()]
        if task == TRAKE:
            return [self.video_id, *[str(x) for x in self.frame_ids]]
        raise ValueError(f"Dạng truy vấn lạ: {task}")


@dataclass
class SubmissionBudget:
    """
    Gom câu trả lời cho MỘT truy vấn: bỏ trùng, cắt ở 100 dòng,
    giữ nguyên thứ tự thêm vào (thứ tự = thứ hạng).
    """

    task: str
    limit: int = MAX_ANSWERS_PER_QUERY
    _answers: list[Answer] = field(default_factory=list)
    _seen: set = field(default_factory=set)
    dropped_duplicate: int = 0
    dropped_overflow: int = 0

    def add(self, answer: Answer) -> bool:
        """Thêm một câu trả lời. Trả về True nếu được nhận."""
        key = answer.dedup_key()
        if key in self._seen:
            self.dropped_duplicate += 1
            return False
        if len(self._answers) >= self.limit:
            self.dropped_overflow += 1
            return False
        self._seen.add(key)
        self._answers.append(answer)
        return True

    def extend(self, answers) -> int:
        return sum(1 for a in answers if self.add(a))

    @property
    def answers(self) -> list[Answer]:
        return list(self._answers)

    def __len__(self) -> int:
        return len(self._answers)

    def is_full(self) -> bool:
        return len(self._answers) >= self.limit

    def o_dap_an_rong(self) -> list[int]:
        """VIỆC 14: các dòng có ô đáp án rỗng. Rỗng nghĩa là ĐẠT.

        Chỉ có ý nghĩa với task="qa". Trả về số thứ tự dòng (đếm từ 1).
        """
        if self.task != QA:
            return []
        return [
            i
            for i, a in enumerate(self._answers, start=1)
            if a.answer is None or not str(a.answer).strip()
        ]

    def write(self, path: Path) -> Path:
        """Ghi ra tệp CSV, không dòng tiêu đề, xuống dòng kiểu \\n."""
        # VIỆC 14: soát TRƯỚC khi mở tệp. Ghi được nửa tệp rồi mới dừng thì
        # còn lại một tệp .partial mà người chạy dễ tưởng là bản tốt.
        rong = self.o_dap_an_rong()
        if rong:
            raise ValueError(
                f"{len(rong)} dòng có ô đáp án Q&A rỗng (dòng {rong[:10]}). "
                "KHÔNG ghi tệp. Một ô trống làm BTC loại nguyên tệp 100 dòng."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".partial")   # quy tắc đặt tên số 7
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            for a in self._answers:
                writer.writerow(a.to_row(self.task))
        tmp.replace(path)
        return path

    def report(self) -> str:
        return (
            f"{self.task}: {len(self._answers)}/{self.limit} dòng | "
            f"bỏ vì trùng: {self.dropped_duplicate} | "
            f"bỏ vì quá 100: {self.dropped_overflow}"
        )


def submission_filename(query_id: str, task: str) -> str:
    """query-1-kis.csv / query-7-qa.csv / query-12-trake.csv"""
    if task not in (KIS, QA, TRAKE):
        raise ValueError(f"Dạng truy vấn lạ: {task}")
    return f"query-{query_id}-{task}.csv"
