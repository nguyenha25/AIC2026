"""
Bảng đối chiếu: TẤM ẢNH THỨ MẤY  <->  VỊ TRÍ KHUNG HÌNH TRONG VIDEO.

ĐỌC KỸ PHẦN NÀY — đây là chỗ dễ sai nhất và sai thì không ai biết.

Tệp raw/map-keyframes/L21_V001.csv có bốn cột:

    n          số thứ tự tấm ảnh. Ảnh 0001.jpg thì n = 1.
    pts_time   tấm ảnh đó nằm ở giây thứ mấy của video.
    fps        video quay bao nhiêu hình một giây.
    frame_idx  VỊ TRÍ KHUNG HÌNH TRONG VIDEO. ĐÂY LÀ SỐ PHẢI NỘP.

n và frame_idx là HAI SỐ HOÀN TOÀN KHÁC NHAU. Ảnh thứ 5 của một video
25 hình/giây có thể nằm ở khung hình thứ 812. Nộp nhầm số 5 thì BTC chấm
0 điểm, mà kết quả tìm kiếm trên màn hình vẫn trông rất đúng.

Quan hệ kiểm chứng được:   frame_idx ≈ pts_time × fps   (lệch 0 hoặc 1
do làm tròn). Đây là cách duy nhất kiểm được bảng này khi máy chưa có video.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .paths import MAP_KEYFRAMES_DIR, list_video_ids, map_keyframes_file

EXPECTED_COLUMNS = ["n", "pts_time", "fps", "frame_idx"]


@dataclass(frozen=True)
class KeyframeRow:
    """Một dòng của map-keyframes."""

    video_id: str
    n: int
    pts_time: float
    fps: float
    frame_idx: int

    @property
    def expected_frame_idx(self) -> int:
        """Vị trí tính lại từ giây × số hình mỗi giây."""
        return round(self.pts_time * self.fps)

    @property
    def drift(self) -> int:
        """Chênh lệch giữa số ghi trong tệp và số tự tính. Phải là 0 hoặc 1."""
        return abs(self.frame_idx - self.expected_frame_idx)

    @property
    def image_name(self) -> str:
        """Tên tệp ảnh tương ứng, giữ đủ bốn chữ số: 0047.jpg"""
        return f"{self.n:04d}.jpg"


class FrameMap:
    """Bảng đối chiếu của MỘT video."""

    def __init__(self, video_id: str, rows: list[KeyframeRow]):
        self.video_id = video_id
        self.rows = rows
        self._by_n = {r.n: r for r in rows}

    # -- đọc --------------------------------------------------------------
    @classmethod
    def load(cls, video_id: str, path: Path | None = None) -> "FrameMap":
        """Đọc raw/map-keyframes/<video_id>.csv"""
        path = path or map_keyframes_file(video_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Không thấy bảng đối chiếu của {video_id} tại {path}. "
                "Đã giải nén map-keyframes-aic25-b1.zip vào raw/map-keyframes/ chưa?"
            )

        rows: list[KeyframeRow] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in EXPECTED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(
                    f"{path.name} thiếu cột {missing}. "
                    f"Cột đọc được: {reader.fieldnames}"
                )
            for raw in reader:
                rows.append(
                    KeyframeRow(
                        video_id=video_id,
                        n=int(raw["n"]),
                        pts_time=float(raw["pts_time"]),
                        fps=float(raw["fps"]),
                        frame_idx=int(raw["frame_idx"]),
                    )
                )
        rows.sort(key=lambda r: r.n)
        return cls(video_id, rows)

    # -- tra cứu ----------------------------------------------------------
    def frame_idx_of(self, n: int) -> int:
        """Số thứ tự tấm ảnh -> vị trí khung hình để NỘP BÀI."""
        if n not in self._by_n:
            raise KeyError(f"{self.video_id} không có tấm ảnh thứ {n}")
        return self._by_n[n].frame_idx

    def row_of(self, n: int) -> KeyframeRow:
        return self._by_n[n]

    def nearest_by_time(self, seconds: float) -> KeyframeRow:
        """Tấm ảnh gần giây thứ `seconds` nhất."""
        if not self.rows:
            raise ValueError(f"{self.video_id} không có dòng nào")
        return min(self.rows, key=lambda r: abs(r.pts_time - seconds))

    # -- kiểm tra ---------------------------------------------------------
    def max_drift(self) -> int:
        """Chênh lệch lớn nhất trên toàn bộ video. Vượt quá 1 là có vấn đề."""
        return max((r.drift for r in self.rows), default=0)

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return f"<FrameMap {self.video_id}: {len(self.rows)} ảnh>"


def available_video_ids() -> list[str]:
    """Danh sách video có bảng đối chiếu trên máy này."""
    return list_video_ids(MAP_KEYFRAMES_DIR, ".csv")
