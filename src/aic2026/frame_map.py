"""
Bảng đối chiếu: TẤM ẢNH THỨ MẤY <-> VỊ TRÍ KHUNG HÌNH TRONG VIDEO.

Tệp raw/map-keyframes/L21_V001.csv có bốn cột:

    n          số thứ tự tấm ảnh. Ảnh 0001.jpg thì n = 1.
    pts_time   tấm ảnh đó nằm ở giây thứ mấy của video.
    fps        video quay bao nhiêu hình một giây.
    frame_idx  VỊ TRÍ KHUNG HÌNH TRONG VIDEO. ĐÂY LÀ SỐ PHẢI NỘP.

n và frame_idx là HAI SỐ HOÀN TOÀN KHÁC NHAU.

Quan hệ kiểm chứng:
    frame_idx ≈ pts_time × fps

Cho phép lệch 0 hoặc 1 do làm tròn.

Việc 1:
    python scripts/build_frame_map.py

Sau khi dựng:
    - index/frame_map.parquet là nguồn dữ liệu dùng chung.
    - load_frame_map() đọc Parquet và cache trong RAM.
    - lookup() tra cứu bằng dict trong RAM.
    - Không đọc lại toàn bộ CSV trong mỗi truy vấn.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import (
    FRAME_MAP_PARQUET,
    MAP_KEYFRAMES_DIR,
    list_video_ids,
    map_keyframes_file,
)


EXPECTED_COLUMNS = [
    "video_id",
    "n",
    "pts_time",
    "fps",
    "frame_idx",
]

SOURCE_COLUMNS = [
    "n",
    "pts_time",
    "fps",
    "frame_idx",
]

EXPECTED_ROWS = 177_321
EXPECTED_VIDEOS = 873
MAX_ALLOWED_DRIFT = 1

FRAME_MAP_DTYPES = {
    "video_id": "string",
    "n": "int32",
    "pts_time": "float64",
    "fps": "float64",
    "frame_idx": "int64",
}


# ---------------------------------------------------------------------------
# Cache dùng trong runtime
# ---------------------------------------------------------------------------

_CACHED_FRAME_MAP: pd.DataFrame | None = None

_CACHED_LOOKUP: dict[
    tuple[str, int],
    tuple[int, float],
] | None = None


# ---------------------------------------------------------------------------
# Một dòng map-keyframes
# ---------------------------------------------------------------------------

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
        """Vị trí tính lại từ pts_time × fps."""
        return round(self.pts_time * self.fps)

    @property
    def drift(self) -> int:
        """Độ lệch giữa frame_idx ghi trong CSV và giá trị tự tính."""
        return abs(
            self.frame_idx - self.expected_frame_idx
        )

    @property
    def image_name(self) -> str:
        """Tên ảnh tương ứng, ví dụ 0047.jpg."""
        return f"{self.n:04d}.jpg"


# ---------------------------------------------------------------------------
# Bảng của một video
# ---------------------------------------------------------------------------

class FrameMap:
    """Bảng đối chiếu của MỘT video."""

    def __init__(
        self,
        video_id: str,
        rows: list[KeyframeRow],
    ):
        self.video_id = video_id
        self.rows = rows
        self._by_n = {
            row.n: row
            for row in rows
        }

    @classmethod
    def load(
        cls,
        video_id: str,
        path: Path | None = None,
    ) -> "FrameMap":
        """Đọc raw/map-keyframes/<video_id>.csv."""

        path = path or map_keyframes_file(video_id)

        if not path.exists():
            raise FileNotFoundError(
                f"Không thấy bảng đối chiếu của {video_id} tại {path}. "
                "Đã giải nén map-keyframes-aic25-b1.zip vào "
                "raw/map-keyframes/ chưa?"
            )

        rows: list[KeyframeRow] = []

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            reader = csv.DictReader(f)

            missing = [
                column
                for column in SOURCE_COLUMNS
                if column not in (reader.fieldnames or [])
            ]

            if missing:
                raise ValueError(
                    f"{path.name} thiếu cột {missing}. "
                    f"Cột đọc được: {reader.fieldnames}"
                )

            for line_no, raw in enumerate(reader, start=2):
                blank = [
                    column
                    for column in SOURCE_COLUMNS
                    if not str(raw.get(column, "")).strip()
                ]

                if blank:
                    raise ValueError(
                        f"{path.name} dòng {line_no} "
                        f"bỏ trống ô {blank}."
                    )

                try:
                    n = int(float(raw["n"]))
                    pts_time = float(raw["pts_time"])
                    fps = float(raw["fps"])
                    frame_idx = int(float(raw["frame_idx"]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path.name} dòng {line_no}: "
                        f"dữ liệu không hợp lệ: {raw}"
                    ) from exc

                rows.append(
                    KeyframeRow(
                        video_id=video_id,
                        n=n,
                        pts_time=pts_time,
                        fps=fps,
                        frame_idx=frame_idx,
                    )
                )

        rows.sort(key=lambda row: row.n)

        return cls(video_id, rows)

    # ------------------------------------------------------------------
    # Tra cứu trong một video
    # ------------------------------------------------------------------

    def frame_idx_of(self, n: int) -> int:
        """n -> frame_idx để dùng khi nộp bài."""

        if n not in self._by_n:
            raise KeyError(
                f"{self.video_id} không có tấm ảnh thứ {n}"
            )

        return self._by_n[n].frame_idx

    def row_of(self, n: int) -> KeyframeRow:
        """Lấy toàn bộ thông tin của keyframe n."""
        if n not in self._by_n:
            raise KeyError(
                f"{self.video_id} không có tấm ảnh thứ {n}"
            )

        return self._by_n[n]

    def nearest_by_time(
        self,
        seconds: float,
    ) -> KeyframeRow:
        """Tìm tấm ảnh gần thời điểm seconds nhất."""

        if not self.rows:
            raise ValueError(
                f"{self.video_id} không có dòng nào"
            )

        return min(
            self.rows,
            key=lambda row: abs(
                row.pts_time - seconds
            ),
        )

    # ------------------------------------------------------------------
    # Kiểm tra
    # ------------------------------------------------------------------

    def max_drift(self) -> int:
        """Độ lệch lớn nhất của toàn bộ video."""

        return max(
            (row.drift for row in self.rows),
            default=0,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return (
            f"<FrameMap {self.video_id}: "
            f"{len(self.rows)} ảnh>"
        )


# ---------------------------------------------------------------------------
# Danh sách video
# ---------------------------------------------------------------------------

def available_video_ids() -> list[str]:
    """Danh sách video có bảng map-keyframes trên máy."""

    return list_video_ids(
        MAP_KEYFRAMES_DIR,
        ".csv",
    )


# ---------------------------------------------------------------------------
# BUILD: chỉ dùng khi dựng lại kho
# ---------------------------------------------------------------------------

def scan_map_keyframes() -> tuple[
    pd.DataFrame,
    int,
    str,
]:
    """
    Đọc toàn bộ CSV gốc và gộp thành một DataFrame.

    Hàm này chỉ dùng khi BUILD/VERIFY.
    Runtime không gọi lại hàm này.

    Trả về:
        (frame_map, worst_drift, worst_drift_at)
    """

    video_ids = available_video_ids()

    if not video_ids:
        raise FileNotFoundError(
            "Không tìm thấy tệp map-keyframes nào trong "
            f"{MAP_KEYFRAMES_DIR}."
        )

    total = len(video_ids)

    print(
        f"Đọc {total} bảng đối chiếu "
        f"từ {MAP_KEYFRAMES_DIR}"
    )

    all_rows: list[dict] = []

    worst_drift = 0
    worst_drift_at = ""

    for index, video_id in enumerate(
        video_ids,
        start=1,
    ):
        frame_map = FrameMap.load(video_id)

        drift = frame_map.max_drift()

        if drift > worst_drift:
            worst_drift = drift
            worst_drift_at = video_id

        for row in frame_map.rows:
            all_rows.append(
                {
                    "video_id": row.video_id,
                    "n": row.n,
                    "pts_time": row.pts_time,
                    "fps": row.fps,
                    "frame_idx": row.frame_idx,
                }
            )

        if index % 50 == 0 or index == total:
            print(
                f"  {index}/{total} video — "
                f"{len(all_rows):,} dòng"
            )

    frame_map_df = pd.DataFrame(
        all_rows,
        columns=EXPECTED_COLUMNS,
    )

    frame_map_df = frame_map_df.astype(
        FRAME_MAP_DTYPES
    )

    frame_map_df = (
        frame_map_df
        .sort_values(
            ["video_id", "n"]
        )
        .reset_index(drop=True)
    )

    return (
        frame_map_df,
        worst_drift,
        worst_drift_at,
    )


# ---------------------------------------------------------------------------
# KIỂM TRA
# ---------------------------------------------------------------------------

def check_frame_map(
    frame_map_df: pd.DataFrame,
    worst_drift: int = 0,
    worst_drift_at: str = "",
) -> list[str]:
    """
    Kiểm tra toàn bộ điều kiện của Việc 1.

    Trả về danh sách lỗi.
    Danh sách rỗng nghĩa là ĐẠT.
    """

    problems: list[str] = []

    # Số dòng
    rows = len(frame_map_df)

    if rows != EXPECTED_ROWS:
        problems.append(
            f"đếm được {rows:,} dòng, "
            f"cần {EXPECTED_ROWS:,}"
        )

    # Số video
    video_count = frame_map_df[
        "video_id"
    ].nunique()

    if video_count != EXPECTED_VIDEOS:
        problems.append(
            f"đếm được {video_count:,} video, "
            f"cần {EXPECTED_VIDEOS:,}"
        )

    # Tên cột
    if list(frame_map_df.columns) != EXPECTED_COLUMNS:
        problems.append(
            "sai cấu trúc cột"
        )

    # Ô trống
    empty_cells = int(
        frame_map_df.isna().sum().sum()
    )

    if empty_cells:
        problems.append(
            f"có {empty_cells:,} ô trống"
        )

    # Duplicate
    duplicated = frame_map_df.duplicated(
        subset=["video_id", "n"]
    )

    if duplicated.any():
        problems.append(
            "có cặp (video_id, n) trùng nhau"
        )

    # Dtype
    for column, expected_dtype in FRAME_MAP_DTYPES.items():
        actual_dtype = str(
            frame_map_df[column].dtype
        )

        if actual_dtype != expected_dtype:
            problems.append(
                f"cột {column} có dtype "
                f"{actual_dtype}, cần {expected_dtype}"
            )

    # Drift
    if len(frame_map_df):
        expected_frame_idx = (
            frame_map_df["pts_time"]
            * frame_map_df["fps"]
        ).round()

        drift = (
            frame_map_df["frame_idx"]
            - expected_frame_idx
        ).abs()

        actual_max_drift = int(
            drift.max()
        )

        if actual_max_drift > MAX_ALLOWED_DRIFT:
            problems.append(
                f"lệch {actual_max_drift} khung"
                + (
                    f" tại {worst_drift_at}"
                    if worst_drift_at
                    else ""
                )
                + f", cho phép tối đa "
                f"{MAX_ALLOWED_DRIFT}"
            )

    return problems


# ---------------------------------------------------------------------------
# GHI PARQUET
# ---------------------------------------------------------------------------

def write_frame_map(
    frame_map_df: pd.DataFrame,
) -> None:
    """
    Ghi atomically vào index/frame_map.parquet.

    Ghi file tạm trước rồi mới replace file chính.
    """

    FRAME_MAP_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Giữ extension .parquet để pandas/pyarrow
    # nhận đúng loại file.
    tmp_path = FRAME_MAP_PARQUET.with_name(
        f"{FRAME_MAP_PARQUET.stem}.tmp.parquet"
    )

    frame_map_df.to_parquet(
        tmp_path,
        index=False,
    )

    os.replace(
        tmp_path,
        FRAME_MAP_PARQUET,
    )


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------

def build_frame_map(
    strict: bool = True,
) -> pd.DataFrame:
    """
    Quét, kiểm tra và ghi frame_map.parquet.

    strict=True:
        Nếu chưa đạt thì không ghi Parquet.

    strict=False:
        Vẫn ghi để phục vụ thử nghiệm.
    """

    (
        frame_map_df,
        worst_drift,
        worst_drift_at,
    ) = scan_map_keyframes()

    problems = check_frame_map(
        frame_map_df,
        worst_drift,
        worst_drift_at,
    )

    if problems:
        message = (
            "CHƯA ĐẠT: "
            + "; ".join(problems)
            + ". Đừng đưa parquet này "
            "cho người khác dùng."
        )

        if strict:
            raise ValueError(message)

        print(message)

    write_frame_map(frame_map_df)

    clear_cache()

    return frame_map_df


# ---------------------------------------------------------------------------
# RUNTIME: dùng Parquet, không đọc CSV
# ---------------------------------------------------------------------------

def load_frame_map(
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Đọc frame_map.parquet và cache trong RAM.

    Lần gọi sau không chạm đĩa.

    Nếu chưa có Parquet thì tự build một lần.
    """

    global _CACHED_FRAME_MAP

    if refresh:
        clear_cache()

    if _CACHED_FRAME_MAP is not None:
        return _CACHED_FRAME_MAP

    if FRAME_MAP_PARQUET.exists():
        frame_map_df = pd.read_parquet(
            FRAME_MAP_PARQUET
        )

        missing = [
            column
            for column in EXPECTED_COLUMNS
            if column not in frame_map_df.columns
        ]

        if missing:
            raise ValueError(
                f"{FRAME_MAP_PARQUET} thiếu cột {missing}."
            )

        frame_map_df = frame_map_df[
            EXPECTED_COLUMNS
        ].astype(FRAME_MAP_DTYPES)

    else:
        print(
            f"Chưa có {FRAME_MAP_PARQUET}, "
            "dựng lần đầu."
        )

        frame_map_df = build_frame_map(
            strict=True
        )

    _CACHED_FRAME_MAP = frame_map_df

    return _CACHED_FRAME_MAP


# ---------------------------------------------------------------------------
# LOOKUP RUNTIME
# ---------------------------------------------------------------------------

def lookup(
    video_id: str,
    n: int,
) -> tuple[int, float]:
    """
    Tra cứu (frame_idx, pts_time) từ bảng trong RAM.

    Không mở CSV.
    Không đọc lại Parquet sau lần đầu.
    """

    global _CACHED_LOOKUP

    if _CACHED_LOOKUP is None:
        frame_map_df = load_frame_map()

        _CACHED_LOOKUP = {
            (
                str(video_id_value),
                int(n_value),
            ): (
                int(frame_idx_value),
                float(pts_time_value),
            )
            for (
                video_id_value,
                n_value,
                frame_idx_value,
                pts_time_value,
            ) in zip(
                frame_map_df["video_id"],
                frame_map_df["n"],
                frame_map_df["frame_idx"],
                frame_map_df["pts_time"],
            )
        }

    key = (
        str(video_id),
        int(n),
    )

    if key not in _CACHED_LOOKUP:
        raise KeyError(
            f"Không có ảnh thứ {n} của {video_id} "
            "trong frame map."
        )

    return _CACHED_LOOKUP[key]


# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """Xoá cache trong tiến trình hiện tại."""

    global _CACHED_FRAME_MAP
    global _CACHED_LOOKUP

    _CACHED_FRAME_MAP = None
    _CACHED_LOOKUP = None