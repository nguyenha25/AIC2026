"""
Việc 3c — HÀNG THỨ i TRONG .npy  ->  n  ->  frame_idx.

CÁI BẪY NÀY ĐÃ LÀM SAI CẢ BÀI NỘP MỘT LẦN
-----------------------------------------
Notebook baseline của BTC gọi số thứ tự keyframe là `frameid`. Nó KHÔNG phải
vị trí khung hình trong video. Ba con số khác nhau, rất dễ nhầm:

    i          chỉ số HÀNG trong tệp .npy      -> đếm từ 0
    n          số thứ tự tấm ảnh (0047.jpg)    -> đếm từ 1
    frame_idx  VỊ TRÍ KHUNG HÌNH TRONG VIDEO   -> SỐ PHẢI NỘP

Nộp `i` hay `n` thay cho `frame_idx` thì tệp nộp vẫn ĐÚNG ĐỊNH DẠNG, vẫn chấm
được, và trượt sạch mà không có một thông báo lỗi nào.

QUAN HỆ i -> n
--------------
Hàng i của .npy ứng với tấm ảnh thứ i khi SẮP XẾP TÊN TỆP tăng dần. Với dữ
liệu này n chạy liền 1, 2, 3... nên n = i + 1. Nhưng tệp này KHÔNG hardcode
công thức đó: nó lấy danh sách n THẬT của video từ frame_map, sắp tăng dần,
rồi tra theo vị trí. Nếu một video nào đó có n khuyết số thì hàm vẫn đúng,
còn `kiem_gia_dinh_n_bang_i_cong_1()` sẽ báo cho biết video nào lệch.

CÁCH DÙNG
---------
    from aic2026.kf_index import hang_sang_frame_idx, doc_dac_trung_theo_hang

    frame_idx = hang_sang_frame_idx("L21_V001", 0)      # hàng đầu tiên
    vector    = doc_dac_trung_theo_hang("L21_V001", 0)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .frame_map import load_frame_map
from .paths import KEYFRAMES_DIR, clip_features_file, resolve


@dataclass(frozen=True)
class MocKeyframe:
    """Một hàng của .npy đã được dịch sang đủ ba hệ số đếm."""

    video_id: str
    i: int            # chỉ số hàng trong .npy, từ 0
    n: int            # số thứ tự tấm ảnh, từ 1
    frame_idx: int    # vị trí khung hình — SỐ PHẢI NỘP
    pts_time: float

    @property
    def ten_anh(self) -> str:
        return f"{self.n:04d}.jpg"


# ---------------------------------------------------------------------------
# Bảng thứ tự của một video
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _bang_video(video_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mảng n, mảng frame_idx, mảng pts_time) đã SẮP THEO n TĂNG DẦN.

    Sắp theo n chứ không theo thứ tự dòng trong parquet: thứ tự hàng của .npy
    là thứ tự TÊN TỆP, và tên tệp là n có đệm 0 nên sắp theo chuỗi trùng với
    sắp theo số.
    """
    bang = load_frame_map()
    nhom = bang[bang["video_id"] == video_id]

    if nhom.empty:
        raise KeyError(
            f"{video_id} không có trong frame_map. Máy này đã tải "
            "map-keyframes của video đó chưa?"
        )

    nhom = nhom.sort_values("n")
    return (
        nhom["n"].to_numpy(),
        nhom["frame_idx"].to_numpy(),
        nhom["pts_time"].to_numpy(),
    )


def so_hang(video_id: str) -> int:
    """Số keyframe của video theo frame_map — cũng là số hàng .npy phải có."""
    return len(_bang_video(video_id)[0])


# ---------------------------------------------------------------------------
# Bốn phép đổi
# ---------------------------------------------------------------------------

def hang_sang_n(video_id: str, i: int) -> int:
    """Hàng i (đếm từ 0) -> n (đếm từ 1)."""
    cot_n, _, _ = _bang_video(video_id)
    if not 0 <= i < len(cot_n):
        raise IndexError(
            f"{video_id} có {len(cot_n)} keyframe, không có hàng {i}."
        )
    return int(cot_n[i])


def n_sang_hang(video_id: str, n: int) -> int:
    """n -> hàng i. Phép ngược của hang_sang_n."""
    cot_n, _, _ = _bang_video(video_id)
    vi_tri = int(np.searchsorted(cot_n, n))
    if vi_tri >= len(cot_n) or int(cot_n[vi_tri]) != int(n):
        raise KeyError(f"{video_id} không có tấm ảnh thứ {n}.")
    return vi_tri


def hang_sang_frame_idx(video_id: str, i: int) -> int:
    """Hàng i trong .npy -> frame_idx. ĐÂY LÀ HÀM CHÍNH CỦA VIỆC 3c."""
    cot_n, cot_frame, _ = _bang_video(video_id)
    if not 0 <= i < len(cot_n):
        raise IndexError(
            f"{video_id} có {len(cot_n)} keyframe, không có hàng {i}."
        )
    return int(cot_frame[i])


def hang_sang_moc(video_id: str, i: int) -> MocKeyframe:
    """Hàng i -> đủ cả bốn số, để in ra hoặc ghi log mà không tra lại ba lần."""
    cot_n, cot_frame, cot_pts = _bang_video(video_id)
    if not 0 <= i < len(cot_n):
        raise IndexError(
            f"{video_id} có {len(cot_n)} keyframe, không có hàng {i}."
        )
    return MocKeyframe(
        video_id=video_id,
        i=i,
        n=int(cot_n[i]),
        frame_idx=int(cot_frame[i]),
        pts_time=float(cot_pts[i]),
    )


# ---------------------------------------------------------------------------
# Đọc .npy đúng thứ tự
# ---------------------------------------------------------------------------

def doc_dac_trung(video_id: str, kiem_so_hang: bool = True) -> np.ndarray:
    """Đọc raw/clip-features-32/<video>.npy, kiểm số hàng khớp frame_map.

    Số hàng lệch nghĩa là thứ tự vector KHÔNG còn khớp bảng đối chiếu, và mọi
    kết quả tìm kiếm sẽ trỏ sai ảnh mà không có triệu chứng gì. Dừng ngay.
    """
    duong_dan = clip_features_file(video_id)
    if not duong_dan.exists():
        raise FileNotFoundError(
            f"Không thấy {duong_dan}. Đã giải nén clip-features-32 chưa?"
        )

    dac_trung = np.load(duong_dan)

    if dac_trung.ndim != 2:
        raise ValueError(f"{duong_dan.name} có shape {dac_trung.shape}, cần 2 chiều.")

    if kiem_so_hang:
        can = so_hang(video_id)
        if len(dac_trung) != can:
            raise ValueError(
                f"{video_id}: .npy có {len(dac_trung)} hàng nhưng frame_map ghi "
                f"{can} keyframe. Lệch số lượng thì mọi phép đổi hàng -> n đều "
                "sai. Kiểm lại xem đã giải nén đủ chưa."
            )

    return dac_trung


def doc_dac_trung_theo_hang(video_id: str, i: int) -> np.ndarray:
    """Vector của hàng i, đã kiểm số hàng."""
    return doc_dac_trung(video_id)[i]


def doc_dac_trung_theo_n(video_id: str, n: int) -> np.ndarray:
    """Vector của tấm ảnh thứ n."""
    return doc_dac_trung(video_id)[n_sang_hang(video_id, n)]


# ---------------------------------------------------------------------------
# Soát lại — dòng "Cách xác nhận đã xong" của checklist
# ---------------------------------------------------------------------------

def kiem_gia_dinh_n_bang_i_cong_1(video_ids: list[str] | None = None) -> dict:
    """Kiểm giả định n = i + 1 trên toàn kho.

    Đây là giả định của notebook baseline BTC. Tệp này không dựa vào nó, nhưng
    biết nó có đúng hay không thì mới yên tâm đọc lại mã của người khác.
    """
    bang = load_frame_map()
    video_ids = video_ids or sorted(bang["video_id"].unique().tolist())

    lech: list[str] = []
    for vid in video_ids:
        cot_n, _, _ = _bang_video(str(vid))
        if not np.array_equal(cot_n, np.arange(1, len(cot_n) + 1)):
            lech.append(str(vid))

    return {
        "so_video_kiem": len(video_ids),
        "so_video_lech": len(lech),
        "video_lech": lech[:20],
        "ket_luan": (
            "n = i + 1 đúng trên mọi video"
            if not lech
            else f"{len(lech)} video có n KHÔNG liền mạch — phải dùng hàm của "
                 "kf_index, không được tự cộng 1"
        ),
    }


def kiem_ngau_nhien(so_mau: int = 20, seed: int = 20260822) -> dict:
    """Lấy ngẫu nhiên `so_mau` keyframe, đối chiếu ba đường tính frame_idx.

    Ba đường phải cho cùng một số:
        1. hang_sang_frame_idx(video, i)
        2. tra thẳng frame_map theo (video, n)
        3. floor(pts_time * fps)

    Đường 3 là phép kiểm độc lập: nó không đi qua bảng nào của nhóm cả.
    """
    bang = load_frame_map()
    rng = random.Random(seed)
    vi_tri = rng.sample(range(len(bang)), min(so_mau, len(bang)))

    mau: list[dict] = []
    so_lech = 0

    for vt in vi_tri:
        dong = bang.iloc[vt]
        vid = str(dong["video_id"])
        n = int(dong["n"])
        i = n_sang_hang(vid, n)

        qua_kf_index = hang_sang_frame_idx(vid, i)
        tu_bang = int(dong["frame_idx"])
        tu_cong_thuc = int(float(dong["pts_time"]) * float(dong["fps"]))  # floor

        khop = qua_kf_index == tu_bang == tu_cong_thuc
        if not khop:
            so_lech += 1

        mau.append(
            {
                "video_id": vid,
                "i": i,
                "n": n,
                "frame_idx_kf_index": qua_kf_index,
                "frame_idx_frame_map": tu_bang,
                "frame_idx_floor_pts_fps": tu_cong_thuc,
                "khop": khop,
            }
        )

    return {
        "so_mau": len(mau),
        "so_lech": so_lech,
        "dat": so_lech == 0,
        "mau": mau,
    }


def kiem_ten_tep_anh(video_id: str) -> dict:
    """So thứ tự TÊN TỆP ảnh thật trên đĩa với thứ tự n của frame_map.

    Chỉ chạy được trên máy đã tải raw/keyframes/ của video đó. Đây là phép
    kiểm duy nhất chạm vào thứ tự tên tệp THẬT — hai hàm trên vẫn chỉ tin
    frame_map.
    """
    goc = resolve(KEYFRAMES_DIR, video_id) or KEYFRAMES_DIR / video_id
    if not goc.is_dir():
        return {"co_anh": False, "ly_do": f"Chưa tải ảnh của {video_id} ({goc})"}

    from .paths import bang_tep_theo_so

    # So theo SỐ trong tên tệp, không so theo chuỗi tên: BTC đệm ba chữ số
    # (001.jpg) chứ không phải bốn, nên so chuỗi là lệch toàn bộ dù dữ liệu
    # hoàn toàn đúng.
    bang = bang_tep_theo_so(goc, ".jpg")
    so_tren_dia = sorted(bang)
    cot_n, _, _ = _bang_video(video_id)
    can = [int(x) for x in cot_n]

    return {
        "co_anh": True,
        "so_anh_tren_dia": len(so_tren_dia),
        "so_anh_theo_frame_map": len(can),
        "khop_thu_tu": so_tren_dia == can,
        "kieu_dem": (
            len(next(iter(bang.values())).stem) if bang else 0
        ),
        "vi_du_lech": [
            (a, b) for a, b in zip(so_tren_dia, can) if a != b
        ][:5],
    }


def xoa_cache() -> None:
    """Gọi sau khi dựng lại frame_map trong cùng một tiến trình."""
    _bang_video.cache_clear()
