"""
Lọc trùng — luật đã chốt ở Việc 4.

    Mỗi video giữ tối đa MỘT kết quả trong mỗi cửa sổ 10 GIÂY,
    tính theo cột giây (pts_time) của bảng đối chiếu.

VÌ SAO PHẢI LỌC
---------------
Cách chấm chỉ lấy MỘT giá trị cho mỗi mốc thứ hạng. Nộp năm tấm ảnh cùng một
cảnh là phí bốn suất trong trăm suất. Triệu chứng của việc quên lọc trùng là
điểm thấp bất thường trong khi nhìn 100 kết quả trên màn hình thấy rất đúng —
không có thông báo lỗi nào cả.

VÌ SAO TÍNH THEO GIÂY, KHÔNG THEO KHUNG HÌNH
--------------------------------------------
fps mỗi video một khác. Cùng một khoảng cách 250 khung hình sẽ ra 10 giây ở
video 25fps nhưng chỉ 8,3 giây ở video 30fps. Cửa sổ tính bằng giây thì mọi
video dùng chung một luật.

VÌ SAO SO VỚI TẤT CẢ ẢNH ĐÃ GIỮ, KHÔNG PHẢI ẢNH GIỮ GẦN NHẤT
-------------------------------------------------------------
Cách làm sai và rất dễ viết nhầm: chia thời gian thành ô cố định
floor(pts_time / 10) rồi mỗi ô giữ một ảnh. Hai ảnh ở giây 9,9 và 10,1 rơi vào
hai ô khác nhau nên cả hai được giữ — trong khi chúng chỉ cách nhau 0,2 giây và
gần như chắc chắn là cùng một cảnh. Cổng thoát số 5 kiểm theo TỪNG CẶP nên cách
chia ô sẽ trượt cổng.

Hàm ở đây so ảnh đang xét với TOÀN BỘ ảnh đã giữ của cùng video đó. Mỗi truy vấn
chỉ có vài trăm ứng viên nên chi phí không đáng kể.

ĐẦU VÀO
-------
Danh sách đã xếp hạng sẵn, tốt nhất đứng đầu. Hàm giữ nguyên thứ tự đó — ảnh
điểm cao được giữ, ảnh điểm thấp hơn trong cùng cửa sổ bị loại. Truyền vào một
danh sách chưa sắp xếp là loại nhầm ảnh tốt mà không có báo lỗi.

Chỉ cần đối tượng có ba thuộc tính: video_id, pts_time, score. Cố ý không import
Hit từ nhánh index để tệp này chạy được mà không cần faiss — bộ test lọc trùng
nhờ vậy chạy trên máy chưa cài xong thư viện nặng.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, TypeVar

# Sai số dấu phẩy động. pts_time đọc từ CSV nên 10.0 có thể thành
# 9.999999999999998; không có ngưỡng này thì một cặp đúng 10 giây chẵn có lúc
# bị loại có lúc không, tuỳ video.
EPSILON = 1e-9


class CoThoiGian(Protocol):
    """Thứ duy nhất hàm lọc trùng cần biết về một kết quả."""

    video_id: str
    pts_time: float
    score: float


T = TypeVar("T", bound=CoThoiGian)


@dataclass
class BaoCaoLocTrung:
    """Con số để in ra và để ghi vào runs/."""

    vao: int
    ra: int
    bo_vi_gan: int
    bo_vi_qua_tran: int
    cua_so_giay: float

    def __str__(self) -> str:
        return (
            f"Lọc trùng {self.cua_so_giay:.1f} giây: "
            f"{self.vao} → {self.ra} kết quả | "
            f"bỏ vì cùng video cách dưới {self.cua_so_giay:.1f} giây: {self.bo_vi_gan} | "
            f"bỏ vì quá trần mỗi video: {self.bo_vi_qua_tran}"
        )


def loc_trung(
    ket_qua: Iterable[T],
    cua_so_giay: float = 10.0,
    so_anh_toi_da_moi_video: int | None = None,
) -> tuple[list[T], BaoCaoLocTrung]:
    """
    Áp luật chốt: mỗi video tối đa một kết quả trong mỗi cửa sổ `cua_so_giay`.

    Tham số:
        ket_qua: danh sách ĐÃ XẾP HẠNG, tốt nhất đứng đầu.
        cua_so_giay: bề rộng cửa sổ. Hai ảnh cùng video cách nhau ĐÚNG bằng
                     con số này thì vẫn giữ cả hai — luật là "cách nhau DƯỚI
                     10 giây" mới bỏ, và cổng thoát số 5 cũng kiểm như vậy.
        so_anh_toi_da_moi_video: None nghĩa là không giới hạn (đúng luật chốt).

    Trả về:
        (danh sách đã lọc, báo cáo)
    """
    if cua_so_giay < 0:
        raise ValueError(f"Cửa sổ lọc trùng không được âm: {cua_so_giay}")

    giu: list[T] = []
    da_giu_theo_video: dict[str, list[float]] = {}

    vao = 0
    bo_vi_gan = 0
    bo_vi_qua_tran = 0

    for muc in ket_qua:
        vao += 1

        moc_thoi_gian = da_giu_theo_video.setdefault(muc.video_id, [])

        if (
            so_anh_toi_da_moi_video is not None
            and len(moc_thoi_gian) >= so_anh_toi_da_moi_video
        ):
            bo_vi_qua_tran += 1
            continue

        giay = float(muc.pts_time)

        qua_gan = any(
            abs(giay - da_co) < cua_so_giay - EPSILON for da_co in moc_thoi_gian
        )

        if qua_gan:
            bo_vi_gan += 1
            continue

        moc_thoi_gian.append(giay)
        giu.append(muc)

    bao_cao = BaoCaoLocTrung(
        vao=vao,
        ra=len(giu),
        bo_vi_gan=bo_vi_gan,
        bo_vi_qua_tran=bo_vi_qua_tran,
        cua_so_giay=cua_so_giay,
    )

    return giu, bao_cao


def tim_cap_qua_gan(
    ket_qua: Iterable[CoThoiGian],
    cua_so_giay: float = 10.0,
) -> list[tuple[int, int, str, float]]:
    """
    Soát cổng thoát số 5: trong danh sách có hai dòng nào cùng video mà cách
    nhau dưới `cua_so_giay` không.

    Trả về danh sách (vị trí dòng 1, vị trí dòng 2, video_id, khoảng cách giây).
    Danh sách RỖNG nghĩa là ĐẠT.

    Đây là phép kiểm độc lập với hàm loc_trung() ở trên — cố ý viết lại bằng
    cách khác (so mọi cặp) chứ không gọi lại hàm kia. Kiểm bằng chính cái mình
    vừa chạy thì lỗi logic sẽ tự xác nhận chính nó là đúng.
    """
    danh_sach = list(ket_qua)
    vi_pham: list[tuple[int, int, str, float]] = []

    for i in range(len(danh_sach)):
        for j in range(i + 1, len(danh_sach)):
            a, b = danh_sach[i], danh_sach[j]

            if a.video_id != b.video_id:
                continue

            khoang_cach = abs(float(a.pts_time) - float(b.pts_time))

            if khoang_cach < cua_so_giay - EPSILON:
                vi_pham.append((i, j, a.video_id, khoang_cach))

    return vi_pham