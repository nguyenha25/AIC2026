"""
Việc 4 — XẾP HẠNG LẠI TOP-100 BẰNG MÔ HÌNH MẠNH HƠN.

Ý tưởng: CLIP B/32 của BTC làm TẦNG LỌC THÔ trên 177.321 ảnh (nhanh, đã có
sẵn vector, không phải mã hoá lại gì). Chỉ 100 ảnh sống sót mới được đưa qua
một mô hình nặng hơn để xếp lại. Mã hoá 100 ảnh thì chạy được cả trên CPU —
đây là lý do Việc 4 có ROI cao nhất Giai đoạn 2: không dựng thêm hạ tầng,
không mã hoá lại toàn kho như Việc 8.

BA ĐIỀU KIỆN ĐỂ CON SỐ ĐO ĐƯỢC LÀ THẬT
--------------------------------------
1. TẤT ĐỊNH. Mô hình ở chế độ eval, không dropout, không random crop, chạy
   trong torch.inference_mode(). Chạy hai lần phải ra kết quả GIỐNG HỆT — checklist
   yêu cầu đúng điều này. `kiem_tat_dinh()` ở cuối tệp kiểm hộ.

2. KHÔNG NUỐT ẢNH THIẾU. Ảnh không có trên đĩa thì ứng viên đó KHÔNG bị xoá,
   mà bị đẩy xuống cuối và đếm vào `so_thieu_anh`. Xoá âm thầm thì một máy
   tải thiếu shard sẽ "cải thiện" điểm một cách giả tạo.

3. ĐO TRƯỚC KHI GỘP NGUỒN. Điểm cosine của mô hình rerank KHÔNG cùng thang
   với điểm RRF đa nguồn (cỡ 0,016). Rerank chỉ thay thứ tự nhánh CLIP rồi
   nhánh đó mới được gộp với OCR/ASR; không cộng thẳng hai loại điểm.

CÁCH DÙNG
---------
    from aic2026.rerank import Reranker

    bo_xep_lai = Reranker()                     # nạp mô hình một lần
    hits_moi, bao_cao = bo_xep_lai.xep_lai(cau_hoi, hits, so_dau=100)

Cắm vào mạch của Ngân qua run_query(tim_ung_vien=...):

    from aic2026.rerank import boc_them_rerank
    tim = boc_them_rerank(tim_ung_vien_goc, bo_xep_lai)
    run_query(cau_hoi, tim_ung_vien=tim)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# Mặc định: ViT-L/14 của LAION. Nặng hơn B/32 khoảng 5 lần nhưng chỉ chạy
# trên 100 ảnh. Muốn thử SigLIP2 thì đổi trong config/settings.yaml:
#   rerank:
#     mo_hinh: ViT-SO400M-14-SigLIP-384
#     pretrained: webli
MO_HINH_MAC_DINH = "ViT-L-14"
PRETRAINED_MAC_DINH = "laion2b_s32b_b82k"
KICH_THUOC_LO_MAC_DINH = 8


@dataclass
class BaoCaoXepLai:
    """Số liệu một lần xếp lại — ghi vào runs/ để sau còn đối chiếu."""

    so_ung_vien_vao: int
    so_da_xep_lai: int
    so_thieu_anh: int
    so_dao_thu_hang: int          # bao nhiêu ứng viên đổi vị trí
    thoi_gian_ms: float
    mo_hinh: str
    kich_thuoc_lo: int = KICH_THUOC_LO_MAC_DINH
    cach_gop: str = "thay"

    def __str__(self) -> str:
        return (
            f"xếp lại {self.so_da_xep_lai}/{self.so_ung_vien_vao} ứng viên | "
            f"thiếu ảnh: {self.so_thieu_anh} | đảo hạng: {self.so_dao_thu_hang} | "
            f"{self.thoi_gian_ms:.0f} ms | {self.mo_hinh} | "
            f"batch={self.kich_thuoc_lo} | {self.cach_gop}"
        )


def _doc_cau_hinh() -> dict[str, Any]:
    try:
        from .rank.config import load_settings

        muc = load_settings().get("rerank")
        return muc if isinstance(muc, dict) else {}
    except Exception:
        return {}


class Reranker:
    """Bộ xếp hạng lại. Nạp mô hình MỘT LẦN rồi dùng cho mọi câu hỏi."""

    def __init__(
        self,
        mo_hinh: str | None = None,
        pretrained: str | None = None,
        thiet_bi: str | None = None,
        duong_dan_anh: Callable[[str, int], Path] | None = None,
        kich_thuoc_lo: int | None = None,
        tat_dinh: bool | None = None,
    ):
        cau_hinh = _doc_cau_hinh()
        self.ten_mo_hinh = mo_hinh or cau_hinh.get("mo_hinh", MO_HINH_MAC_DINH)
        self.pretrained = pretrained or cau_hinh.get("pretrained", PRETRAINED_MAC_DINH)
        self._thiet_bi_yeu_cau = thiet_bi or cau_hinh.get("thiet_bi")
        self.kich_thuoc_lo = int(
            kich_thuoc_lo or cau_hinh.get("kich_thuoc_lo", KICH_THUOC_LO_MAC_DINH)
        )
        if self.kich_thuoc_lo <= 0:
            raise ValueError("kich_thuoc_lo phải lớn hơn 0")
        self.so_dau_mac_dinh = int(cau_hinh.get("so_dau", 100))
        self.cach_gop_mac_dinh = str(cau_hinh.get("cach_gop", "thay"))
        self.tat_dinh = bool(
            cau_hinh.get("tat_dinh", True) if tat_dinh is None else tat_dinh
        )

        if duong_dan_anh is None:
            from .paths import keyframe_image

            duong_dan_anh = keyframe_image
        self.duong_dan_anh = duong_dan_anh

        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._thiet_bi = None
        # Vector ảnh đã mã hoá, dùng lại giữa các câu hỏi. Bộ dev 60 câu tra
        # trùng ảnh rất nhiều; không nhớ lại thì mã hoá lặp hàng nghìn lần.
        self._nho_anh: dict[tuple[str, int], np.ndarray] = {}

    # -- nạp mô hình -------------------------------------------------------

    def _nap(self):
        if self._model is not None:
            return

        import open_clip
        import torch

        self._thiet_bi = self._thiet_bi_yeu_cau or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.tat_dinh:
            # Hai lần nghiệm thu phải cho cùng thứ tự. Các cờ này tắt việc
            # cuDNN tự chọn thuật toán khác nhau giữa hai lượt trên GPU.
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # torch cũ chưa có warn_only
                torch.use_deterministic_algorithms(True)

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.ten_mo_hinh,
            pretrained=self.pretrained,
            device=self._thiet_bi,
        )
        model.eval()                       # BẮT BUỘC: tắt dropout -> tất định

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.ten_mo_hinh)

    @property
    def ten_day_du(self) -> str:
        return f"{self.ten_mo_hinh}/{self.pretrained}"

    # -- mã hoá ------------------------------------------------------------

    def ma_hoa_cau(self, cau_hoi: str) -> np.ndarray:
        import torch

        if not cau_hoi or not cau_hoi.strip():
            raise ValueError("Câu truy vấn rỗng.")

        self._nap()
        with torch.inference_mode():
            token = self._tokenizer([cau_hoi.strip()]).to(self._thiet_bi)
            v = self._model.encode_text(token)
            v = v / v.norm(dim=-1, keepdim=True)
        ra = v.float().cpu().numpy().astype(np.float32, copy=False).ravel()
        if not np.isfinite(ra).all():
            raise ValueError("Mô hình sinh vector câu chứa NaN hoặc vô hạn.")
        return ra

    def ma_hoa_anh(self, duong_dan: Path) -> np.ndarray:
        """Mã hoá một ảnh. Đường chạy chính dùng `ma_hoa_nhieu_anh()` theo lô."""
        return self.ma_hoa_nhieu_anh([duong_dan], kich_thuoc_lo=1)[0]

    def ma_hoa_nhieu_anh(
        self,
        cac_duong_dan: Sequence[Path],
        kich_thuoc_lo: int | None = None,
    ) -> list[np.ndarray]:
        """Mã hoá ảnh theo lô để top-100 không thành 100 lượt gọi mô hình.

        Chỉ giữ tensor của một lô trong RAM/VRAM. Vì vậy máy không có GPU vẫn
        chạy được, còn GPU nhỏ có thể hạ `rerank.kich_thuoc_lo` xuống 4 hoặc 2.
        Thứ tự vector trả về luôn khớp thứ tự đường dẫn đầu vào.
        """
        import torch
        from PIL import Image

        self._nap()
        lo = int(kich_thuoc_lo or self.kich_thuoc_lo)
        if lo <= 0:
            raise ValueError("kich_thuoc_lo phải lớn hơn 0")

        ra: list[np.ndarray] = []
        for bat_dau in range(0, len(cac_duong_dan), lo):
            duong_dan_lo = cac_duong_dan[bat_dau: bat_dau + lo]
            tensor_anh = []
            for duong_dan in duong_dan_lo:
                with Image.open(duong_dan) as anh:
                    tensor_anh.append(self._preprocess(anh.convert("RGB")))

            if not tensor_anh:
                continue

            x = torch.stack(tensor_anh).to(self._thiet_bi)
            with torch.inference_mode():
                v = self._model.encode_image(x)
                v = v / v.norm(dim=-1, keepdim=True)
            mang = v.float().cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(mang).all():
                raise ValueError("Mô hình sinh vector ảnh chứa NaN hoặc vô hạn.")
            ra.extend(hang.copy() for hang in mang)

        return ra

    def _vector_anh(self, video_id: str, n: int) -> np.ndarray | None:
        khoa = (video_id, int(n))
        if khoa in self._nho_anh:
            return self._nho_anh[khoa]

        duong_dan = self.duong_dan_anh(video_id, int(n))
        if not Path(duong_dan).exists():
            return None

        v = self.ma_hoa_anh(Path(duong_dan))
        self._nho_anh[khoa] = v
        return v

    def _vector_nhieu_anh(
        self,
        cac_khoa: Sequence[tuple[str, int]],
    ) -> dict[tuple[str, int], np.ndarray | None]:
        """Lấy vector cache và mã hoá gộp mọi ảnh chưa có trong cache."""
        ket_qua: dict[tuple[str, int], np.ndarray | None] = {}
        khoa_moi: list[tuple[str, int]] = []
        duong_dan_moi: list[Path] = []
        da_xet: set[tuple[str, int]] = set()

        for video_id, n in cac_khoa:
            khoa = (str(video_id), int(n))
            if khoa in da_xet:
                continue
            da_xet.add(khoa)
            if khoa in self._nho_anh:
                ket_qua[khoa] = self._nho_anh[khoa]
                continue

            duong_dan = Path(self.duong_dan_anh(*khoa))
            if not duong_dan.exists():
                ket_qua[khoa] = None
                continue
            khoa_moi.append(khoa)
            duong_dan_moi.append(duong_dan)

        if duong_dan_moi:
            cac_vector = self.ma_hoa_nhieu_anh(duong_dan_moi)
            if len(cac_vector) != len(khoa_moi):
                raise RuntimeError(
                    "Số vector ảnh trả về không khớp số ảnh đưa vào rerank."
                )
            for khoa, vector in zip(khoa_moi, cac_vector):
                self._nho_anh[khoa] = vector
                ket_qua[khoa] = vector

        return ket_qua

    def xoa_nho_anh(self) -> None:
        """Xoá cache vector ảnh, dùng trước lượt nghiệm thu độc lập thứ hai."""
        self._nho_anh.clear()

    # -- xếp lại -----------------------------------------------------------

    def xep_lai(
        self,
        cau_hoi: str,
        ung_vien: Sequence,
        so_dau: int | None = None,
        cach_gop: str | None = None,
        k_rrf: int = 60,
    ) -> tuple[list, BaoCaoXepLai]:
        """Xếp lại `so_dau` ứng viên đầu tiên. Phần còn lại giữ nguyên phía sau.

        cach_gop:
            "thay"  thứ tự mới hoàn toàn theo điểm mô hình rerank. Đây là
                    nghĩa gốc của rerank, dễ giải thích khi điểm đổi.
            "rrf"   gộp HẠNG cũ với HẠNG mới bằng RRF. Chịu lỗi tốt hơn khi
                    mô hình rerank thỉnh thoảng đoán sai, nhưng chậm cải
                    thiện hơn. Đo cả hai trên dev v2 rồi hãy chốt.
        """
        import time

        bat_dau = time.perf_counter()
        so_dau = self.so_dau_mac_dinh if so_dau is None else int(so_dau)
        cach_gop = cach_gop or self.cach_gop_mac_dinh
        if so_dau <= 0:
            raise ValueError("so_dau phải lớn hơn 0")
        if k_rrf < 0:
            raise ValueError("k_rrf không được âm")
        ung_vien = list(ung_vien)
        dau = ung_vien[:so_dau]
        duoi = ung_vien[so_dau:]

        if not dau:
            return ung_vien, BaoCaoXepLai(
                0, 0, 0, 0, 0.0, self.ten_day_du,
                self.kich_thuoc_lo, cach_gop,
            )

        v_cau = self.ma_hoa_cau(cau_hoi)
        cac_khoa = [(str(h.video_id), int(h.n)) for h in dau]
        cac_vector = self._vector_nhieu_anh(cac_khoa)

        cham_diem: list[tuple[float, int, Any]] = []   # (điểm, hạng cũ, hit)
        thieu: list = []

        for hang_cu, (h, khoa) in enumerate(zip(dau, cac_khoa)):
            v_anh = cac_vector[khoa]
            if v_anh is None:
                thieu.append(h)
                continue
            cham_diem.append((float(np.dot(v_cau, v_anh)), hang_cu, h))

        if cach_gop == "rrf":
            theo_moi = sorted(cham_diem, key=lambda t: t[0], reverse=True)
            hang_moi = {id(t[2]): i for i, t in enumerate(theo_moi, start=1)}
            cham_diem.sort(
                key=lambda t: (
                    1.0 / (k_rrf + t[1] + 1) + 1.0 / (k_rrf + hang_moi[id(t[2])])
                ),
                reverse=True,
            )
        elif cach_gop == "thay":
            # Khoá phụ là hạng cũ: hai ảnh cùng điểm thì giữ nguyên thứ tự cũ,
            # nhờ vậy chạy hai lần ra kết quả giống hệt.
            cham_diem.sort(key=lambda t: (-t[0], t[1]))
        else:
            raise ValueError(f"cach_gop lạ: {cach_gop}")

        ra = []
        for diem, _, h in cham_diem:
            # Hit là frozen dataclass -> dùng replace, không gán trực tiếp.
            ra.append(_thay_diem(h, diem))

        # Thiếu ảnh xuống cuối nhóm đầu, KHÔNG bị xoá.
        ket_qua = ra + thieu + duoi

        dao = sum(
            1
            for i, (moi, cu) in enumerate(zip(ket_qua[: len(dau)], dau))
            if str(moi.video_id) != str(cu.video_id) or int(moi.n) != int(cu.n)
        )

        bao_cao = BaoCaoXepLai(
            so_ung_vien_vao=len(ung_vien),
            so_da_xep_lai=len(ra),
            so_thieu_anh=len(thieu),
            so_dao_thu_hang=dao,
            thoi_gian_ms=(time.perf_counter() - bat_dau) * 1000,
            mo_hinh=self.ten_day_du,
            kich_thuoc_lo=self.kich_thuoc_lo,
            cach_gop=cach_gop,
        )
        return ket_qua, bao_cao

    def kiem_tat_dinh(
        self,
        cau_hoi: str,
        ung_vien: Sequence,
        so_dau: int | None = None,
        cach_gop: str | None = None,
    ) -> bool:
        """Mã hoá lại từ đầu hai lần và so cả thứ tự lẫn điểm số."""
        lan_1, _ = self.xep_lai(
            cau_hoi, ung_vien, so_dau=so_dau, cach_gop=cach_gop
        )
        self.xoa_nho_anh()
        lan_2, _ = self.xep_lai(
            cau_hoi, ung_vien, so_dau=so_dau, cach_gop=cach_gop
        )
        return _chu_ky(lan_1) == _chu_ky(lan_2)


def _thay_diem(hit, diem: float):
    """Đổi score của một Hit (frozen dataclass) mà giữ nguyên các trường khác."""
    import dataclasses

    if isinstance(hit, Mapping):
        ra = dict(hit)
        nguon = str(ra.get("source", ""))
        ra["score"] = float(diem)
        ra["source"] = nguon if "rerank" in nguon.split("+") else f"{nguon}+rerank".lstrip("+")
        return ra

    if not dataclasses.is_dataclass(hit):
        raise TypeError(
            "Ứng viên rerank phải là dataclass Hit hoặc mapping có video_id/n."
        )

    nguon = str(getattr(hit, "source", ""))
    nguon_moi = nguon if "rerank" in nguon.split("+") else f"{nguon}+rerank".lstrip("+")
    return dataclasses.replace(hit, score=float(diem), source=nguon_moi)


def _chu_ky(hits: Sequence) -> list[tuple[str, int, float]]:
    """Chữ ký nghiệm thu: định danh, thứ tự và điểm float32 của từng hit."""
    ra = []
    for h in hits:
        if isinstance(h, Mapping):
            video_id, n, score = h["video_id"], h["n"], h.get("score", 0.0)
        else:
            video_id, n, score = h.video_id, h.n, h.score
        ra.append((str(video_id), int(n), float(np.float32(score))))
    return ra


def boc_them_rerank(
    tim_ung_vien: Callable[[str, int], list],
    bo_xep_lai: Reranker | None = None,
    so_dau: int | None = None,
    cach_gop: str | None = None,
    bien_doi_cau: Callable[[str], str] | None = None,
) -> Callable[[str, int], list]:
    """Bọc một hàm tìm ứng viên để nó tự xếp lại top-N trước khi trả về.

    Giữ nguyên chữ ký (câu chữ, số ứng viên) -> list[Hit] nên cắm thẳng vào
    run_query(tim_ung_vien=...) mà không phải sửa mạch của Ngân.
    """
    bo_xep_lai = bo_xep_lai or Reranker()

    def _tim(cau_hoi: str, so_ung_vien: int):
        tho = tim_ung_vien(cau_hoi, so_ung_vien)
        cau_cho_mo_hinh = bien_doi_cau(cau_hoi) if bien_doi_cau else cau_hoi
        da_xep, _ = bo_xep_lai.xep_lai(
            cau_cho_mo_hinh, tho, so_dau=so_dau, cach_gop=cach_gop
        )
        return da_xep

    return _tim
