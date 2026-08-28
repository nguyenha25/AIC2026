"""
Việc 8 — CHỈ MỤC ẢNH THỨ HAI, mã hoá bằng mô hình mạnh hơn.

HYBRID, KHÔNG PHẢI THAY THẾ
---------------------------
Chỉ mục BTC (CLIP ViT-B/32, 512 chiều) giữ NGUYÊN. Đây là nhánh RRF thứ năm,
tên `clip_l`. Trọng số mặc định 0 cho tới khi đo được.

Ba lý do chọn hybrid:

1. KHÔNG RỦI RO. Mô hình mới tệ hơn thì `do_trong_so_rrf` tự quét ra trọng số
   0, và không mất gì.

2. MÃ HOÁ TỪNG PHẦN VẪN DÙNG ĐƯỢC. Thay thế thì phải xong đủ 873 video mới
   chạy được. Hybrid thì mã hoá video nào dùng video đó — nhánh mới chỉ đơn
   giản là không trả kết quả cho video chưa mã hoá. Bốn máy chia nhau shard,
   ai xong trước dùng trước.

3. HAI MÔ HÌNH SAI THEO KIỂU KHÁC NHAU. Đó chính là thứ RRF cần: gộp CLIP với
   OCR cho 0,5000 trong khi mỗi nhánh riêng chỉ 0,3667 và 0,3000.

VÌ SAO KHÔNG PHẢI RERANK
------------------------
Rerank (Việc 4) chỉ xếp lại top-100 do B/32 trả về — nó KHÔNG kéo vào được
thứ B/32 đã bỏ sót. Câu dev 11 không nằm trong top-300 của CLIP, nên rerank
vô dụng với nó. Một chỉ mục riêng thì tìm độc lập.

Đo được: 7/12 câu KIS có khung đúng không nằm trong 500 kết quả OCR. Với nhóm
câu đó, nâng tầng lọc thô là đường duy nhất.

CÁCH KHÔNG NÊN LÀM
------------------
Nối hai vector thành 1280 chiều. Hai thang đo khác nhau, không tách ra đo
riêng được, và không tắt được nếu tệ.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Import faiss TRƯỚC torch là 0xC0000005 trên Windows — xem rank/search.py.
import faiss  # noqa: E402
import pandas as pd  # noqa: E402

from ..paths import DERIVED_DIR, FAISS_DIR

# Mô hình mặc định. Đổi thì phải mã hoá lại toàn bộ và dựng lại chỉ mục —
# tên mô hình được ghi vào tệp kèm để không lẫn hai đời vector.
MO_HINH = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
SO_CHIEU = 768

# Vector do nhóm tự mã hoá. KHÔNG để chung raw/ — đó là thư mục BTC phát.
DAC_TRUNG_DIR = DERIVED_DIR / "clip_l"

DUONG_DAN_INDEX = FAISS_DIR / "clip_l.index"
DUONG_DAN_IDS = FAISS_DIR / "clip_l_ids.parquet"
DUONG_DAN_META = FAISS_DIR / "clip_l_meta.json"

_INDEX = None
_IDS = None


def tep_dac_trung(video_id: str) -> Path:
    return DAC_TRUNG_DIR / f"{video_id}.npy"


def video_da_ma_hoa() -> list[str]:
    if not DAC_TRUNG_DIR.is_dir():
        return []
    return sorted(p.stem for p in DAC_TRUNG_DIR.glob("*.npy"))


def doc_dac_trung(video_id: str, so_hang_can: int | None = None) -> np.ndarray:
    """Đọc vector của một video, kiểm số hàng khớp frame_map.

    Số hàng lệch nghĩa là thứ tự vector KHÔNG còn khớp bảng đối chiếu, và mọi
    kết quả sẽ trỏ sai ảnh mà không có triệu chứng gì. Đây đúng cái bẫy
    n/frame_idx, chỉ khác chỗ xảy ra.
    """
    duong_dan = tep_dac_trung(video_id)
    if not duong_dan.exists():
        raise FileNotFoundError(f"Chưa mã hoá {video_id}: không thấy {duong_dan}")

    v = np.load(duong_dan)
    if v.ndim != 2:
        raise ValueError(f"{duong_dan.name} có shape {v.shape}, cần 2 chiều.")
    if v.shape[1] != SO_CHIEU:
        raise ValueError(
            f"{duong_dan.name} có {v.shape[1]} chiều, mô hình {MO_HINH} cho "
            f"{SO_CHIEU}. Trộn hai đời vector là hỏng cả chỉ mục."
        )
    if so_hang_can is not None and len(v) != so_hang_can:
        raise ValueError(
            f"{video_id}: tệp có {len(v)} hàng nhưng frame_map ghi "
            f"{so_hang_can} keyframe. Lệch số lượng thì mọi phép đổi hàng -> n "
            "đều sai."
        )
    return v


# ---------------------------------------------------------------------------
# Dựng chỉ mục
# ---------------------------------------------------------------------------

def dung_chi_muc(chi_video: list[str] | None = None) -> tuple[faiss.Index, pd.DataFrame]:
    """Gộp mọi tệp .npy đã có thành một chỉ mục FAISS + bảng thứ tự.

    Chỉ nhận video ĐÃ mã hoá. Video chưa có thì vắng mặt khỏi chỉ mục, và
    nhánh clip_l đơn giản là không trả kết quả cho nó — đúng ý đồ hybrid.
    """
    from ..frame_map import load_frame_map

    bang = load_frame_map()
    co_san = video_da_ma_hoa()
    if chi_video:
        giu = set(chi_video)
        co_san = [v for v in co_san if v in giu]

    if not co_san:
        raise FileNotFoundError(
            f"Không thấy tệp .npy nào trong {DAC_TRUNG_DIR}. "
            "Chạy notebook mã hoá trước, rồi chép kết quả về thư mục đó."
        )

    khoi: list[np.ndarray] = []
    dong: list[pd.DataFrame] = []

    for vid in co_san:
        nhom = bang[bang["video_id"] == vid].sort_values("n")
        if nhom.empty:
            print(f"  bỏ qua {vid}: không có trong frame_map")
            continue

        v = doc_dac_trung(vid, so_hang_can=len(nhom))

        # Chuẩn hoá để tích vô hướng = cosine. Chỉ mục dùng IndexFlatIP.
        do_dai = np.linalg.norm(v, axis=1, keepdims=True)
        do_dai[do_dai == 0] = 1.0
        khoi.append((v / do_dai).astype(np.float32))

        dong.append(
            nhom[["video_id", "n", "frame_idx", "pts_time"]].reset_index(drop=True)
        )

    if not khoi:
        raise ValueError("Không video nào dựng được.")

    tat_ca = np.vstack(khoi)
    ids = pd.concat(dong, ignore_index=True)

    if len(tat_ca) != len(ids):
        raise ValueError(
            f"{len(tat_ca):,} vector nhưng {len(ids):,} dòng thứ tự — lệch nhau."
        )

    index = faiss.IndexFlatIP(SO_CHIEU)
    index.add(tat_ca)
    return index, ids


def ghi_chi_muc(index: faiss.Index, ids: pd.DataFrame) -> None:
    """Ghi qua tệp tạm rồi đổi tên — ngắt giữa chừng không để lại tệp cụt."""
    import json

    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    tam = DUONG_DAN_INDEX.with_suffix(".tmp")
    faiss.write_index(index, str(tam))
    os.replace(tam, DUONG_DAN_INDEX)

    tam_ids = DUONG_DAN_IDS.with_suffix(".tmp")
    ids.to_parquet(tam_ids, index=False)
    os.replace(tam_ids, DUONG_DAN_IDS)

    with DUONG_DAN_META.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mo_hinh": MO_HINH,
                "pretrained": PRETRAINED,
                "so_chieu": SO_CHIEU,
                "so_vector": int(index.ntotal),
                "so_video": int(ids["video_id"].nunique()),
                "video": sorted(ids["video_id"].unique().tolist()),
            },
            f,
            ensure_ascii=False,
            indent=1,
        )


def nap_chi_muc() -> tuple[faiss.Index, pd.DataFrame]:
    global _INDEX, _IDS
    if _INDEX is not None:
        return _INDEX, _IDS

    if not DUONG_DAN_INDEX.exists():
        raise FileNotFoundError(
            f"Chưa có {DUONG_DAN_INDEX}. Chạy: python -m scripts.build_faiss_l"
        )

    index = faiss.read_index(str(DUONG_DAN_INDEX))
    ids = pd.read_parquet(DUONG_DAN_IDS)

    if index.ntotal != len(ids):
        raise ValueError(
            f"Chỉ mục có {index.ntotal:,} vector nhưng bảng thứ tự có "
            f"{len(ids):,} dòng. Dựng lại cả hai."
        )

    _INDEX, _IDS = index, ids
    return index, ids


def xoa_cache() -> None:
    global _INDEX, _IDS
    _INDEX = _IDS = None


# ---------------------------------------------------------------------------
# Tra cứu
# ---------------------------------------------------------------------------

_ENCODER = None


def _encoder():
    """Mô hình chữ. Nạp một lần, dùng lại."""
    global _ENCODER
    if _ENCODER is None:
        import open_clip
        import torch

        thiet_bi = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms(
            MO_HINH, pretrained=PRETRAINED, device=thiet_bi
        )
        model.eval()
        _ENCODER = (model, open_clip.get_tokenizer(MO_HINH), thiet_bi)
    return _ENCODER


def ma_hoa_cau(cau: str) -> np.ndarray:
    import torch

    model, tokenizer, thiet_bi = _encoder()
    with torch.no_grad():
        v = model.encode_text(tokenizer([cau]).to(thiet_bi))
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype(np.float32)


def tim(cau: str, top_k: int = 100) -> list[dict]:
    """Tra chỉ mục clip_l -> list[dict] đúng khuôn các nhánh khác.

    Câu vào phải là TIẾNG ANH — dùng chung bộ mở rộng của Việc 5. Nhánh gọi
    chịu trách nhiệm dịch, giống nhánh clip gốc.
    """
    index, ids = nap_chi_muc()
    diem, vi_tri = index.search(ma_hoa_cau(cau), min(top_k, index.ntotal))

    ra = []
    for d, i in zip(diem[0], vi_tri[0]):
        if i < 0:
            continue
        r = ids.iloc[int(i)]
        ra.append(
            {
                "video_id": str(r["video_id"]),
                "n": int(r["n"]),
                "frame_idx": int(r["frame_idx"]),
                "pts_time": float(r["pts_time"]),
                "score": float(d),
                "source": "clip_l",
            }
        )
    return ra
