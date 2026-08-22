"""
Gộp nhiều bảng xếp hạng bằng Reciprocal Rank Fusion — Task 10.

VÌ SAO LÀ TỆP RIÊNG, KHÔNG NỐI VÀO search.py
--------------------------------------------
`rank/search.py` là mạch đầu-cuối của Ngân (Việc 4). Task 10 chỉ cần cắm vào
tham số `tim_ung_vien` mà Ngân đã chừa sẵn ở `run_query()`, nên KHÔNG có lý do
gì phải sửa tệp của Ngân. Đặt riêng ở đây thì hai người làm song song không
đụng nhau, và việc gộp có thể thay/tắt mà mạch chính không phải đổi một dòng.

(Trước đây bản Task 10 ghi đè `search.py`, làm mất ~340 dòng mạch đầu-cuối của
Ngân — gồm cả `_gom_trake` khiến TRAKE sai định dạng và luôn 0 điểm. Tách tệp
là để chuyện đó không lặp lại.)

=============================================================================
VIỆC 7 (GIAI ĐOẠN 2) — ĐỔI KHOÁ GỘP SANG (video_id, pts_time)
=============================================================================
Bản cũ gộp theo `(video_id, frame_idx)`. Sai, và sai âm thầm.

`frame_idx = floor(pts_time × fps)`. Hai keyframe KỀ NHAU trong cùng video có
thể làm tròn về CÙNG một frame_idx — đã đếm được ~1.228 dòng như vậy trên 192
video. Hậu quả:

  * Hai tấm ảnh KHÁC NHAU bị coi là MỘT khi gộp. Ảnh đúng bị ảnh sai nuốt
    mất, hoặc ngược lại, tuỳ tấm nào vào bảng trước.
  * `pts_time` và `n` của kết quả gộp lấy từ tấm vào trước — nên bước lọc
    trùng theo giây ở sau đó tính trên số giây của NHẦM TẤM.

`pts_time` là mốc gốc trong map-keyframes, không qua phép làm tròn nào, nên
mỗi keyframe một giá trị riêng. Đó mới là danh tính thật của một tấm ảnh.

Khoá vẫn làm tròn tới 3 chữ số thập phân: pts_time là float64, hai nhánh có
thể trả cùng một tấm qua hai đường đọc khác nhau (parquet với sqlite) và lệch
nhau ở chữ số cuối. Làm tròn 3 chữ số ≈ 1 mili giây, nhỏ hơn mọi khoảng cách
keyframe thật (2,6-3,5 giây) hàng nghìn lần nên không thể gộp nhầm hai tấm.

Mục vào nào KHÔNG có `pts_time` thì lùi về khoá frame_idx cũ và bị đếm vào
`CANH_BAO_THIEU_PTS`. Hai loại khoá KHÔNG bao giờ gộp với nhau (tiền tố "t"
với "f") — thà tách đôi một tấm còn hơn gộp nhầm hai tấm.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

# Số chữ số thập phân giữ lại khi làm khoá. 3 = mili giây.
SO_LE_PTS = 3

# Đếm số mục vào thiếu pts_time, theo tên nhánh. Xoá lại ở mỗi lần gọi.
CANH_BAO_THIEU_PTS: dict[str, int] = {}


def khoa_gop(item: Mapping[str, Any]) -> tuple:
    """Danh tính của MỘT tấm ảnh khi gộp.

    ("t", video_id, pts_time làm tròn)  — đường đúng
    ("f", video_id, frame_idx)          — chỉ khi thiếu pts_time
    """
    pts = item.get("pts_time")
    if pts is not None:
        try:
            return ("t", str(item["video_id"]), round(float(pts), SO_LE_PTS))
        except (TypeError, ValueError):
            pass
    return ("f", str(item["video_id"]), int(item["frame_idx"]))


def reciprocal_rank_fusion(
    modality_results: Mapping[str, Sequence[dict[str, Any]]],
    weights: Mapping[str, float] | None = None,
    k_rrf: int = 60,
) -> list[dict[str, Any]]:
    """
    RRF Score(d) = sum_{m in M} (w_m / (k_rrf + rank_m(d))).

    Mỗi phần tử vào cần `video_id` và `pts_time` (nên có thêm `frame_idx`, `n`).
    Phần tử ra có thêm `ranks` — hạng của nó trong từng nhánh, dùng để biết
    tấm nào lên nhờ nguồn nào.

    KHỬ TRÙNG KHOÁ TRONG TỪNG NHÁNH
    -------------------------------
    Cộng thẳng `rrf_scores[key] += ...` là sai khi MỘT nhánh trả cùng một tấm
    nhiều lần — chuyện có thật ở nhánh ASR, nơi hai câu nói liền nhau cùng nở
    ra một keyframe. Tấm đó được cộng điểm nhiều lần cho cùng một nhánh rồi
    nhảy lên trên tấm có hạng cao hơn, tức RRF tự đảo thứ tự dù nhánh khác
    không đóng góp gì. Đo trên bộ dev 32 câu: 6 câu bị đảo.

    Công thức RRF chuẩn tính MỘT hạng cho mỗi tài liệu trong mỗi nhánh, nên ở
    đây khử trùng trước, giữ hạng nhỏ nhất, rồi mới cộng.
    """
    weights = weights or {}
    CANH_BAO_THIEU_PTS.clear()

    rrf_scores: dict[tuple, float] = defaultdict(float)
    provenance: dict[tuple, dict[str, int]] = defaultdict(dict)
    dai_dien: dict[tuple, dict[str, Any]] = {}

    for mod_name, results in modality_results.items():
        w = float(weights.get(mod_name, 1.0))
        thieu = 0

        hang_tot_nhat: dict[tuple, int] = {}
        for rank_idx, item in enumerate(results, start=1):
            khoa = khoa_gop(item)
            if khoa[0] == "f":
                thieu += 1
            if khoa not in hang_tot_nhat:
                hang_tot_nhat[khoa] = rank_idx
            dai_dien.setdefault(khoa, dict(item))

        if thieu:
            CANH_BAO_THIEU_PTS[mod_name] = thieu

        for khoa, rank_idx in hang_tot_nhat.items():
            rrf_scores[khoa] += w / (k_rrf + rank_idx)
            provenance[khoa][mod_name] = rank_idx

    fused: list[dict[str, Any]] = []
    for khoa, score in rrf_scores.items():
        goc = dai_dien.get(khoa, {})
        muc: dict[str, Any] = {
            "video_id": khoa[1],
            "score": float(score),
            "ranks": provenance[khoa],
        }
        # Giữ nguyên các trường định danh của tấm đại diện. `n` và `pts_time`
        # là bắt buộc cho bước lọc trùng theo giây và bước tra ảnh ở sau.
        for truong in ("n", "frame_idx", "pts_time"):
            if goc.get(truong) is not None:
                muc[truong] = goc[truong]
        if "frame_idx" not in muc and khoa[0] == "f":
            muc["frame_idx"] = khoa[2]
        fused.append(muc)

    # Khoá phụ là danh tính tấm ảnh: hai tấm cùng điểm luôn ra cùng một thứ tự
    # ở mọi lần chạy. Không có khoá phụ thì thứ tự phụ thuộc thứ tự dict và
    # phép đo A/B sẽ dao động vì lý do không liên quan gì tới chất lượng.
    fused.sort(
        key=lambda x: (
            -x["score"],
            str(x["video_id"]),
            float(x.get("pts_time") or 0.0),
        )
    )
    return fused
