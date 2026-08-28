"""
Việc 3b — nạp gói soi vào FiftyOne.

Dùng chung cho CẢ HAI đường: chạy tại chỗ trên Windows, và chạy trên Colab.
Notebook Colab chỉ việc gọi `nap(thu_muc)` rồi `launch()`.

    from scripts.nap_fiftyone import nap
    ds = nap("D:/aic-data/derived/goi_soi/dev_25")
    ds.launch()

Ba thứ hiện lên trong giao diện:
  * hộp vật thể của BTC, kèm điểm tin cậy
  * chữ OCR đọc được trên khung
  * hạng, điểm, nguồn (clip/ocr/asr) — để biết ứng viên này từ nhánh nào lên

Có `--dev` thì thêm trường `la_dap_an`, lọc `la_dap_an == True` là thấy ngay
khung đúng nằm ở hạng bao nhiêu.
"""

from __future__ import annotations

import json
from pathlib import Path


def nap(thu_muc, ten_bo: str | None = None):
    """Đọc gói soi -> fiftyone.Dataset đã gắn nhãn."""
    import fiftyone as fo

    goc = Path(thu_muc)
    with (goc / "du_lieu.json").open("r", encoding="utf-8") as f:
        goi = json.load(f)

    ten_bo = ten_bo or goi.get("ten") or goc.name
    if ten_bo in fo.list_datasets():
        fo.delete_dataset(ten_bo)

    ds = fo.Dataset(ten_bo, persistent=False)
    ds.info = {
        "cau_hoi": goi.get("cau_hoi") or "",
        "khoang_dap_an": goi.get("khoang_dap_an"),
    }

    mau = []
    for k in goi["khung"]:
        s = fo.Sample(filepath=str(goc / "anh" / k["anh"]))

        s["video_id"] = k["video_id"]
        s["n"] = k["n"]
        s["frame_idx"] = k["frame_idx"]        # SỐ PHẢI NỘP
        s["giay"] = round(k["pts_time"], 2)
        s["hang"] = k["hang"]
        s["diem"] = round(k["diem"], 5)
        s["nguon"] = k["nguon"]

        if k.get("vat_the"):
            s["vat_the"] = fo.Detections(
                detections=[
                    fo.Detection(
                        label=v["nhan"],
                        bounding_box=v["hop"],       # [x, y, rộng, cao]
                        confidence=v["diem"],
                    )
                    for v in k["vat_the"]
                ]
            )

        if k.get("ocr"):
            s["ocr"] = " | ".join(
                b["chu"] for b in k["ocr"] if str(b.get("chu", "")).strip()
            )
            s["ocr_conf_cao_nhat"] = max(b["conf"] for b in k["ocr"])

        if "la_dap_an" in k:
            s["la_dap_an"] = bool(k["la_dap_an"])

        mau.append(s)

    ds.add_samples(mau)

    # Sắp theo hạng để lưới hiện đúng thứ tự mạch trả về.
    ds = ds.sort_by("hang")

    print(f"Bộ {ten_bo!r}: {len(mau)} khung")
    if goi.get("cau_hoi"):
        print(f"Câu hỏi: {goi['cau_hoi'][:100]}")
    co_dap_an = [k for k in goi["khung"] if k.get("la_dap_an")]
    if goi.get("khoang_dap_an"):
        if co_dap_an:
            print(
                f"Khung ĐÚNG nằm ở hạng: {[k['hang'] for k in co_dap_an]} "
                f"(lọc la_dap_an == True để xem)"
            )
        elif goi.get("video_dap_an_co_anh") is False:
            print(
                f"KHÔNG có khung đúng trong gói — video đáp án "
                f"{goi['khoang_dap_an']['video_id']} không có ảnh trên máy đã xuất "
                "gói.\nĐây là chuyện THIẾU SHARD, KHÔNG phải mạch trượt."
            )
        else:
            print(
                "KHÔNG khung nào trong gói này rơi vào khoảng đáp án — "
                "hoặc mạch trượt hẳn, hoặc khoảng đáp án không chứa keyframe nào "
                "(chạy scripts/tran_du_lieu.py để biết)."
            )
        if goi.get("so_thieu_anh"):
            print(
                f"Lưu ý: {goi['so_thieu_anh']}/{goi.get('so_khung_yeu_cau', '?')} "
                "khung bị bỏ vì máy xuất gói không có ảnh."
            )
    return ds


def mo(ds):
    """Mở giao diện. Trên Colab tự nhúng vào ô kết quả."""
    import fiftyone as fo

    return fo.launch_app(ds)
