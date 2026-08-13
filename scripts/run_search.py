"""
VIỆC 4 — chạy mạch tìm kiếm đầu–cuối và ghi tệp nộp.

    python -m scripts.run_search                       # năm câu thử sẵn
    python -m scripts.run_search --cau "..." --id demo
    python -m scripts.run_search --tep D:/aic-data/dev/dev_questions.jsonl
    python -m scripts.run_search --khong-ghi           # chạy khô, không đụng đĩa

Mỗi lần chạy sinh một thư mục trong <gốc dữ liệu>/runs/ chứa:

    params.json     tham số đã dùng (lấy từ config/settings.yaml)
    settings.yaml   bản chép nguyên của tệp cấu hình lúc chạy
    ket_qua.jsonl   một dòng mỗi truy vấn: số ứng viên, số dòng, thời gian
    top10.jsonl     mười kết quả đầu của mỗi câu, để mắt người soát lại

Vì sao phải chép settings.yaml: hai tuần nữa nhìn lại con số nền mà không biết
lần đó chạy với cửa sổ lọc trùng bao nhiêu giây thì con số đó vô nghĩa.

MÃ THOÁT
    0   xong, mọi câu ĐẠT cổng thoát số 5
    1   có câu TRƯỢT cổng 5, hoặc không chạy được câu nào
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from src.aic2026.paths import RUNS_DIR, SUBMISSIONS_DIR
from src.aic2026.rank.config import SETTINGS_PATH, tham_so_da_dung
from src.aic2026.rank.search import run_query
from src.aic2026.submit import KIS


# ---------------------------------------------------------------------------
# Năm câu tự nghĩ — điều kiện "Xong khi" của Việc 4
# ---------------------------------------------------------------------------
#
# LƯU Ý VỀ NGÔN NGỮ — ĐỌC TRƯỚC KHI THẤY KẾT QUẢ XẤU RỒI ĐỔ CHO KHO.
#
# CLIP ViT-B/32 bản "openai" mà BTC phát học gần như toàn bộ trên chú thích
# ảnh TIẾNG ANH. Gõ tiếng Việt vào encode_text() vẫn chạy, vẫn trả về đủ 100
# kết quả, chỉ là kém hơn hẳn — đúng kiểu lỗi không có triệu chứng.
#
# Nên mỗi câu ở đây ghi cả hai bản. Mạch tìm kiếm ưu tiên `text_en`. Bản tiếng
# Việt giữ lại để người soát đọc cho nhanh, và để sau này so khi Giai đoạn 2
# đổi sang mô hình đa ngữ.
#
# Đây là chỗ cần Thi (Việc 3) và Nguyên (Việc 11) biết: bộ câu hỏi tự chấm
# đang soạn bằng tiếng Việt, nên con số nền đo được sẽ THẤP HƠN năng lực thật
# của kho nếu đưa thẳng câu tiếng Việt vào CLIP.

CAU_THU_MAC_DINH = [
    {
        "query_id": "thu-1",
        "task": KIS,
        "text": "Người dẫn chương trình thời sự ngồi trong trường quay, "
                "phía sau là màn hình lớn.",
        "text_en": "a television news anchor sitting in a studio with a large "
                   "screen behind them",
    },
    {
        "query_id": "thu-2",
        "task": KIS,
        "text": "Cảnh đường phố đông xe máy giờ tan tầm ở thành phố Việt Nam.",
        "text_en": "a crowded city street full of motorbikes during rush hour "
                   "in Vietnam",
    },
    {
        "query_id": "thu-3",
        "task": KIS,
        "text": "Ruộng lúa xanh chụp từ trên cao, có nông dân đang làm việc.",
        "text_en": "aerial view of green rice fields with farmers working",
    },
    {
        "query_id": "thu-4",
        "task": KIS,
        "text": "Một người phát biểu sau bục có micro trong hội nghị.",
        "text_en": "a person speaking at a podium with microphones at a "
                   "conference",
    },
    {
        "query_id": "thu-5",
        "task": KIS,
        "text": "Cảnh ngập lụt, nước dâng cao ngoài đường, người dân lội nước.",
        "text_en": "a flooded street with high water and people wading through it",
    },
]


# ---------------------------------------------------------------------------
# Đọc câu hỏi
# ---------------------------------------------------------------------------


def doc_tep_cau_hoi(duong_dan: Path) -> list[dict]:
    """Đọc tệp JSONL theo mẫu docs/schema/example_dev_queries.jsonl."""
    if not duong_dan.exists():
        raise FileNotFoundError(f"Không thấy tệp câu hỏi: {duong_dan}")

    cau_hoi: list[dict] = []

    with duong_dan.open("r", encoding="utf-8") as f:
        for so_dong, dong in enumerate(f, start=1):
            dong = dong.strip()
            if not dong:
                continue
            try:
                cau_hoi.append(json.loads(dong))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{duong_dan.name} dòng {so_dong} không phải JSON hợp lệ."
                ) from exc

    if not cau_hoi:
        raise ValueError(f"{duong_dan.name} không có dòng nào.")

    return cau_hoi


# ---------------------------------------------------------------------------
# Ghi thư mục lần chạy
# ---------------------------------------------------------------------------


def tao_thu_muc_lan_chay(nhan: str | None = None) -> Path:
    """runs/2026-08-15_1430_clipb32 — mỗi lần một thư mục, không ghi đè."""
    ten = nhan or f"{datetime.now():%Y-%m-%d_%H%M}_clipb32"
    thu_muc = RUNS_DIR / ten
    thu_muc.mkdir(parents=True, exist_ok=True)
    return thu_muc


def ghi_lan_chay(thu_muc: Path, ket_qua: list) -> None:
    """Ghi params.json, settings.yaml, ket_qua.jsonl, top10.jsonl."""
    (thu_muc / "params.json").write_text(
        json.dumps(tham_so_da_dung(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if SETTINGS_PATH.exists():
        shutil.copy2(SETTINGS_PATH, thu_muc / "settings.yaml")

    with (thu_muc / "ket_qua.jsonl").open("w", encoding="utf-8") as f:
        for kq in ket_qua:
            f.write(
                json.dumps(
                    {
                        "query_id": kq.query_id,
                        "task": kq.task,
                        "text": kq.cau_hoi,
                        "so_ung_vien": kq.so_ung_vien,
                        "so_sau_loc_trung": kq.bao_cao_loc_trung.ra,
                        "so_dong_nop": kq.so_dong_nop,
                        "dat_cong_5": kq.dat_cong_5,
                        "thoi_gian_ms": {
                            k: round(v, 1) for k, v in kq.thoi_gian_ms.items()
                        },
                        "tep_nop": str(kq.duong_dan_nop) if kq.duong_dan_nop else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with (thu_muc / "top10.jsonl").open("w", encoding="utf-8") as f:
        for kq in ket_qua:
            for hang, hit in enumerate(kq.ket_qua[:10], start=1):
                f.write(
                    json.dumps(
                        {
                            "query_id": kq.query_id,
                            "hang": hang,
                            "video_id": hit.video_id,
                            "n": hit.n,                    # để mở đúng tệp ảnh
                            "frame_idx": hit.frame_idx,    # số ĐI VÀO tệp nộp
                            "pts_time": round(float(hit.pts_time), 3),
                            "score": round(float(hit.score), 4),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Việc 4 — mạch tìm kiếm đầu–cuối, ghi tệp nộp."
    )
    parser.add_argument("--cau", help="Chạy đúng một câu truy vấn.")
    parser.add_argument("--id", default="demo", help="Mã truy vấn, dùng đặt tên tệp.")
    parser.add_argument("--task", default=KIS, choices=["kis", "qa", "trake"])
    parser.add_argument("--dap-an", dest="dap_an", help="Bắt buộc khi --task qa.")
    parser.add_argument("--tep", help="Tệp JSONL chứa nhiều câu.")
    parser.add_argument("--nhan", help="Tên thư mục trong runs/.")
    parser.add_argument(
        "--khong-ghi",
        action="store_true",
        help="Chạy hết mạch nhưng không ghi tệp nộp và không tạo thư mục runs/.",
    )
    doi_so = parser.parse_args()

    if doi_so.tep:
        cau_hoi = doc_tep_cau_hoi(Path(doi_so.tep))
        nguon = Path(doi_so.tep).name
    elif doi_so.cau:
        cau_hoi = [
            {
                "query_id": doi_so.id,
                "task": doi_so.task,
                "text": doi_so.cau,
                "gt_answer": doi_so.dap_an,
            }
        ]
        nguon = "dòng lệnh"
    else:
        cau_hoi = CAU_THU_MAC_DINH
        nguon = "năm câu thử sẵn trong scripts/run_search.py"

    ghi_tep = not doi_so.khong_ghi

    print("=" * 72)
    print("VIỆC 4 — MẠCH TÌM KIẾM ĐẦU–CUỐI")
    print("=" * 72)
    print(f"Nguồn câu hỏi : {nguon}  ({len(cau_hoi)} câu)")
    print(f"Tệp nộp       : {SUBMISSIONS_DIR if ghi_tep else '(không ghi)'}")
    print()

    ket_qua: list = []
    that_bai: list[tuple[str, str]] = []
    da_in_canh_bao = False

    for muc in cau_hoi:
        van_ban = muc.get("text_en") or muc.get("text") or ""
        query_id = str(muc.get("query_id") or "demo")

        try:
            kq = run_query(
                cau_hoi=van_ban,
                query_id=query_id,
                task=str(muc.get("task") or KIS).lower(),
                dap_an=muc.get("gt_answer"),
                ghi_tep=ghi_tep,
            )
        except Exception as exc:                      # noqa: BLE001
            that_bai.append((query_id, f"{type(exc).__name__}: {exc}"))
            print(f"[{query_id}] LỖI — {type(exc).__name__}: {exc}")
            print()
            continue

        # Cảnh báo cấu hình in MỘT LẦN, LÊN ĐẦU — người chạy phải thấy trước
        # khi tin vào kết quả, không phải đọc thấy ở cuối màn hình.
        if kq.canh_bao and not da_in_canh_bao:
            print("CẢNH BÁO CẤU HÌNH:")
            for dong in kq.canh_bao:
                print(f"  ! {dong}")
            print()
            da_in_canh_bao = True

        ket_qua.append(kq)
        print(kq.bao_cao_day_du())
        print()

    if not ket_qua:
        print("Không chạy được câu nào.")
        for query_id, loi in that_bai:
            print(f"  {query_id}: {loi}")
        return 1

    # --- tổng kết -----------------------------------------------------------
    truot = [kq for kq in ket_qua if not kq.dat_cong_5]
    thieu_dong = [kq for kq in ket_qua if kq.so_dong_nop < kq.tham_so["so_dong_toi_da"]]
    trung_binh = sum(kq.thoi_gian_ms.get("tong", 0) for kq in ket_qua) / len(ket_qua)

    print("=" * 72)
    print(f"Chạy xong {len(ket_qua)}/{len(cau_hoi)} câu — trung bình {trung_binh:.0f} ms")

    if that_bai:
        print(f"Lỗi: {len(that_bai)} câu")
        for query_id, loi in that_bai:
            print(f"  {query_id}: {loi}")

    if thieu_dong:
        print(
            f"Chưa đủ 100 dòng: {len(thieu_dong)} câu "
            f"({', '.join(kq.query_id for kq in thieu_dong[:5])})"
        )

    if truot:
        print(f"CỔNG THOÁT SỐ 5: TRƯỢT — {len(truot)} câu có hai dòng cùng video")
        print("  cách nhau dưới cửa sổ đã chốt. KHÔNG nộp tệp này.")
    else:
        print("CỔNG THOÁT SỐ 5: ĐẠT trên toàn bộ câu vừa chạy.")

    if ghi_tep:
        thu_muc = tao_thu_muc_lan_chay(doi_so.nhan)
        ghi_lan_chay(thu_muc, ket_qua)
        print(f"Nhật ký lần chạy: {thu_muc}")

    print("=" * 72)

    return 1 if (truot or that_bai) else 0


if __name__ == "__main__":
    sys.exit(main())