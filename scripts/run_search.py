"""
VIỆC 4 — chạy mạch tìm kiếm đầu–cuối và ghi tệp nộp.

    python -m scripts.run_search
    python -m scripts.run_search --cau "..." --id demo
    python -m scripts.run_search --tep D:/aic-data/dev/dev_questions.jsonl
    python -m scripts.run_search --khong-ghi

Mỗi lần chạy sinh một thư mục trong <gốc dữ liệu>/runs/ chứa:

    params.json
    settings.yaml
    ket_qua.jsonl
    top10.jsonl

MÃ THOÁT
    0   xong, mọi câu ĐẠT cổng thoát số 5
    1   có câu lỗi / TRƯỢT cổng 5 / không chạy được câu nào
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Phải đặt trước khi import torch/faiss trên Windows.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from src.aic2026.paths import RUNS_DIR, SUBMISSIONS_DIR
from src.aic2026.rank.config import SETTINGS_PATH, tham_so_da_dung
from src.aic2026.rank.search import run_query
from src.aic2026.submit import KIS, QA, TRAKE


# ---------------------------------------------------------------------------
# Năm câu thử mặc định
# ---------------------------------------------------------------------------

CAU_THU_MAC_DINH = [
    {
        "query_id": "thu-1",
        "task": KIS,
        "text": (
            "Người dẫn chương trình thời sự ngồi trong trường quay, "
            "phía sau là màn hình lớn."
        ),
        "text_en": (
            "a television news anchor sitting in a studio with a large "
            "screen behind them"
        ),
    },
    {
        "query_id": "thu-2",
        "task": KIS,
        "text": (
            "Cảnh đường phố đông xe máy giờ tan tầm ở thành phố Việt Nam."
        ),
        "text_en": (
            "a crowded city street full of motorbikes during rush hour "
            "in Vietnam"
        ),
    },
    {
        "query_id": "thu-3",
        "task": KIS,
        "text": (
            "Ruộng lúa xanh chụp từ trên cao, có nông dân đang làm việc."
        ),
        "text_en": (
            "aerial view of green rice fields with farmers working"
        ),
    },
    {
        "query_id": "thu-4",
        "task": KIS,
        "text": (
            "Một người phát biểu sau bục có micro trong hội nghị."
        ),
        "text_en": (
            "a person speaking at a podium with microphones at a "
            "conference"
        ),
    },
    {
        "query_id": "thu-5",
        "task": KIS,
        "text": (
            "Cảnh ngập lụt, nước dâng cao ngoài đường, người dân lội nước."
        ),
        "text_en": (
            "a flooded street with high water and people wading through it"
        ),
    },
]


# ---------------------------------------------------------------------------
# Chuẩn hóa loại truy vấn
# ---------------------------------------------------------------------------

def map_loai_truy_van(loai_truy_van: object) -> str:
    """
    Chuyển loai_truy_van của dev_questions.jsonl sang task lõi.

    rank/search.py chỉ nhận:
        kis
        qa
        trake

    Không tự động đoán loại chưa biết thành KIS.
    """

    if loai_truy_van is None:
        return ""

    value = str(loai_truy_van).strip().lower()

    # Đã đúng schema lõi.
    if value in {KIS, QA, TRAKE}:
        return value

    mapping = {
        # Mô tả / tìm ảnh.
        "mo_ta": KIS,
        "mô_tả": KIS,
        "mo-ta": KIS,
        "mô-tả": KIS,
        "tim_anh": KIS,
        "tìm_ảnh": KIS,
        "tim-anh": KIS,
        "tìm-ảnh": KIS,

        # Hỏi đáp.
        "qa": QA,
        "hoi_dap": QA,
        "hỏi_đáp": QA,
        "hoi-dap": QA,
        "hỏi-đáp": QA,

        # Chuỗi sự kiện.
        "chuoi_su_kien": TRAKE,
        "chuỗi_sự_kiện": TRAKE,
        "chuoi-su-kien": TRAKE,
        "chuỗi-sự-kiện": TRAKE,
    }

    return mapping.get(value, "")


# ---------------------------------------------------------------------------
# Chuẩn hóa một record JSONL
# ---------------------------------------------------------------------------

def chuan_hoa_cau_hoi(muc: dict, index: int) -> dict:
    """
    Hỗ trợ cả:

    Schema chuẩn:
        query_id
        task
        text / text_en
        gt_answer
        so_moc_trake

    Schema dev_questions.jsonl:
        id
        loai_truy_van
        cau_hoi
        ...
    """

    if not isinstance(muc, dict):
        raise ValueError(
            f"Câu {index}: record phải là JSON object."
        )

    # ID
    query_id = str(
        muc.get("query_id")
        or muc.get("id")
        or f"q{index}"
    ).strip()

    # Nội dung câu hỏi.
    # Ưu tiên schema chuẩn, sau đó fallback sang cau_hoi.
    cau_hoi = (
        muc.get("text_en")
        or muc.get("text")
        or muc.get("cau_hoi")
        or ""
    )

    cau_hoi = str(cau_hoi).strip()

    if not cau_hoi:
        raise ValueError(
            f"Câu {query_id}: câu hỏi rỗng. "
            f"Đã thử text_en, text và cau_hoi."
        )

    # Task.
    task_raw = muc.get("task")

    if task_raw:
        task = str(task_raw).strip().lower()
    else:
        task = map_loai_truy_van(
            muc.get("loai_truy_van")
        )

    if task not in (KIS, QA, TRAKE):
        raise ValueError(
            f"Câu {query_id}: loại truy vấn "
            f"'{muc.get('loai_truy_van')}' chưa được "
            f"map sang KIS/QA/TRAKE."
        )

    # Đáp án QA.
    gt_answer = (
        muc.get("gt_answer")
        or muc.get("dap_an")
        or muc.get("đáp_án")
        or muc.get("cau_tra_loi")
    )

    # Số mốc TRAKE.
    so_moc_raw = (
        muc.get("so_moc_trake")
        or muc.get("so_moc")
        or 4
    )

    try:
        so_moc_trake = int(so_moc_raw)
    except (TypeError, ValueError):
        so_moc_trake = 4

    return {
        "query_id": query_id,
        "task": task,
        "text": cau_hoi,
        "text_en": muc.get("text_en"),
        "gt_answer": gt_answer,
        "so_moc_trake": so_moc_trake,
    }


# ---------------------------------------------------------------------------
# Đọc JSONL
# ---------------------------------------------------------------------------

def doc_tep_cau_hoi(duong_dan: Path) -> list[dict]:
    """Đọc JSONL và chuẩn hóa về schema mà rank.search.py cần."""

    if not duong_dan.exists():
        raise FileNotFoundError(
            f"Không thấy tệp câu hỏi: {duong_dan}"
        )

    cau_hoi: list[dict] = []

    with duong_dan.open("r", encoding="utf-8") as f:
        for so_dong, dong in enumerate(f, start=1):
            dong = dong.strip()

            if not dong:
                continue

            try:
                muc = json.loads(dong)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{duong_dan.name} dòng {so_dong} "
                    f"không phải JSON hợp lệ."
                ) from exc

            try:
                cau_hoi.append(
                    chuan_hoa_cau_hoi(
                        muc,
                        len(cau_hoi) + 1,
                    )
                )
            except Exception as exc:
                # Giữ lại lỗi của riêng câu đó để các câu khác vẫn chạy.
                query_id = str(
                    muc.get("query_id")
                    or muc.get("id")
                    or f"q{len(cau_hoi) + 1}"
                )

                cau_hoi.append(
                    {
                        "__normalize_error__": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "query_id": query_id,
                    }
                )

    if not cau_hoi:
        raise ValueError(
            f"{duong_dan.name} không có dòng nào."
        )

    return cau_hoi


# ---------------------------------------------------------------------------
# Ghi log lần chạy
# ---------------------------------------------------------------------------

def tao_thu_muc_lan_chay(
    nhan: str | None = None,
) -> Path:

    ten = (
        nhan
        or f"{datetime.now():%Y-%m-%d_%H%M}_clipb32"
    )

    thu_muc = RUNS_DIR / ten
    thu_muc.mkdir(
        parents=True,
        exist_ok=True,
    )

    return thu_muc


def ghi_lan_chay(
    thu_muc: Path,
    ket_qua: list,
) -> None:

    (thu_muc / "params.json").write_text(
        json.dumps(
            tham_so_da_dung(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if SETTINGS_PATH.exists():
        shutil.copy2(
            SETTINGS_PATH,
            thu_muc / "settings.yaml",
        )

    with (
        thu_muc / "ket_qua.jsonl"
    ).open("w", encoding="utf-8") as f:

        for kq in ket_qua:
            f.write(
                json.dumps(
                    {
                        "query_id": kq.query_id,
                        "task": kq.task,
                        "text": kq.cau_hoi,
                        "so_ung_vien": kq.so_ung_vien,
                        "so_sau_loc_trung": (
                            kq.bao_cao_loc_trung.ra
                        ),
                        "so_dong_nop": kq.so_dong_nop,
                        "dat_cong_5": kq.dat_cong_5,
                        "thoi_gian_ms": {
                            k: round(v, 1)
                            for k, v in kq.thoi_gian_ms.items()
                        },
                        "tep_nop": (
                            str(kq.duong_dan_nop)
                            if kq.duong_dan_nop
                            else None
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with (
        thu_muc / "top10.jsonl"
    ).open("w", encoding="utf-8") as f:

        for kq in ket_qua:
            for hang, hit in enumerate(
                kq.ket_qua[:10],
                start=1,
            ):
                f.write(
                    json.dumps(
                        {
                            "query_id": kq.query_id,
                            "hang": hang,
                            "video_id": hit.video_id,
                            "n": hit.n,
                            "frame_idx": hit.frame_idx,
                            "pts_time": round(
                                float(hit.pts_time),
                                3,
                            ),
                            "score": round(
                                float(hit.score),
                                4,
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Việc 4 — mạch tìm kiếm đầu–cuối, "
            "ghi tệp nộp."
        )
    )

    parser.add_argument(
        "--cau",
        help="Chạy đúng một câu truy vấn.",
    )

    parser.add_argument(
        "--id",
        default="demo",
        help="Mã truy vấn.",
    )

    parser.add_argument(
        "--task",
        default=KIS,
        choices=[KIS, QA, TRAKE],
    )

    parser.add_argument(
        "--dap-an",
        dest="dap_an",
        help="Đáp án khi task=qa.",
    )

    parser.add_argument(
        "--tep",
        help="Tệp JSONL chứa nhiều câu.",
    )

    parser.add_argument(
        "--nhan",
        help="Tên thư mục trong runs/.",
    )

    parser.add_argument(
        "--khong-ghi",
        action="store_true",
        help=(
            "Chạy nhưng không ghi submission "
            "và không tạo runs."
        ),
    )

    doi_so = parser.parse_args()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    if doi_so.tep:

        cau_hoi = doc_tep_cau_hoi(
            Path(doi_so.tep)
        )

        nguon = Path(
            doi_so.tep
        ).name

    elif doi_so.cau:

        cau_hoi = [
            {
                "query_id": doi_so.id,
                "task": doi_so.task,
                "text": doi_so.cau,
                "gt_answer": doi_so.dap_an,
                "so_moc_trake": 4,
            }
        ]

        nguon = "dòng lệnh"

    else:

        cau_hoi = CAU_THU_MAC_DINH
        nguon = (
            "năm câu thử sẵn "
            "trong scripts/run_search.py"
        )

    ghi_tep = not doi_so.khong_ghi

    print("=" * 72)
    print("VIỆC 4 — MẠCH TÌM KIẾM ĐẦU–CUỐI")
    print("=" * 72)

    print(
        f"Nguồn câu hỏi : {nguon} "
        f"({len(cau_hoi)} câu)"
    )

    print(
        f"Tệp nộp       : "
        f"{SUBMISSIONS_DIR if ghi_tep else '(không ghi)'}"
    )

    print()

    # ------------------------------------------------------------------
    # Chạy từng câu
    # ------------------------------------------------------------------

    ket_qua: list = []
    that_bai: list[tuple[str, str]] = []

    da_in_canh_bao = False

    for muc in cau_hoi:

        # Lỗi normalize.
        if "__normalize_error__" in muc:

            query_id = str(
                muc.get("query_id")
                or "unknown"
            )

            loi = str(
                muc["__normalize_error__"]
            )

            that_bai.append(
                (query_id, loi)
            )

            print(
                f"[{query_id}] LỖI — {loi}"
            )
            print()

            continue

        van_ban = (
            muc.get("text_en")
            or muc.get("text")
            or ""
        )

        query_id = str(
            muc.get("query_id")
            or "demo"
        )

        task = str(
            muc.get("task")
            or KIS
        ).lower()

        try:

            kq = run_query(
                cau_hoi=van_ban,
                query_id=query_id,
                task=task,
                dap_an=muc.get("gt_answer"),
                so_moc_trake=int(
                    muc.get("so_moc_trake")
                    or 4
                ),
                ghi_tep=ghi_tep,
            )

        except Exception as exc:

            that_bai.append(
                (
                    query_id,
                    f"{type(exc).__name__}: {exc}",
                )
            )

            print(
                f"[{query_id}] LỖI — "
                f"{type(exc).__name__}: {exc}"
            )

            print()
            continue

        if (
            kq.canh_bao
            and not da_in_canh_bao
        ):

            print(
                "CẢNH BÁO CẤU HÌNH:"
            )

            for dong in kq.canh_bao:
                print(f"  ! {dong}")

            print()

            da_in_canh_bao = True

        ket_qua.append(kq)

        print(
            kq.bao_cao_day_du()
        )

        print()

    # ------------------------------------------------------------------
    # Không chạy được câu nào
    # ------------------------------------------------------------------

    if not ket_qua:

        print(
            "Không chạy được câu nào."
        )

        for query_id, loi in that_bai:
            print(
                f"  {query_id}: {loi}"
            )

        return 1

    # ------------------------------------------------------------------
    # Tổng kết
    # ------------------------------------------------------------------

    truot = [
        kq
        for kq in ket_qua
        if not kq.dat_cong_5
    ]

    thieu_dong = [
        kq
        for kq in ket_qua
        if kq.so_dong_nop
        < kq.tham_so["so_dong_toi_da"]
    ]

    trung_binh = (
        sum(
            kq.thoi_gian_ms.get(
                "tong",
                0,
            )
            for kq in ket_qua
        )
        / len(ket_qua)
    )

    print("=" * 72)

    print(
        f"Chạy xong "
        f"{len(ket_qua)}/{len(cau_hoi)} câu "
        f"— trung bình "
        f"{trung_binh:.0f} ms"
    )

    if that_bai:

        print(
            f"Lỗi: {len(that_bai)} câu"
        )

        for query_id, loi in that_bai:

            print(
                f"  {query_id}: {loi}"
            )

    if thieu_dong:

        print(
            f"Chưa đủ 100 dòng: "
            f"{len(thieu_dong)} câu "
            f"("
            f"{', '.join(k.query_id for k in thieu_dong[:5])}"
            f")"
        )

    if truot:

        print(
            "CỔNG THOÁT SỐ 5: TRƯỢT — "
            f"{len(truot)} câu."
        )

    else:

        print(
            "CỔNG THOÁT SỐ 5: ĐẠT "
            "trên toàn bộ câu vừa chạy."
        )

    # ------------------------------------------------------------------
    # Ghi log
    # ------------------------------------------------------------------

    if ghi_tep:

        thu_muc = tao_thu_muc_lan_chay(
            doi_so.nhan
        )

        ghi_lan_chay(
            thu_muc,
            ket_qua,
        )

        print(
            f"Nhật ký lần chạy: {thu_muc}"
        )

    print("=" * 72)

    return (
        1
        if truot or that_bai
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())