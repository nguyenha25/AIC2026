"""
Việc 15 (bước cuối) — Nạp derived/asr/ vào CÙNG kho tra cứu chữ của Việc 9.

Cùng một tệp index/fts/text.sqlite, nhưng BẢNG RIÊNG `asr_fts`.
Lý do vì sao tách bảng: xem đầu tệp src/aic2026/index/fts_index.py.

Điều quan trọng nhất cần nhớ: lệnh này KHÔNG đụng tới bảng `ocr_fts`.
Nghi chạy lại OCR cũng không xoá mất ASR, và ngược lại.

Cách chạy (PowerShell):
    python -m scripts.nap_asr_vao_fts                    # nạp rồi in thống kê
    python -m scripts.nap_asr_vao_fts --kiem "Hà Nội"    # nạp xong thử một câu
    python -m scripts.nap_asr_vao_fts --chi-thong-ke     # không nạp, chỉ xem đang có gì
"""

from __future__ import annotations

import argparse
import sys

try:
    from src.aic2026.paths import ASR_DIR, FTS_DIR
    from src.aic2026.index.fts_index import TextSearchIndex
except ImportError as e:
    print("Không import được aic2026. Đã chạy 'pip install -e .' chưa?")
    print(f"Chi tiết: {e}")
    sys.exit(1)

# Câu thử mặc định: có chữ hai dấu (ề, ệ) để bắt lỗi remove_diacritics.
# Mặc định của SQLite là remove_diacritics 1, nó làm hỏng ề/ậ/ữ và mất ~95%
# kết quả cho câu gõ ĐÚNG chính tả. Bảng asr_fts đặt remove_diacritics 2.
CAU_THU_MAC_DINH = "Việt Nam"


def main():
    p = argparse.ArgumentParser(description="Nạp derived/asr/ vào index/fts/text.sqlite")
    p.add_argument("--kiem", default=None, metavar="CÂU",
                   help="Chạy thử một câu tra cứu sau khi nạp xong.")
    p.add_argument("--chi-thong-ke", action="store_true",
                   help="Không nạp lại, chỉ in xem trong kho đang có gì.")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    duong_dan_db = FTS_DIR / "text.sqlite"
    kho = TextSearchIndex(db_path=duong_dan_db)

    print("=" * 66)
    print("NẠP LỜI NÓI VÀO KHO TRA CỨU CHỮ")
    print("=" * 66)
    print(f"Nguồn : {ASR_DIR}")
    print(f"Kho   : {duong_dan_db}")
    print()

    if not args.chi_thong_ke:
        if not ASR_DIR.exists():
            print(f"Không có thư mục {ASR_DIR}. Chạy Việc 15 trước:")
            print("    python -m scripts.run_asr_batch")
            sys.exit(1)

        so_tep_co = len([x for x in ASR_DIR.glob("*.jsonl") if not x.name.startswith("_")])
        if so_tep_co == 0:
            print(f"Thư mục {ASR_DIR} chưa có tệp .jsonl nào. Chạy Việc 15 trước.")
            sys.exit(1)

        print(f"Đang nạp {so_tep_co} tệp ...", flush=True)
        try:
            kq = kho.build_asr_index_from_jsonl_dir(ASR_DIR)
        except ValueError as e:
            # Dữ liệu sai mẫu: dừng hẳn, không nạp nửa vời rồi để người sau đoán
            print(f"\nDỪNG — dữ liệu không đúng mẫu:\n   {e}")
            print("\nKho KHÔNG bị đụng tới: cả lượt nạp nằm trong một giao dịch,")
            print("lỗi giữa chừng thì SQLite trả bảng về đúng như trước khi chạy.")
            print("Sửa tệp sai rồi chạy lại lệnh này.")
            sys.exit(1)

        print(f"   tệp đọc            : {kq['so_tep']}")
        print(f"   dòng đọc được      : {kq['so_dong_doc']}")
        print(f"   dòng nạp vào kho   : {kq['so_dong_nap']}")
        bo_qua = kq["so_dong_doc"] - kq["so_dong_nap"]
        if bo_qua:
            print(f"   bỏ qua (text rỗng) : {bo_qua}  — video không có lời nói, bình thường")
        print()

    tk = kho.thong_ke()
    print("-" * 66)
    print("ĐANG CÓ TRONG text.sqlite")
    print("-" * 66)
    print(f"  ocr_fts : {tk['so_ban_ghi_ocr']:>8} bản ghi / {tk['so_video_ocr']:>4} video")
    print(f"  asr_fts : {tk['so_ban_ghi_asr']:>8} bản ghi / {tk['so_video_asr']:>4} video")
    if tk["so_ban_ghi_ocr"] == 0:
        print("\n  (ocr_fts rỗng — máy này chưa chạy bước nạp OCR của Việc 9.")
        print("   KHÔNG phải lỗi của lệnh này: hai bảng độc lập nhau.)")
    print()

    cau = args.kiem or CAU_THU_MAC_DINH
    print("-" * 66)
    print(f'THỬ TRA CỨU: "{cau}"')
    print("-" * 66)
    ket_qua = kho.search_asr(cau, top_k=args.top_k)
    if not ket_qua:
        print("Không có kết quả.")
        print("Nếu câu này CHẮC CHẮN có trong lời nói thì kiểm hai điều:")
        print("  1. bảng asr_fts có đúng tokenize 'unicode61 remove_diacritics 2' không")
        print("  2. câu gõ có đúng dấu như trong tệp .jsonl không")
    else:
        for i, r in enumerate(ket_qua, start=1):
            chu = r["text"][:80] + ("..." if len(r["text"]) > 80 else "")
            print(f"{i}. {r['video_id']}  {r['start']:.2f}s–{r['end']:.2f}s  "
                  f"(khung {r['frame_idx_start']}–{r['frame_idx_end']})")
            print(f"   {chu}")
    print("=" * 66)
    print("Nhắc: frame_idx_start là khung hình BẮT ĐẦU CÂU NÓI, không phải tấm ảnh")
    print("có trong kho. Việc 10 phải đối chiếu khoảng [start, end] với frame_map")
    print("để lấy ra các tấm ảnh nằm trong khoảng, rồi mới đưa vào RRF.")


if __name__ == "__main__":
    main()
