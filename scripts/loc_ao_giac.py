"""
Việc 15 (bước dọn) — Lọc câu ẢO GIÁC khỏi derived/asr/.

VẤN ĐỀ: phần lớn dữ liệu tiếng Việt của Whisper là phụ đề YouTube. Gặp đoạn
nhạc nền hoặc im lặng, mô hình không im mà rơi về câu quen nhất nó từng thấy:

    "Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn"

Câu này KHÔNG có trong video. Nó có mốc thời gian đàng hoàng, chữ đúng chính
tả, nhìn không khác gì dữ liệu thật — nên không bao giờ tự lộ ra.

VÌ SAO PHẢI LỌC dù chỉ vài dòng: nó chứa 'subscribe', 'kênh', 'video' — từ
hiếm trong kho tin tức, đúng loại từ mà chiến lược tra cứu của nhóm nhắm vào.
BM25 cho điểm cao cho từ hiếm, nên một dòng rác kiểu này gây hại hơn một trăm
dòng chữ sai chính tả thông thường.

BÀI HỌC ĐÃ TRẢ GIÁ (đọc trước khi thêm mẫu mới):
Bản đầu của tệp này chặn theo cụm 'subscribe cho kenh'. Nó bắt đúng 45 dòng
bịa — và bắt oan MỘT dòng thật:

    541.8s  "Và hãy nhớ ủng hộ cho chương trình bằng cách like, share và
             subscribe cho kênh youtube Việt Nam Đi Là Ghiền và HTV giải trí"

Đó là lời dẫn cuối chương trình, MC nói thật, đã nghe tay xác nhận. Kho có
chương trình giải trí truyền hình, và họ NÓI THẬT những câu rao kênh.

Nhìn lại 45 dòng bịa thì cả 45 đều kết thúc bằng đúng một cụm:
'Để không bỏ lỡ những video hấp dẫn'. Tên kênh thì đổi (Ghiền Mì Gõ, La La
School), phần đuôi thì không bao giờ đổi — vì đó mới là phần Whisper học
thuộc lòng. Chặn phần đuôi thì bắt hết 45 dòng bịa và không đụng dòng thật nào.

VÌ VẬY danh sách chia hai bậc, và chỉ bậc 1 mới bị xoá tự động.

CHỖ NÀY LÀ DUY NHẤT giữ danh sách câu chặn. Cố ý không nhét vào asr.py: bốn
máy chạy xong ở bốn thời điểm khác nhau, sửa danh sách trong asr.py thì máy
chạy trước và máy chạy sau ra kết quả khác nhau. Chạy lệnh này sau khi ASR
xong thì cả bốn phần đi qua cùng một danh sách.

Cách chạy (PowerShell):
    python -m scripts.loc_ao_giac              # CHỈ BÁO CÁO, không sửa gì
    python -m scripts.loc_ao_giac --xoa        # xoá thật, sau khi đã đọc báo cáo

Mặc định KHÔNG sửa gì. Đọc báo cáo trước rồi mới chạy --xoa.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

try:
    from src.aic2026.paths import ASR_DIR
except ImportError as e:
    print("Không import được aic2026. Đã chạy 'pip install -e .' chưa?")
    print(f"Chi tiết: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# DANH SÁCH HAI BẬC
# ---------------------------------------------------------------------------
# So khớp KHÔNG dấu, KHÔNG phân biệt hoa thường, theo kiểu chứa-chuỗi.

# BẬC 1 — XOÁ TỰ ĐỘNG.
# Chỉ những chuỗi Whisper nhả ra NGUYÊN XI, không bao giờ đổi một chữ. Đây là
# văn bản mô hình học thuộc, không phải chủ đề mô hình hay nói tới. Thêm mẫu
# vào đây thì phải chắc con người KHÔNG BAO GIỜ nói y hệt vậy trên truyền hình.
CHAN_CHAC = [
    "de khong bo lo nhung video hap dan",
]

# BẬC 2 — CHỈ BÁO CÁO, KHÔNG BAO GIỜ TỰ XOÁ.
# Đáng ngờ nhưng người thật CÓ nói. Máy không phân biệt được, chỉ có tai người
# nghe đúng mốc thời gian mới biết. In ra để người chạy tự quyết định.
#
# CỐ Ý KHÔNG có "cảm ơn quý vị đã theo dõi" hay "xin chào quý vị và các bạn":
# bản tin nào cũng nói, đưa vào chỉ tổ làm báo cáo dài ra vô ích.
CAN_XEM = [
    "ghien mi go",
    "subscribe",
    "dang ky kenh",
    "dang ky cho kenh",
    "bam chuong thong bao",
    "an chuong thong bao",
    "ung ho kenh",
    "hen gap lai cac ban trong video",
]


def bo_dau(chu: str) -> str:
    """Bỏ dấu tiếng Việt và hạ chữ thường, để so khớp không phụ thuộc chính tả.

    PHẢI xử lý 'đ' RIÊNG. 'đ' không phải 'd' có dấu — nó là một ký tự độc lập
    trong Unicode (U+0111), NFD không tách được. Quên dòng thay 'đ'->'d' thì
    mọi mẫu bắt đầu bằng 'de'/'dang' đều trượt lặng lẽ: không báo lỗi, chỉ
    không khớp gì cả. Đã dính bẫy này một lần khi thử.
    """
    tach = unicodedata.normalize("NFD", chu.lower())
    khong_dau = "".join(c for c in tach if unicodedata.category(c) != "Mn")
    return khong_dau.replace("\u0111", "d")


def xep_bac(text: str) -> tuple[int, str] | None:
    """Trả về (bậc, mẫu khớp), hoặc None nếu sạch.

    Bậc 1 thắng bậc 2: dòng bịa nguyên xi cũng chứa chữ 'subscribe', nhưng nó
    là bịa chắc chắn chứ không phải đáng ngờ.
    """
    khong_dau = bo_dau(text)
    for mau in CHAN_CHAC:
        if mau in khong_dau:
            return 1, mau
    for mau in CAN_XEM:
        if mau in khong_dau:
            return 2, mau
    return None


def doc_dong(duong_dan: Path) -> list[dict]:
    ket_qua = []
    with duong_dan.open("r", encoding="utf-8") as f:
        for dong in f:
            if dong.strip():
                ket_qua.append(json.loads(dong))
    return ket_qua


def ghi_dong(duong_dan: Path, cac_dong: list[dict]) -> None:
    """Ghi ra .tmp rồi đổi tên — không để lại tệp ghi dở."""
    tmp = duong_dan.with_suffix(".jsonl.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for dong in cac_dong:
                f.write(json.dumps(dong, ensure_ascii=False) + "\n")
        tmp.replace(duong_dan)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main():
    p = argparse.ArgumentParser(description="Lọc câu ảo giác YouTube khỏi derived/asr/")
    p.add_argument("--xoa", action="store_true",
                   help="Xoá thật. Không có cờ này thì chỉ báo cáo.")
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    if not ASR_DIR.exists():
        print(f"Không có {ASR_DIR}")
        sys.exit(1)

    cac_tep = [x for x in sorted(ASR_DIR.glob("*.jsonl")) if not x.name.startswith("_")]
    if not cac_tep:
        print(f"Không có tệp .jsonl nào trong {ASR_DIR}")
        sys.exit(1)

    print("=" * 70)
    print("DÒ CÂU ẢO GIÁC" + ("  —  CHẾ ĐỘ XOÁ THẬT" if args.xoa else "  —  chỉ báo cáo"))
    print("=" * 70)
    print(f"Thư mục : {ASR_DIR}")
    print(f"Số tệp  : {len(cac_tep)}\n")

    tong_dong = 0
    so_bac1 = 0
    so_bac2 = 0
    can_xem: list[tuple[str, float, str, str]] = []   # video, start, mẫu, chữ

    for tep in cac_tep:
        cac_dong = doc_dong(tep)
        tong_dong += len(cac_dong)

        giu: list[dict] = []
        xoa: list[tuple[str, str]] = []
        for ban_ghi in cac_dong:
            ket = xep_bac(ban_ghi.get("text") or "")
            if ket and ket[0] == 1:
                xoa.append((ket[1], ban_ghi["text"]))
            else:
                giu.append(ban_ghi)
                if ket:
                    can_xem.append((tep.stem, float(ban_ghi.get("start", 0.0)),
                                    ket[1], ban_ghi["text"]))

        if not xoa:
            continue

        so_bac1 += len(xoa)
        video_id = tep.stem
        print(f"{video_id}  —  {len(xoa)} dòng, còn lại {len(giu)}/{len(cac_dong)}")
        for mau, chu in xoa[:3]:
            print(f"    [{mau}]  {chu[:72]}")
        if len(xoa) > 3:
            print(f"    ... và {len(xoa) - 3} dòng nữa")

        # Video mất SẠCH đoạn có chữ: nhiều khả năng chỉ có nhạc nền, và toàn
        # bộ "lời nói" của nó là mô hình bịa ra.
        con_co_chu = [d for d in giu if (d.get("text") or "").strip()]
        if not con_co_chu:
            print("    -> sau khi lọc, video này KHÔNG còn đoạn nào có chữ")

        if args.xoa:
            if con_co_chu:
                ghi_dong(tep, giu)
            else:
                # Giữ đúng quy ước: tệp rỗng thì người đọc sau không phân biệt
                # được "chưa chạy" với "chạy rồi mà không có gì".
                mau_dong = cac_dong[0]
                ghi_dong(tep, [{
                    "video_id": video_id,
                    "start": 0.0, "end": 0.0, "text": "",
                    "frame_idx_start": 0, "frame_idx_end": 0,
                    "lang": mau_dong.get("lang", "vi"),
                    "engine": mau_dong.get("engine", ""),
                }])
            with (ASR_DIR / "_nhat_ky.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "video_id": video_id,
                    "trang_thai": "loc_ao_giac",
                    "so_dong_da_xoa": len(xoa),
                    "so_doan_con_lai": len(con_co_chu),
                    "cac_mau": sorted({m for m, _ in xoa}),
                }, ensure_ascii=False) + "\n") 

    so_bac2 = len(can_xem)

    # --- Bậc 2: chỉ in ra, KHÔNG đụng tới ------------------------------------
    if can_xem:
        print("\n" + "-" * 70)
        print("BẬC 2 — ĐÁNG NGỜ, KHÔNG TỰ XOÁ. Tự nghe rồi tự quyết.")
        print("-" * 70)
        print("Người thật CÓ nói những câu này. Nghe đúng mốc thời gian mới biết.\n")
        for video_id, giay, mau, chu in can_xem:
            print(f"{video_id}  {giay:8.1f}s  [{mau}]")
            print(f"    {chu[:78]}")
        print("\nNghe thử một dòng (thay video và số giây):")
        print("    ffmpeg -y -v error -ss <giây-3> -t 20 -i "
              "D:\\aic-data\\derived\\audio\\<video>.wav %TEMP%\\kiem.wav")
        print("    start %TEMP%\\kiem.wav")
        print("\nNGHE RA lời nói thật -> để nguyên, đó là dữ liệu đúng.")
        print("KHÔNG nghe thấy gì  -> xoá tay đúng dòng đó.")

    print("\n" + "=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)
    print(f"Tổng số đoạn đã dò      : {tong_dong}")
    print(f"Bậc 1 — bịa chắc chắn   : {so_bac1}"
          + ("  (ĐÃ XOÁ)" if args.xoa and so_bac1 else ""))
    print(f"Bậc 2 — cần tự nghe     : {so_bac2}  (không đụng tới)")

    if not so_bac1:
        print("\nKhông có dòng bậc 1 nào. Không cần chạy --xoa.")
    elif args.xoa:
        print("\nĐã xoá bậc 1, ghi vào _nhat_ky.jsonl với trạng thái 'loc_ao_giac'.")
        print("Nhớ nói với Nguyên là phần này đã lọc, để bốn phần cùng chuẩn.")
    else:
        print("\nChưa sửa gì. Đọc lại danh sách bậc 1 ở trên, thấy đúng thì chạy:")
        print("    python -m scripts.loc_ao_giac --xoa")


if __name__ == "__main__":
    main()