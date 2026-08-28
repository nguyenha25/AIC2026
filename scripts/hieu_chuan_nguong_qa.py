"""
Hiệu chuẩn ngưỡng chấm Q&A — Việc 6, Giai đoạn 2.

VẤN ĐỀ
------
`config/settings.yaml` đặt `cham_diem.nguong_tuong_dong_qa = 0.80`, và
`docs/decisions/nguong_cham_trake_va_qa.md` ghi thẳng rằng con số đó CHƯA qua
kiểm định thực tế và CẦN NHÓM CHẠY THỬ.

Đây là thước đo của một phần ba số câu trong bộ dev. Hỏng theo cả hai chiều,
và cả hai chiều đều IM LẶNG:

  · Ngưỡng quá CHẶT  -> đáp án đúng nhưng diễn đạt khác bị chấm sai.
                        Điểm Q&A thấp giả tạo. Nhóm đi sửa một nhánh vốn tốt.
  · Ngưỡng quá LỎNG  -> đáp án sai nghĩa được chấm đúng.
                        Điểm Q&A cao giả tạo. Nhóm tưởng mình mạnh.

Script này KHÔNG tự chọn ngưỡng. Nó bày ra bảng số để người nhìn tận mắt rồi
quyết định — vì việc "câu này đúng hay sai" là phán đoán của con người, không
uỷ quyền cho máy được.

CÁCH DÙNG
---------
1) Tạo tệp mẫu để điền:

       python -m scripts.hieu_chuan_nguong_qa --tao-mau cap_dap_an.csv

2) Điền vào tệp đó. Mỗi dòng một cặp, ba cột:

       dap_an_dung,dap_an_may_sinh,nguoi_cham
       "màu xanh lá","xanh lá cây",dung
       "5 người","năm người",dung
       "5 người","3 người",sai

   Cột `nguoi_cham` là phán đoán của NGƯỜI: `dung` hoặc `sai`. Đây là phần
   không thể tự động hoá — nó chính là chuẩn để chấm cái thước.

   Cần tối thiểu 20 cặp, trong đó ít nhất 8 cặp `sai`. Toàn cặp đúng thì
   không tìm được ngưỡng, vì không có gì để cắt.

3) Chạy hiệu chuẩn:

       python -m scripts.hieu_chuan_nguong_qa --tep cap_dap_an.csv

ĐỌC KẾT QUẢ
-----------
Script in ra các cặp đã sắp theo độ tương đồng giảm dần. Ngưỡng tốt nằm ở
KHOẢNG TRỐNG rõ ràng giữa cụm `dung` và cụm `sai`. Nếu hai cụm chồng lên nhau
thì KHÔNG có ngưỡng nào tốt — lúc đó vấn đề không phải con số, mà là mô hình
embedding không phân biệt được, và cần đổi cách chấm chứ không phải chỉnh số.

MÃ THOÁT
    0   chạy xong, đã in bảng
    1   thiếu tệp, thiếu thư viện, hoặc dữ liệu không đủ để hiệu chuẩn
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MO_HINH = "paraphrase-multilingual-MiniLM-L12-v2"
TOI_THIEU_CAP = 20
TOI_THIEU_SAI = 8

MAU = """dap_an_dung,dap_an_may_sinh,nguoi_cham
màu xanh lá,xanh lá cây,dung
5 người,năm người,dung
5 người,3 người,sai
"""


def tao_mau(duong_dan: Path) -> int:
    if duong_dan.exists():
        print(f"Tệp đã tồn tại, không ghi đè: {duong_dan}")
        return 1
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    duong_dan.write_text(MAU, encoding="utf-8", newline="\n")
    print(f"Đã tạo mẫu: {duong_dan}")
    print()
    print("Điền tối thiểu 20 cặp thật, trong đó ít nhất 8 cặp mà NGƯỜI chấm là 'sai'.")
    print("Lấy cặp thật từ đâu: chạy Việc 8 (đường ống Q&A đầu-cuối) trên bộ dev,")
    print("ghi lại đáp án máy sinh, đặt cạnh đáp án đúng trong dev_question.jsonl.")
    return 0


def doc_cap(duong_dan: Path) -> list[tuple[str, str, str]]:
    with duong_dan.open("r", encoding="utf-8-sig", newline="") as f:
        doc = list(csv.reader(f))
    if not doc:
        return []
    dau = [c.strip().lower() for c in doc[0]]
    bat_dau = 1 if dau[:2] == ["dap_an_dung", "dap_an_may_sinh"] else 0
    ra = []
    for d in doc[bat_dau:]:
        if len(d) < 3 or not d[0].strip():
            continue
        nhan = d[2].strip().lower()
        if nhan not in ("dung", "sai"):
            continue
        ra.append((d[0].strip(), d[1].strip(), nhan))
    return ra


def do_tuong_dong(cap: list[tuple[str, str, str]]) -> list[float] | None:
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        print("Thiếu sentence-transformers. Cài bằng:")
        print("    pip install sentence-transformers")
        print("(Gói này đã có trong requirements.txt — nhiều khả năng máy này chưa cài đủ.)")
        return None

    print(f"Đang nạp mô hình {MO_HINH} (lần đầu sẽ tải về, cần mạng)...")
    m = SentenceTransformer(MO_HINH)

    a = m.encode([x[0] for x in cap], convert_to_tensor=True, normalize_embeddings=True)
    b = m.encode([x[1] for x in cap], convert_to_tensor=True, normalize_embeddings=True)
    return [float(util.cos_sim(a[i], b[i])) for i in range(len(cap))]


def bao_cao(cap, diem) -> None:
    ban = sorted(zip(cap, diem), key=lambda x: -x[1])

    print()
    print("=" * 78)
    print("BẢNG ĐỘ TƯƠNG ĐỒNG — sắp giảm dần. Tìm KHOẢNG TRỐNG giữa cụm dung và cụm sai.")
    print("=" * 78)
    print(f"{'điểm':>6}  {'người chấm':<11}  {'đáp án đúng':<28}  đáp án máy sinh")
    print("-" * 78)

    # Chỉ đánh dấu LẦN chuyển đầu tiên. Đánh dấu mọi lần chuyển sẽ rải vạch
    # khắp bảng đúng lúc hai cụm chồng nhau — tức là đúng lúc KHÔNG có ranh giới.
    truoc_nhan = None
    da_danh_dau = False
    for (dung, may, nhan), d in ban:
        vach = ""
        if truoc_nhan == "dung" and nhan == "sai" and not da_danh_dau:
            vach = "   <-- cụm sai bắt đầu từ đây"
            da_danh_dau = True
        print(f"{d:>6.3f}  {nhan:<11}  {dung[:28]:<28}  {may[:26]}{vach}")
        truoc_nhan = nhan

    diem_dung = [d for (c, d) in zip(cap, diem) if c[2] == "dung"]
    diem_sai = [d for (c, d) in zip(cap, diem) if c[2] == "sai"]

    print()
    print("-" * 78)
    print(f"Cụm ĐÚNG ({len(diem_dung)} cặp): thấp nhất {min(diem_dung):.3f} · cao nhất {max(diem_dung):.3f}")
    print(f"Cụm SAI  ({len(diem_sai)} cặp): thấp nhất {min(diem_sai):.3f} · cao nhất {max(diem_sai):.3f}")
    print()

    thap_nhat_dung = min(diem_dung)
    cao_nhat_sai = max(diem_sai)

    if thap_nhat_dung > cao_nhat_sai:
        de_xuat = (thap_nhat_dung + cao_nhat_sai) / 2
        print(f"HAI CỤM TÁCH RỜI — khoảng trống rộng {thap_nhat_dung - cao_nhat_sai:.3f}")
        print(f"Ngưỡng đề xuất: {de_xuat:.2f}   (đặt vào giữa khoảng trống)")
        print()
        print("Sửa trong config/settings.yaml:")
        print(f"    cham_diem.nguong_tuong_dong_qa: {de_xuat:.2f}")
    else:
        chong = cao_nhat_sai - thap_nhat_dung
        print(f"HAI CỤM CHỒNG NHAU {chong:.3f} — KHÔNG có ngưỡng nào tách sạch được.")
        print()
        print("Nghĩa là vấn đề KHÔNG nằm ở con số ngưỡng. Mô hình embedding không")
        print("phân biệt được các cặp này. Ba đường đi:")
        print("  1. Nhìn các cặp nằm trong vùng chồng — có thể vài cặp bị người chấm nhầm.")
        print("  2. Chuẩn hoá đáp án trước khi so (bỏ dấu câu, viết thường, đổi số sang chữ).")
        print("  3. Chấp nhận và ghi rõ sai số của thước đo vào Nhật ký điểm, để về sau")
        print("     không ai diễn giải quá mức các thay đổi nhỏ ở điểm Q&A.")

    # Ngưỡng 0.80 hiện tại cắt đúng hay sai?
    print()
    print("-" * 78)
    hien_tai = 0.80
    sai_lam = [
        (c, d) for c, d in zip(cap, diem)
        if (c[2] == "dung" and d < hien_tai) or (c[2] == "sai" and d >= hien_tai)
    ]
    print(f"NGƯỠNG ĐANG DÙNG ({hien_tai:.2f}) chấm sai {len(sai_lam)}/{len(cap)} cặp:")
    for c, d in sai_lam[:10]:
        kieu = "đúng bị chấm SAI" if c[2] == "dung" else "sai bị chấm ĐÚNG"
        print(f"    {d:.3f}  [{kieu}]  {c[0][:24]!r} vs {c[1][:24]!r}")
    if not sai_lam:
        print("    (không cặp nào) — ngưỡng hiện tại đang tốt, không cần đổi.")


def main() -> int:
    args = sys.argv[1:]

    def lay(ten):
        return args[args.index(ten) + 1] if ten in args and args.index(ten) + 1 < len(args) else None

    if "--tao-mau" in args:
        return tao_mau(Path(lay("--tao-mau") or "cap_dap_an.csv"))

    tep = lay("--tep")
    if not tep:
        print(__doc__)
        return 1

    duong_dan = Path(tep)
    if not duong_dan.exists():
        print(f"Không thấy tệp: {duong_dan}")
        return 1

    cap = doc_cap(duong_dan)
    so_sai = sum(1 for c in cap if c[2] == "sai")

    print(f"Đọc được {len(cap)} cặp ({so_sai} cặp người chấm là 'sai').")

    if len(cap) < TOI_THIEU_CAP:
        print(f"CHƯA ĐỦ. Cần tối thiểu {TOI_THIEU_CAP} cặp — dưới mức đó thì ngưỡng chọn ra")
        print("chỉ phản ánh vài ví dụ lẻ, không phản ánh bộ dev.")
        return 1

    if so_sai < TOI_THIEU_SAI:
        print(f"CHƯA ĐỦ CẶP SAI. Cần tối thiểu {TOI_THIEU_SAI}. Không có cặp sai thì")
        print("không biết ngưỡng phải cắt ở đâu — mọi giá trị đều 'đúng hết'.")
        return 1

    diem = do_tuong_dong(cap)
    if diem is None:
        return 1

    bao_cao(cap, diem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
