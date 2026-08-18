"""
Việc 15 — Chạy chuyển lời nói thành chữ trên PHẦN MÌNH ĐÃ TÁCH.

Đọc  : derived/audio/*.wav   (kết quả Việc 14)
Ghi  : derived/asr/<video_id>.jsonl

Cách chạy (PowerShell):
    python -m scripts.run_asr_batch --dry-run          # xem sẽ làm gì, không chạy mô hình
    python -m scripts.run_asr_batch --limit 1          # thử thật 1 video trước
    python -m scripts.run_asr_batch                    # chạy hết phần mình
    python -m scripts.run_asr_batch --tien-to L23 L27  # chỉ vài mã video
    python -m scripts.run_asr_batch --khong-vad        # khi máy không nạp được onnxruntime

CHẠY LẠI ĐƯỢC: video nào đã có .jsonl hợp lệ thì bỏ qua. Ngắt giữa chừng
bằng Ctrl+C không để lại tệp ghi dở (ghi ra .tmp rồi mới đổi tên).

HAI LOẠI LỖI, XỬ LÝ KHÁC NHAU:
  - lỗi MỘT VIDEO (thiếu map-keyframes, wav hỏng): ghi nhật ký, chạy tiếp
  - lỗi CẤU HÌNH (thiếu gói, VAD không chạy):      DỪNG NGAY ở video đầu tiên
Kiểm cấu hình làm TRƯỚC vòng lặp, không để phát hiện 137 lần cùng một chuyện.

Nhật ký chạy: derived/asr/_nhat_ky.jsonl — mỗi video một dòng, dùng để báo
số liệu cho nhóm và để đối chiếu khi Nguyên gộp bốn phần.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# NẠP onnxruntime TRƯỚC MỌI IMPORT CỦA DỰ ÁN.
# Trên Windows, pandas (vào qua frame_map) làm onnxruntime nạp không được, và
# hậu quả chỉ lộ ra dưới dạng "faster-whisper báo thiếu gói onnxruntime".
# Chi tiết và số đo: phần "LUẬT THỨ TỰ NẠP" đầu tệp src/aic2026/enrich/asr.py.
#
# Không cần bắt lỗi ở đây: asr.py cũng nạp lại và giữ lý do thật để in ra.
# Dòng này chỉ cốt giành chỗ trước pandas.
# ---------------------------------------------------------------------------
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from src.aic2026.paths import ASR_DIR, AUDIO_DIR
    from src.aic2026.enrich.asr import (
        KIEU_TINH,
        MODEL_MAC_DINH,
        NGON_NGU_MAC_DINH,
        LoiCauHinh,
        chuyen_mot_video,
        dong_danh_dau_khong_loi_noi,
        kiem_onnxruntime,
        lay_fps,
        nap_model,
        ten_engine,
    )
except ImportError as e:
    print("Không import được aic2026. Kiểm hai điều:")
    print("  1. đã chạy 'pip install -e .' trong D:\\AIC2026 chưa")
    print("  2. đang đứng ở gốc dự án và gọi bằng 'python -m scripts.run_asr_batch'")
    print(f"Chi tiết: {e}")
    sys.exit(1)

NHAT_KY = "_nhat_ky.jsonl"


def jsonl_hop_le(duong_dan: Path) -> bool:
    """Tệp có tồn tại, khác rỗng, và dòng cuối là JSON đọc được không.

    Kiểm dòng CUỐI chứ không kiểm dòng đầu: tệp bị ngắt giữa chừng thì dòng
    đầu vẫn đẹp, chỉ dòng cuối mới cụt.
    """
    if not duong_dan.exists() or duong_dan.stat().st_size == 0:
        return False
    try:
        dong_cuoi = ""
        with duong_dan.open("r", encoding="utf-8") as f:
            for dong in f:
                if dong.strip():
                    dong_cuoi = dong
        ban_ghi = json.loads(dong_cuoi)
        return "video_id" in ban_ghi and "start" in ban_ghi and "frame_idx_start" in ban_ghi
    except (OSError, ValueError):
        return False


def ghi_nhat_ky(ban_ghi: dict) -> None:
    with (ASR_DIR / NHAT_KY).open("a", encoding="utf-8") as f:
        f.write(json.dumps(ban_ghi, ensure_ascii=False) + "\n")


def ghi_jsonl(duong_dan: Path, cac_dong: list[dict]) -> None:
    """Ghi ra .tmp rồi đổi tên — không bao giờ để lại tệp ghi dở."""
    tmp = duong_dan.with_suffix(".jsonl.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for dong in cac_dong:
                f.write(json.dumps(dong, ensure_ascii=False) + "\n")
        tmp.replace(duong_dan)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def don_tep_tmp() -> int:
    so = 0
    for tmp in ASR_DIR.glob("*.jsonl.tmp"):
        tmp.unlink(missing_ok=True)
        so += 1
    return so


def in_huong_dan_onnxruntime(ly_do: str) -> None:
    """faster-whisper chỉ nói 'cần onnxruntime'. Đây là lý do THẬT và cách sửa."""
    print("=" * 70)
    print("DỪNG — bộ lọc VAD không chạy được trên máy này")
    print("=" * 70)
    print("Lý do thật (faster-whisper giấu câu này đi):")
    print(f"    {ly_do}")
    print()
    print("Gói onnxruntime có trong pip list mà vẫn lỗi nghĩa là gói NẠP KHÔNG")
    print("ĐƯỢC, không phải chưa cài.")
    print()
    print("Nguyên nhân đã đo được trên máy nhóm: nạp pandas/pyarrow trước thì")
    print("onnxruntime nạp không nổi nữa. Tệp này đã nạp onnxruntime ở dòng đầu")
    print("để giành chỗ trước pandas — nếu vẫn thấy thông báo này thì có thứ")
    print("khác kéo pandas vào sớm hơn, hoặc bệnh khác hẳn. Chạy:")
    print("         python -m scripts.chuan_doan_vad")
    print()
    print("Hai cách sửa dứt điểm, không phụ thuộc thứ tự nạp:")
    print('  1. pip install "onnxruntime==1.18.1"   (bản cùng thời faster-whisper 1.0.2)')
    print("  2. Thấy 'DLL load failed' ngay cả khi nạp onnxruntime một mình:")
    print("     cài Visual C++ Redistributable (vc_redist.x64.exe), mở lại PowerShell.")
    print()
    print("Cần chạy ngay, sửa sau: thêm --khong-vad. Vẫn ra kết quả dùng được,")
    print("nhưng whisper dễ 'ảo giác' trên đoạn im lặng hơn — đọc phần VAD ở")
    print("đầu tệp src/aic2026/enrich/asr.py trước khi quyết định.")
    print("=" * 70)


def doc_tham_so():
    p = argparse.ArgumentParser(
        description="Việc 15 — chuyển lời nói thành chữ, kèm mốc thời gian từng câu."
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ liệt kê sẽ xử lý video nào, KHÔNG nạp mô hình, KHÔNG ghi tệp.")
    p.add_argument("--limit", type=int, default=0, metavar="N",
                   help="Chỉ xử lý N video đầu. Dùng để thử trước khi chạy hết.")
    p.add_argument("--tien-to", nargs="+", default=None, metavar="L23",
                   help="Chỉ chạy video có mã bắt đầu bằng các tiền tố này.")
    p.add_argument("--model", default=MODEL_MAC_DINH, choices=["tiny", "base", "small", "medium"],
                   help=f"Bản mô hình. Mặc định {MODEL_MAC_DINH}. KHÔNG có 'large' — chốt ở Việc 13.")
    p.add_argument("--lang", default=NGON_NGU_MAC_DINH,
                   help="Ép ngôn ngữ. Để 'auto' cho mô hình tự nhận.")
    p.add_argument("--beam-size", type=int, default=5,
                   help="5 là mặc định của thư viện. Để 1 thì nhanh gần gấp đôi, kém chính xác hơn.")
    p.add_argument("--threads", type=int, default=0,
                   help="Số nhân CPU. 0 = tự chọn.")
    p.add_argument("--khong-vad", action="store_true",
                   help="Chạy không cần onnxruntime. Kém an toàn hơn — xem phần VAD trong asr.py.")
    p.add_argument("--lam-lai", action="store_true",
                   help="Chạy lại cả những video đã có .jsonl (mặc định là bỏ qua).")
    return p.parse_args()


def main():
    args = doc_tham_so()

    # In tiếng Việt trên PowerShell không văng UnicodeEncodeError
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    if not AUDIO_DIR.exists():
        print(f"Không có thư mục {AUDIO_DIR}")
        print("Chạy Việc 14 trước: python -m scripts.extract_audio")
        sys.exit(1)

    ASR_DIR.mkdir(parents=True, exist_ok=True)

    danh_sach = sorted(AUDIO_DIR.glob("*.wav"))
    if args.tien_to:
        danh_sach = [p for p in danh_sach if p.stem.startswith(tuple(args.tien_to))]

    if not danh_sach:
        print(f"Không tìm thấy tệp .wav nào trong {AUDIO_DIR}")
        return

    if not args.lam_lai:
        con_lai = [p for p in danh_sach if not jsonl_hop_le(ASR_DIR / f"{p.stem}.jsonl")]
        da_xong = len(danh_sach) - len(con_lai)
        danh_sach = con_lai
    else:
        da_xong = 0

    if args.limit > 0:
        danh_sach = danh_sach[: args.limit]

    ngon_ngu = None if args.lang.lower() == "auto" else args.lang
    engine = ten_engine(args.model)
    dung_vad = not args.khong_vad

    print("=" * 70)
    print("VIỆC 15 — CHUYỂN LỜI NÓI THÀNH CHỮ")
    print("=" * 70)
    print(f"Nguồn tiếng      : {AUDIO_DIR}")
    print(f"Nơi ghi kết quả  : {ASR_DIR}")
    print(f"Cấu hình         : {engine}, beam={args.beam_size}, "
          f"ngôn ngữ={'tự nhận' if ngon_ngu is None else ngon_ngu}, "
          f"VAD={'bật' if dung_vad else 'TẮT'}")
    print(f"Đã xong từ trước : {da_xong} video (bỏ qua)")
    print(f"Sẽ xử lý lần này : {len(danh_sach)} video")
    print("=" * 70 + "\n")

    if args.dry_run:
        for i, wav in enumerate(danh_sach, start=1):
            print(f"[{i}/{len(danh_sach)}] SẼ xử lý {wav.name}")
        print("\nĐây là dry-run — chạy lại KHÔNG có --dry-run để làm thật.")
        return

    if not danh_sach:
        print("Không còn video nào cần chạy. Xong.")
        return

    # --- KIỂM CẤU HÌNH TRƯỚC VÒNG LẶP ---------------------------------------
    # Thứ hỏng cho cả 137 video thì phải phát hiện MỘT lần, trước khi tải model
    # về và trước khi đụng vào video đầu tiên.
    if dung_vad:
        vad_chay_duoc, ly_do = kiem_onnxruntime()
        if not vad_chay_duoc:
            in_huong_dan_onnxruntime(ly_do)
            sys.exit(1)
    else:
        print("CẢNH BÁO: đang chạy KHÔNG có VAD. Đã siết ngưỡng lọc và bật cắt")
        print("câu lặp để bù, nhưng nên sửa onnxruntime rồi chạy lại về sau.\n")

    stray = don_tep_tmp()
    if stray:
        print(f"Đã dọn {stray} tệp .jsonl.tmp còn sót từ lần chạy trước\n")

    print("Đang nạp mô hình (lần đầu có thể phải tải về, vài trăm MB)...", flush=True)
    try:
        model = nap_model(args.model, KIEU_TINH, args.threads)
    except LoiCauHinh as e:
        print(f"\nDỪNG — {e}")
        sys.exit(1)
    print("Nạp xong.\n")

    so_xong = 0
    so_khong_loi_noi = 0
    so_loi = 0
    so_dong_lap_cat = 0
    tep_loi: list[str] = []
    tong_giay_tieng = 0.0
    tong_giay_may = 0.0

    tong = len(danh_sach)
    for i, wav in enumerate(danh_sach, start=1):
        video_id = wav.stem
        # In TRƯỚC khi chạy — video dài làm màn hình đứng im vài phút
        print(f"[{i}/{tong}] {video_id} ... ", end="", flush=True)

        bat_dau = time.perf_counter()
        try:
            fps = lay_fps(video_id)
            cac_doan, thong_tin = chuyen_mot_video(
                duong_dan_wav=wav,
                video_id=video_id,
                fps=fps,
                model=model,
                ten_model=args.model,
                ngon_ngu=ngon_ngu,
                beam_size=args.beam_size,
                dung_vad=dung_vad,
            )
        except KeyboardInterrupt:
            print("\n\nĐã dừng theo yêu cầu. Chạy lại để tiếp tục từ chỗ dở.")
            break
        except LoiCauHinh as e:
            # Lỗi môi trường: 136 video sau sẽ hỏng y hệt. Dừng luôn.
            print("LỖI CẤU HÌNH\n")
            if "onnxruntime" in str(e).lower() or "vad" in str(e).lower():
                in_huong_dan_onnxruntime(str(e))
            else:
                print(f"DỪNG — {e}")
            sys.exit(1)
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
            # Lỗi của RIÊNG video này: ghi lại rồi chạy tiếp.
            mat = time.perf_counter() - bat_dau
            print(f"LỖI: {e} ({mat:.1f}s)")
            so_loi += 1
            tep_loi.append(video_id)
            ghi_nhat_ky({"video_id": video_id, "trang_thai": "loi",
                         "ly_do": str(e)[:300], "engine": engine})
            continue

        if cac_doan:
            cac_dong = [d.to_dict() for d in cac_doan]
            trang_thai = "xong"
        else:
            cac_dong = [dong_danh_dau_khong_loi_noi(
                video_id, fps, thong_tin["lang"], engine)]
            trang_thai = "khong_co_loi_noi"
            so_khong_loi_noi += 1

        ghi_jsonl(ASR_DIR / f"{video_id}.jsonl", cac_dong)

        mat = time.perf_counter() - bat_dau
        do_dai = thong_tin["do_dai_audio_giay"]
        tong_giay_tieng += do_dai
        tong_giay_may += mat
        so_dong_lap_cat += thong_tin["so_dong_lap_da_cat"]
        if trang_thai == "xong":
            so_xong += 1

        ghi_nhat_ky({
            "video_id": video_id,
            "trang_thai": trang_thai,
            "so_doan": len(cac_doan),
            "do_dai_audio_giay": do_dai,
            "giay_may": round(mat, 1),
            "fps": fps,
            "lang": thong_tin["lang"],
            "lang_probability": thong_tin["lang_probability"],
            "engine": engine,
            "beam_size": args.beam_size,
            "vad": thong_tin["vad"],
            "so_dong_lap_da_cat": thong_tin["so_dong_lap_da_cat"],
        })

        nhan = "không có lời nói" if trang_thai == "khong_co_loi_noi" else f"{len(cac_doan)} đoạn"
        if thong_tin["so_dong_lap_da_cat"]:
            nhan += f" (cắt {thong_tin['so_dong_lap_da_cat']} dòng lặp)"
        con = tong - i
        if tong_giay_may > 0 and con > 0:
            uoc = con * (tong_giay_may / i) / 60
            print(f"{nhan}, tiếng {do_dai/60:.1f} phút ({mat:.1f}s) — còn ~{uoc:.0f} phút")
        else:
            print(f"{nhan}, tiếng {do_dai/60:.1f} phút ({mat:.1f}s)")

    print("\n" + "=" * 70)
    print("KẾT QUẢ LẦN CHẠY NÀY")
    print("=" * 70)
    print(f"Có lời nói, đã ghi .jsonl : {so_xong}")
    print(f"Không có lời nói          : {so_khong_loi_noi}")
    print(f"Lỗi (cần xem lại)         : {so_loi}")
    if tep_loi:
        for ten in tep_loi:
            print(f"   - {ten}")
    if so_dong_lap_cat:
        print(f"Dòng lặp đã cắt           : {so_dong_lap_cat}")
        if not dung_vad:
            print("   (số này cao bất thường là dấu hiệu nên sửa onnxruntime rồi chạy lại)")
    if tong_giay_tieng > 0:
        ti_le = tong_giay_may / tong_giay_tieng
        print("-" * 70)
        print(f"Tổng tiếng đã xử lý       : {tong_giay_tieng/3600:.2f} giờ")
        print(f"Tổng thời gian máy        : {tong_giay_may/60:.1f} phút")
        print(f"ĐO ĐƯỢC: 1 giờ tiếng mất khoảng {ti_le*60:.1f} phút máy")
        print("   -> điền con số này vào sheet Nhật ký (Việc 13 mới là số ước lượng)")
    print("-" * 70)
    tong_tep = len([p for p in ASR_DIR.glob("*.jsonl") if not p.name.startswith("_")])
    print(f"TỔNG SỐ .jsonl ĐANG CÓ (cộng dồn từ trước tới giờ): {tong_tep}")
    print("=" * 70)


if __name__ == "__main__":
    main()