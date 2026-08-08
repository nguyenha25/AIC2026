"""Tách tiếng từ video ra .wav 16kHz mono, xác minh khớp, MẶC ĐỊNH GIỮ VIDEO.

Việc 14 · AIC 2026 Giai đoạn 1.

Quyết định đã chốt: KHÔNG xoá video sau khi tách tiếng. Ổ của cả bốn máy còn dư,
và video còn cần cho TRAKE, frames_dense/ và VideoQA ở Giai đoạn 2.
Cờ --purge chỉ dùng khi thật sự chật ổ.

Cách chạy:
    python scripts/extract_audio.py --dry-run       # xem sẽ làm gì, không đụng tệp nào
    python scripts/extract_audio.py                 # tách tiếng, GIỮ video
    python scripts/extract_audio.py --limit 1       # chỉ 1 video, dùng cho việc 13
    python scripts/extract_audio.py --purge         # tách tiếng RỒI XOÁ video (chỉ khi chật ổ)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Đường dẫn lấy từ paths.py -- tệp DUY NHẤT biết đường dẫn thật trên đĩa.
# Nếu paths.py export tên khác thì sửa ĐÚNG hai dòng import và gán dưới đây,
# KHÔNG sửa paths.py (mục 5.1: không ai được sửa tệp đó).
try:
    from src.aic2026.paths import DATA_ROOT
except ImportError as e:
    print("Không import được aic2026.paths.")
    print("Kiểm hai điều: đã chạy 'pip install -e .' chưa, và paths.py có export")
    print("đúng tên DATA_ROOT không (nếu tên khác thì sửa dòng import trong tệp này).")
    print(f"Chi tiết: {e}")
    sys.exit(1)

VIDEO_DIR = DATA_ROOT / "raw" / "videos"
AUDIO_DIR = DATA_ROOT / "derived" / "audio"

VIDEO_EXTENSION = ".mp4"
OTHER_VIDEO_EXTENSIONS = (".mkv", ".webm", ".mov", ".avi", ".m4v")

# Độ dài wav lệch dưới ngưỡng này so với LUỒNG TIẾNG của video thì coi là khớp.
DURATION_TOLERANCE_SEC = 1.0

# Luồng tiếng bắt đầu trễ hơn ngưỡng này thì mốc thời gian ASR sẽ lệch so với
# cột giây của bảng đối chiếu -> không nhận wav đó.
START_TIME_TOLERANCE_SEC = 0.1


def get_duration_seconds(path: Path) -> float:
    """Độ dài (giây) theo container của bất kỳ tệp media nào.

    Dùng cho tệp .wav (chỉ có một luồng nên container = luồng tiếng), và dùng
    làm đường lui khi không đọc được luồng tiếng của video.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def probe_audio_stream(path: Path) -> dict | None:
    """Doc thong tin LUONG TIENG dau tien, khong phai cua container.

    Vi sao khong dung container: container dai bang luong dai nhat, thuong la
    luong hinh. Luong tieng ngan hon vai giay la binh thuong. So wav (= luong
    tieng) voi container (= luong hinh) la so sai dai luong, sinh bao dong gia.

    Tra ve None neu video KHONG CO luong tieng nao.
    Tra ve dict {'duration': float|None, 'start_time': float|None} neu co.

    BAT BUOC doc theo KHOA, khong theo so dong: ffprobe in ra theo thu tu noi
    bo cua no chu khong theo thu tu ghi trong -show_entries. Voi cap nay no in
    start_time TRUOC duration -- doc theo vi tri la doc nguoc hai gia tri.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=duration,start_time",
        "-of", "default=noprint_wrappers=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Hay gặp nhất: tệp .mp4 tải dở (thiếu moov atom). Chỉ lấy dòng cuối
        # của ffprobe -- đó là chỗ nó nói lý do thật.
        last_line = (result.stderr.strip().splitlines() or ["không rõ lý do"])[-1]
        raise RuntimeError(f"ffprobe không đọc được tệp: {last_line}")

    text = result.stdout.strip()
    if not text:
        return None  # không có luồng tiếng nào

    info: dict = {"duration": None, "start_time": None}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key in info:
            try:
                info[key] = float(value)
            except ValueError:
                info[key] = None  # "N/A" hoặc rỗng
    return info


def get_source_audio_duration(video_path: Path,
                              info: dict) -> tuple[float, float, str]:
    """Độ dài và start_time của luồng tiếng, kèm ghi chú nếu phải lui về container.

    Trả về (duration, start_time, ghi_chu).
    """
    start_time = info["start_time"] if info["start_time"] is not None else 0.0
    if info["duration"] is not None:
        return (info["duration"], start_time, "")
    # Luồng tiếng không khai độ dài -> lui về container, và nói rõ trong kết quả
    return (get_duration_seconds(video_path), start_time,
            " [luong tieng khong khai do dai, da so voi container]")


def extract_audio(video_path: Path, tmp_output_path: Path) -> None:
    """Tách tiếng ra .wav 16kHz mono bằng ffmpeg, ghi ra tệp tạm.

    Dung -f wav de chi dinh dang tuong minh, khong de ffmpeg tu doan qua duoi
    file -- vi tep tam co duoi ".wav.tmp", ffmpeg se doc nham duoi cuoi la
    ".tmp" va khong biet dinh dang neu khong chi ro.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-f", "wav",
        str(tmp_output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace")
        # Chi lay vai dong cuoi -- do la cho ffmpeg in ly do loi that su
        last_lines = "\n".join(stderr_text.strip().splitlines()[-5:])
        raise RuntimeError(f"ffmpeg lỗi (mã {result.returncode}):\n{last_lines}")


def is_audio_done(video_id: str) -> bool:
    """Đã có .wav kết quả cho video này chưa."""
    return (AUDIO_DIR / f"{video_id}.wav").exists()


def count_total_processed() -> int:
    """Tổng số .wav đang có -- cộng dồn từ trước tới giờ, không chỉ lần chạy này."""
    return len(list(AUDIO_DIR.glob("*.wav")))


def clean_stray_tmp_files() -> int:
    """Xoá tệp .wav.tmp còn sót từ lần chạy bị ngắt giữa chừng.

    An toàn: tệp .tmp luôn là tệp ghi dở, chưa bao giờ được xác minh.
    """
    count = 0
    for tmp_path in AUDIO_DIR.glob("*.wav.tmp"):
        tmp_path.unlink(missing_ok=True)
        count += 1
    return count


def _verify_existing_wav(video_path: Path, video_id: str, purge: bool,
                         dry_run: bool) -> str:
    """Xử lý trường hợp .wav đã có sẵn (lần chạy trước, hoặc lấy từ Drive)."""
    if not purge:
        return "đã có .wav, bỏ qua"

    if dry_run:
        return "đã có .wav, SẼ xác minh rồi xoá video"

    info = probe_audio_stream(video_path)
    if info is None:
        return ("LỖI: video không có luồng tiếng nhưng lại có .wav "
                "-- KHÔNG xoá video, cần kiểm tra lại")

    wav_duration = get_duration_seconds(AUDIO_DIR / f"{video_id}.wav")
    src_duration, _, ghi_chu = get_source_audio_duration(video_path, info)
    diff = abs(src_duration - wav_duration)

    if diff > DURATION_TOLERANCE_SEC:
        return (f"LỖI: .wav có sẵn nhưng lệch {diff:.2f}s so với video "
                f"(luồng tiếng {src_duration:.1f}s, wav {wav_duration:.1f}s) "
                f"-- KHÔNG xoá video, cần kiểm tra lại")

    video_path.unlink()
    return f"đã có .wav, xác minh khớp (lệch {diff:.2f}s), đã xoá video{ghi_chu}"


def process_video(video_path: Path, dry_run: bool, purge: bool) -> str:
    """Tách tiếng một video -> xác minh -> (chỉ khi purge) xoá video.

    Trả về chuỗi mô tả kết quả để in ra màn hình. Chuỗi bắt đầu bằng "LỖI"
    nghĩa là video KHÔNG bị đụng tới và cần người kiểm lại bằng tay.
    """
    video_id = video_path.stem

    if is_audio_done(video_id):
        return _verify_existing_wav(video_path, video_id, purge, dry_run)

    # Kiểm luồng tiếng TRƯỚC khi gọi ffmpeg. Video không có tiếng thì ffmpeg
    # sẽ văng lỗi khó đọc, mà đây không phải lỗi -- chỉ là video không cần ASR.
    info = probe_audio_stream(video_path)
    if info is None:
        return "KHÔNG CÓ TIẾNG: video không có luồng tiếng, không cần ASR"

    if dry_run:
        return "SẼ xử lý (dry-run, chưa làm gì)"

    output_path = AUDIO_DIR / f"{video_id}.wav"
    tmp_output_path = AUDIO_DIR / f"{video_id}.wav.tmp"

    try:
        # Bước 1: tách tiếng ra tệp tạm
        extract_audio(video_path, tmp_output_path)

        # Bước 2: xác minh TRƯỚC khi làm bất cứ điều gì khác.
        # So với luồng tiếng của video, không so với container.
        audio_duration = get_duration_seconds(tmp_output_path)
        src_duration, src_start, ghi_chu = get_source_audio_duration(video_path, info)
        diff = abs(src_duration - audio_duration)

        if diff > DURATION_TOLERANCE_SEC:
            tmp_output_path.unlink(missing_ok=True)
            return (f"LỖI: độ dài lệch {diff:.2f} giây (luồng tiếng "
                    f"{src_duration:.1f}s, wav {audio_duration:.1f}s) "
                    f"-- cần kiểm tra lại")

        # start_time > 0 nghĩa là wav bắt đầu trễ hơn video -> mọi mốc thời gian
        # ASR sẽ lệch đúng chừng đó so với cột giây của bảng đối chiếu.
        # Lỗi này KHÔNG có triệu chứng ở các bước sau nên phải chặn ngay đây.
        if src_start > START_TIME_TOLERANCE_SEC:
            tmp_output_path.unlink(missing_ok=True)
            return (f"LỖI: luồng tiếng bắt đầu ở giây {src_start:.2f} chứ không "
                    f"phải 0 -- mốc ASR sẽ lệch, cần xử lý riêng video này")

        # Bước 3: đổi tên tệp tạm thành tệp chính thức
        tmp_output_path.rename(output_path)

    except BaseException:
        # Ngắt giữa chừng hoặc lỗi bất kỳ -> không để lại tệp ghi dở
        tmp_output_path.unlink(missing_ok=True)
        raise

    # Bước 4: CHỈ xoá video khi có --purge, và chỉ sau khi đã xác minh ở bước 2
    if purge:
        video_path.unlink()
        return f"xong, độ dài khớp (lệch {diff:.2f}s), đã xoá video{ghi_chu}"

    return f"xong, độ dài khớp (lệch {diff:.2f}s), giữ video{ghi_chu}"


def warn_other_video_extensions() -> None:
    """Cảnh báo nếu thư mục có video đuôi khác .mp4 -- script sẽ bỏ qua chúng."""
    others = [p.name for p in VIDEO_DIR.iterdir()
              if p.is_file() and p.suffix.lower() in OTHER_VIDEO_EXTENSIONS]
    if others:
        print(f"CẢNH BÁO: có {len(others)} tệp video đuôi khác {VIDEO_EXTENSION} "
              f"sẽ bị BỎ QUA, ví dụ: {', '.join(others[:3])}")
        print("   Chuyển sang .mp4 hoặc sửa VIDEO_EXTENSION nếu cần xử lý chúng.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Tách tiếng từ video ra .wav, xác minh khớp. MẶC ĐỊNH GIỮ VIDEO.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ in danh sách sẽ xử lý, không đụng tới tệp nào")
    parser.add_argument("--purge", action="store_true",
                        help="Xoá video gốc SAU KHI xác minh khớp. Mặc định GIỮ video. "
                             "Chỉ dùng khi chật ổ.")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="Chỉ xử lý N video đầu. Dùng cho việc 13 (đo tốc độ máy).")
    args = parser.parse_args()

    # In tiếng Việt ra tệp log không bị văng UnicodeEncodeError trên Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    if not VIDEO_DIR.exists():
        print(f"Không có thư mục {VIDEO_DIR}")
        print("Kiểm .env đã trỏ đúng gốc dữ liệu chưa, và đã tải video chưa.")
        sys.exit(1)

    if not args.dry_run:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        stray = clean_stray_tmp_files()
        if stray:
            print(f"Đã dọn {stray} tệp .wav.tmp còn sót từ lần chạy trước\n")

    video_list = sorted(VIDEO_DIR.glob(f"*{VIDEO_EXTENSION}"))
    warn_other_video_extensions()

    if not video_list:
        print(f"Không tìm thấy video {VIDEO_EXTENSION} nào trong {VIDEO_DIR}")
        return

    if args.limit > 0:
        video_list = video_list[:args.limit]
        print(f"Giới hạn: chỉ xử lý {len(video_list)} video đầu\n")

    if args.dry_run:
        print("=== CHẾ ĐỘ DRY-RUN -- CHƯA ĐỤNG TỚI TỆP NÀO ===\n")
    if args.purge:
        print("=== CHẾ ĐỘ --purge: VIDEO SẼ BỊ XOÁ sau khi xác minh khớp ===\n")
    else:
        print("Chế độ mặc định: GIỮ video sau khi tách tiếng\n")

    print(f"Tìm thấy {len(video_list)} video cần xét\n")

    skipped_count = 0
    success_count = 0
    error_count = 0
    no_audio_count = 0
    error_files: list[str] = []
    no_audio_files: list[str] = []

    total = len(video_list)
    for i, video_path in enumerate(video_list, start=1):
        # In TRƯỚC khi chạy -- video dài làm màn hình đứng im vài phút
        print(f"[{i}/{total}] {video_path.name} ... ", end="", flush=True)

        start_time = time.perf_counter()
        try:
            result = process_video(video_path, dry_run=args.dry_run, purge=args.purge)
        except (subprocess.CalledProcessError, RuntimeError, ValueError) as e:
            # ValueError: ffprobe trả về rỗng hoặc "N/A", hay gặp với tệp tải dở.
            # Bắt ở đây để một tệp hỏng không làm chết cả phiên đang chạy dở.
            result = f"LỖI: {e}"
        except KeyboardInterrupt:
            print("\n\nĐã dừng theo yêu cầu. Chạy lại để tiếp tục từ chỗ dở.")
            break

        elapsed = time.perf_counter() - start_time
        print(f"{result} ({elapsed:.1f}s)")

        if result.startswith("LỖI"):
            error_count += 1
            error_files.append(video_path.name)
        elif result.startswith("KHÔNG CÓ TIẾNG"):
            no_audio_count += 1
            no_audio_files.append(video_path.name)
        elif "bỏ qua" in result or "đã có .wav" in result:
            skipped_count += 1
        elif not args.dry_run:
            success_count += 1

    print("\n" + "=" * 60)
    print("KẾT QUẢ LẦN CHẠY NÀY")
    print("=" * 60)
    print(f"Đã có .wav từ trước:               {skipped_count}")

    if args.dry_run:
        print("\nĐây là dry-run -- chạy lại KHÔNG có --dry-run để xử lý thật.")
        print("=" * 60)
        return

    print(f"Tách tiếng thành công lần này:     {success_count}")
    print(f"Không có luồng tiếng (bỏ qua):     {no_audio_count}")
    print(f"Lỗi (cần xem lại):                 {error_count}")

    if no_audio_files:
        print("\nVideo không có tiếng -- KHÔNG phải lỗi, chỉ là không có gì cho ASR:")
        for name in no_audio_files:
            print(f"   - {name}")

    if error_files:
        print("\nDanh sách video bị lỗi -- video KHÔNG bị đụng tới, kiểm lại bằng tay:")
        for name in error_files:
            print(f"   - {name}")

    print("-" * 60)
    print(f"TỔNG SỐ .wav ĐANG CÓ (cộng dồn từ trước tới giờ): {count_total_processed()}")
    if not args.purge:
        print("Video gốc được GIỮ NGUYÊN (không dùng --purge).")
    print("=" * 60)


if __name__ == "__main__":
    main()
