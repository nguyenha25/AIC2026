import argparse
import subprocess
import sys
import time
from pathlib import Path
 
# CHỈNH: sửa đường dẫn này cho khớp paths.py thật của nhóm
# (nên import từ aic2026.paths thay vì ghi tay, xem ghi chú cuối file)
GOC_DU_LIEU = Path("D:/aic-data")   # <-- SỬA theo .env của máy bạn
 
VIDEO_DIR = GOC_DU_LIEU / "raw" / "videos"
AUDIO_DIR = GOC_DU_LIEU / "derived" / "audio"
 
VIDEO_EXTENSION = ".mp4"
DURATION_TOLERANCE_SEC = 1.0  # độ dài audio và video lệch dưới 1 giây thì coi là khớp
 
 
def get_duration_seconds(path: Path) -> float:
    """Dùng ffprobe để lấy độ dài (giây) của bất kỳ tệp media nào."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())
 
 
def extract_audio(video_path: Path, tmp_output_path: Path) -> None:
    """Tách tiếng ra .wav 16kHz mono bằng ffmpeg, ghi ra tệp tạm.
 
    Dung -f wav de chi dinh dang tuong minh, khong de ffmpeg tu doan
    qua duoi file -- vi tep tam co duoi ".wav.tmp", ffmpeg se doc nham
    duoi cuoi la ".tmp" va khong biet dinh dang neu khong chi ro.
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
 
 
def is_video_done(video_id: str) -> bool:
    """Video đã được xử lý xong chưa -- kiểm bằng sự tồn tại của tệp .wav kết quả."""
    return (AUDIO_DIR / f"{video_id}.wav").exists()
 
 
def count_total_processed() -> int:
    """Đếm tổng số video đã xử lý CỘNG DỒN từ trước tới giờ (đếm số tệp .wav đang có)."""
    return len(list(AUDIO_DIR.glob("*.wav")))
 
 
def process_video(video_path: Path, dry_run: bool) -> str:
    """
    Xử lý một video: tách tiếng -> xác minh độ dài khớp -> xoá video.
    Trả về chuỗi mô tả kết quả để in ra màn hình.
    """
    video_id = video_path.stem
 
    if is_video_done(video_id):
        return "đã xong, bỏ qua"
 
    if dry_run:
        return "SẼ xử lý (dry-run, chưa làm gì)"
 
    output_path = AUDIO_DIR / f"{video_id}.wav"
    tmp_output_path = AUDIO_DIR / f"{video_id}.wav.tmp"
 
    # Bước 1: tách tiếng ra tệp tạm
    extract_audio(video_path, tmp_output_path)
 
    # Bước 2: xác minh độ dài khớp trước khi làm bất cứ điều gì khác
    video_duration = get_duration_seconds(video_path)
    audio_duration = get_duration_seconds(tmp_output_path)
    diff = abs(video_duration - audio_duration)
 
    if diff > DURATION_TOLERANCE_SEC:
        # KHÔNG xoá video, KHÔNG giữ tệp tạm lỗi -- để người kiểm tra lại bằng tay
        tmp_output_path.unlink(missing_ok=True)
        return (f"LỖI: độ dài lệch {diff:.2f} giây (video {video_duration:.1f}s, "
                f"audio {audio_duration:.1f}s) -- KHÔNG xoá video, cần kiểm tra lại")
 
    # Bước 3: đổi tên tệp tạm thành tệp chính thức
    tmp_output_path.rename(output_path)
 
    # Bước 4: CHỈ xoá video sau khi đã xác minh khớp ở bước 2
    video_path.unlink()
 
    return f"xong, độ dài khớp (lệch {diff:.2f}s), đã xoá video"
 
 
def main():
    parser = argparse.ArgumentParser(description="Tách tiếng và xoá video sau khi xác minh")
    parser.add_argument("--dry-run", action="store_true",
                         help="Chỉ in danh sách sẽ xử lý, không tách tiếng và không xoá gì cả")
    args = parser.parse_args()
 
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
 
    video_list = sorted(VIDEO_DIR.glob(f"*{VIDEO_EXTENSION}"))
 
    if not video_list:
        print(f"Không tìm thấy video nào trong {VIDEO_DIR}")
        return
 
    if args.dry_run:
        print("=== CHẾ ĐỘ DRY-RUN -- CHƯA LÀM GÌ CẢ ===\n")
 
    print(f"Tìm thấy {len(video_list)} video trong thư mục\n")
 
    done_count = 0
    success_count = 0
    error_count = 0
 
    for i, video_path in enumerate(video_list, start=1):
        start_time = time.time()
        try:
            result = process_video(video_path, dry_run=args.dry_run)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            result = f"LỖI: {e}"
 
        elapsed = time.time() - start_time
        print(f"[{i}/{len(video_list)}] {video_path.name} -> {result} ({elapsed:.1f}s)")
 
        if "đã xong, bỏ qua" in result:
            done_count += 1
        elif result.startswith("LỖI"):
            error_count += 1
        elif not args.dry_run:
            success_count += 1
 
    print("\n" + "=" * 50)
    print("KẾT QUẢ LẦN CHẠY NÀY")
    print("=" * 50)
    print(f"Đã xong từ trước (bỏ qua lần này): {done_count}")
    if not args.dry_run:
        print(f"Xử lý thành công lần này:          {success_count}")
        print(f"Lỗi (chưa xoá video, cần xem lại):  {error_count}")
 
        # Tổng số cộng dồn từ trước tới giờ, không chỉ riêng lần chạy này
        total_processed = count_total_processed()
        print("-" * 50)
        print(f"TỔNG SỐ VIDEO ĐÃ XỬ LÝ (cộng dồn từ trước tới giờ): {total_processed}")
    else:
        print("Đây là dry-run -- chạy lại KHÔNG có --dry-run để xử lý thật.")
    print("=" * 50)
 
 
if __name__ == "__main__":
    main()