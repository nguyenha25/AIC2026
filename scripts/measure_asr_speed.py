"""
scripts/measure_asr_speed.py
Viec 13 - Do toc do xu ly ASR de lay con so uoc luong

Script nay DOC LAP, khong phu thuoc src/aic2026/enrich/asr.py
(vi asr.py la cua Viec 15, chua can co ngay trong ngay dau).
Dung de do nhanh va bao cao con so ngay ngay dau tien.

Cach dung:
    pip install faster-whisper
    python scripts/measure_asr_speed.py duong_dan_video_5_phut.mp4
"""

import subprocess
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

# ----- Cau hinh da chot o Viec 13 -----
MODEL_SIZE = "small"       # hoac "medium", KHONG dung large
COMPUTE_TYPE = "int8"      # bat buoc
AUDIO_TAM = Path("audio_output_tam.wav")


def lay_do_dai_video_giay(duong_dan_video: str) -> float:
    """Dung ffprobe (di kem ffmpeg) de lay do dai that cua video, tinh bang giay."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        duong_dan_video,
    ]
    ket_qua = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(ket_qua.stdout.strip())


def tach_tieng(duong_dan_video: str) -> float:
    """Tach tieng bang ffmpeg, tra ve so giay da mat."""
    if AUDIO_TAM.exists():
        AUDIO_TAM.unlink()

    cmd = [
        "ffmpeg", "-i", duong_dan_video,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(AUDIO_TAM),
    ]
    bat_dau = time.time()
    subprocess.run(cmd, capture_output=True, check=True)
    return time.time() - bat_dau


def chay_asr_thu(duong_dan_audio: Path) -> tuple[float, str]:
    """Chay faster-whisper tren tep audio, tra ve (so giay da mat, doan chu dau)."""
    bat_dau = time.time()
    model = WhisperModel(MODEL_SIZE, compute_type=COMPUTE_TYPE)
    segments, _info = model.transcribe(str(duong_dan_audio), language="vi")
    van_ban = " ".join(seg.text for seg in segments)
    thoi_gian = time.time() - bat_dau
    return thoi_gian, van_ban


def main():
    if len(sys.argv) != 2:
        print("Cach dung: python scripts/measure_asr_speed.py duong_dan_video.mp4")
        sys.exit(1)

    duong_dan_video = sys.argv[1]
    if not Path(duong_dan_video).exists():
        print(f"Khong tim thay tep: {duong_dan_video}")
        sys.exit(1)

    print(f"Dang do voi video: {duong_dan_video}")
    print(f"Cau hinh: faster-whisper, model={MODEL_SIZE}, compute_type={COMPUTE_TYPE}\n")

    do_dai_giay = lay_do_dai_video_giay(duong_dan_video)
    print(f"[1/3] Do dai video: {do_dai_giay:.1f} giay ({do_dai_giay/60:.2f} phut)")

    print("[2/3] Dang tach tieng bang ffmpeg...")
    thoi_gian_tach = tach_tieng(duong_dan_video)
    print(f"      -> Mat {thoi_gian_tach:.1f} giay")

    print("[3/3] Dang chay faster-whisper (co the mat vai phut)...")
    thoi_gian_asr, van_ban = chay_asr_thu(AUDIO_TAM)
    print(f"      -> Mat {thoi_gian_asr:.1f} giay")
    print(f"      -> Doan chu dau tien: {van_ban[:150]}...")

    tong_giay = thoi_gian_tach + thoi_gian_asr
    ti_le_quy_doi = 3600 / do_dai_giay
    uoc_tinh_phut_moi_gio = (tong_giay * ti_le_quy_doi) / 60

    print("\n" + "=" * 50)
    print("KET QUA")
    print("=" * 50)
    print(f"Do dai video thu:        {do_dai_giay/60:.2f} phut")
    print(f"Thoi gian tach tieng:    {thoi_gian_tach:.1f} giay")
    print(f"Thoi gian chay ASR:      {thoi_gian_asr:.1f} giay")
    print(f"Tong thoi gian may:      {tong_giay:.1f} giay ({tong_giay/60:.2f} phut)")
    print("-" * 50)
    print(f"UOC TINH: 1 gio audio se mat khoang {uoc_tinh_phut_moi_gio:.1f} phut may")
    print("=" * 50)
    print("\n=> Dien so nay vao thong bao gui nhom va sheet Nhat ky.")

    AUDIO_TAM.unlink(missing_ok=True)


if __name__ == "__main__":
    main()