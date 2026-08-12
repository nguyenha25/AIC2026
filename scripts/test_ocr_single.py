import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from aic2026.frame_map import FrameMap
from aic2026.enrich.ocr import OCRExtractor

def main():
    # Nhận video_id từ tham số truyền vào hoặc dùng mặc định
    video_id = sys.argv[1] if len(sys.argv) > 1 else "L22_V001"
    
    data_root = Path(os.environ.get("DATA_ROOT", r"D:\aic-data"))
    keyframes_base = data_root / "raw" / "keyframes"
    output_dir = Path("derived/ocr")

    print(f"Loading FrameMap for {video_id}...")
    frame_map = FrameMap.load(video_id)

    print("Initializing EasyOCR...")
    extractor = OCRExtractor(languages=["vi", "en"], gpu=False)

    print(f"Processing OCR for {video_id}...")
    out_file = extractor.process_video_keyframes(
        video_id=video_id,
        keyframes_dir=keyframes_base,
        output_dir=output_dir,
        frame_map=frame_map
    )

    print(f"Done! Output saved to: {out_file}")

if __name__ == "__main__":
    main()