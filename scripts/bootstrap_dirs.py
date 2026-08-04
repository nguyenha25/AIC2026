from src.aic2026.paths import DATA_ROOT, RAW_DIR, DERIVED_DIR, INDEX_DIR

def create_directories():
    print(f"Đang tạo cấu trúc dữ liệu tại: {DATA_ROOT}")
    
    # Danh sách các thư mục cần tạo theo chuẩn (Task 4 & Giai đoạn 1)
    dirs_to_create = [
        # Tầng RAW
        RAW_DIR / "map-keyframes",
        RAW_DIR / "clip-features-32",
        RAW_DIR / "objects",
        RAW_DIR / "media-info",
        RAW_DIR / "keyframes",
        RAW_DIR / "videos",
        
        # Tầng DERIVED
        DERIVED_DIR / "thumbnails",
        DERIVED_DIR / "ocr",
        DERIVED_DIR / "asr",
        DERIVED_DIR / "audio",
        DERIVED_DIR / "frames_dense",
        DERIVED_DIR / "captions",
        
        # Tầng INDEX
        INDEX_DIR / "frame_map.parquet",
        INDEX_DIR / "faiss",
        INDEX_DIR / "fts",
        
        # Các thư mục lưu kết quả và bài nộp
        DATA_ROOT / "dev",
        DATA_ROOT / "runs",
        DATA_ROOT / "submissions"
    ]
    
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  + Đã tạo: {d.relative_to(DATA_ROOT)}")
        
    print("Hoàn tất tạo cấu trúc thư mục!")

if __name__ == "__main__":
    create_directories()