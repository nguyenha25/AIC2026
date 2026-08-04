from src.aic2026.paths import DATA_ROOT, PROJECT_ROOT, RAW_DIR

def verify():
    errors = []
    
    # 1. Kiểm tra đã khai báo đường dẫn chưa
    if not DATA_ROOT.exists():
        errors.append(f"Lỗi: Không tìm thấy thư mục gốc dữ liệu tại {DATA_ROOT}.")
        
    # 2. Kiểm tra dữ liệu có nằm nhầm trong mã nguồn không
    if PROJECT_ROOT in DATA_ROOT.parents or DATA_ROOT.resolve() == PROJECT_ROOT.resolve():
        errors.append("Lỗi: Gốc dữ liệu đang nằm bên trong thư mục mã nguồn! Tuyệt đối nguy hiểm!")
        
    # 3. Kiểm tra đủ các thư mục chuẩn chưa
    if not RAW_DIR.exists() or not (RAW_DIR / "map-keyframes").exists():
        errors.append("Lỗi: Thiếu các thư mục chuẩn. Hãy chạy bootstrap_dirs.py trước.")

    # 4. Đếm số lượng video và keyframes
    # Danh sách các thư mục chứa dữ liệu giải nén thực tế
    folders_to_count = ["clip-features-32", "objects", "media-info", "map-keyframes"]
    counts = {}
    
    for folder_name in folders_to_count:
        folder_path = RAW_DIR / folder_name
        if folder_path.exists():
            # rglob('*') sẽ chui vào từng thư mục con để đếm toàn bộ file
            counts[folder_name] = sum(1 for f in folder_path.rglob('*') if f.is_file())
        else:
            counts[folder_name] = 0

    # 5. Xuất kết quả
    if errors:
        for err in errors:
            print(err)
        print("\n=> KẾT LUẬN: KHÔNG ĐẠT")
    else:
        print("\n=> KẾT LUẬN: ĐẠT")
        print("-" * 50)
        print("THỐNG KÊ DỮ LIỆU ĐÃ GIẢI NÉN (TASK 4):")
        # In kết quả đếm của từng thư mục ra màn hình
        for folder_name, count in counts.items():
            print(f"  + Thư mục raw/{folder_name:<18}: {count:>8,} files")
        print("-" * 50)

if __name__ == "__main__":
    verify()