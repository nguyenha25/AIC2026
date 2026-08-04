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
        
    # Xuất kết quả
    if errors:
        for err in errors:
            print(err)
        print("\n=> KẾT LUẬN: KHÔNG ĐẠT")
    else:
        print("\n=> KẾT LUẬN: ĐẠT")

if __name__ == "__main__":
    verify()