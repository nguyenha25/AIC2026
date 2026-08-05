"""
Tạo toàn bộ thư mục rỗng của gốc dữ liệu.

Chạy:  python -m scripts.bootstrap_dirs   (đứng ở thư mục gốc dự án)

Chạy lại nhiều lần vô hại: thư mục đã có thì bỏ qua, không xóa gì cả.
"""

from src.aic2026.paths import (
    DATA_ROOT,
    PROJECT_ROOT,
    REQUIRED_DIRS,
    free_space_gb,
)


def main() -> int:
    # Chặn ngay trường hợp nguy hiểm nhất: để dữ liệu trong thư mục mã nguồn.
    if DATA_ROOT.resolve() == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in DATA_ROOT.resolve().parents:
        print("DỪNG: gốc dữ liệu đang nằm bên trong thư mục mã nguồn.")
        print(f"  Mã nguồn : {PROJECT_ROOT}")
        print(f"  Dữ liệu  : {DATA_ROOT}")
        print("Sửa DATA_ROOT trong .env sang một chỗ khác rồi chạy lại.")
        return 1

    print(f"Đang tạo cấu trúc dữ liệu tại: {DATA_ROOT}")
    created = 0
    for d in REQUIRED_DIRS:
        if not d.exists():
            created += 1
        d.mkdir(parents=True, exist_ok=True)
        print(f"  + {d.relative_to(DATA_ROOT)}")

    print("-" * 60)
    print(f"Xong. {len(REQUIRED_DIRS)} thư mục chuẩn, tạo mới {created}.")
    print(f"Ổ đĩa còn trống: {free_space_gb(DATA_ROOT):.1f} GB")
    print("Bước tiếp theo:  python -m scripts.verify_layout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
