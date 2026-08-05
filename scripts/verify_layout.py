"""
Kiểm tra máy này có đúng chuẩn nhóm không.

Chạy:  python -m scripts.verify_layout

Bốn máy chạy lệnh này phải ra CÙNG số video ở cả bốn loại dữ liệu.
Lệch một con số nghĩa là có người tải thiếu hoặc giải nén nhầm chỗ.

Mã thoát: 0 = ĐẠT, 1 = KHÔNG ĐẠT (dùng được trong script tự động).
"""

from src.aic2026.paths import (
    DATA_ROOT,
    PHASE0_RAW_DIRS,
    PROJECT_ROOT,
    REQUIRED_DIRS,
    free_space_gb,
    list_video_ids,
)

ERROR_GB = 10.0
WARN_GB = 30.0


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    print("=" * 62)
    print("KIỂM TRA CẤU TRÚC MÁY NÀY")
    print("=" * 62)
    print(f"Mã nguồn : {PROJECT_ROOT}")
    print(f"Dữ liệu  : {DATA_ROOT}")
    print()

    # --- 1. Đã khai báo đường dẫn và đường dẫn có thật chưa ---------------
    if not DATA_ROOT.exists():
        errors.append(
            f"Không thấy gốc dữ liệu tại {DATA_ROOT}. "
            "Kiểm tra lại DATA_ROOT trong .env, hoặc chạy bootstrap_dirs trước."
        )

    # --- 2. Dữ liệu có nằm nhầm trong mã nguồn không ----------------------
    if DATA_ROOT.resolve() == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in DATA_ROOT.resolve().parents:
        errors.append(
            "Gốc dữ liệu đang nằm BÊN TRONG thư mục mã nguồn. "
            "Sớm muộn cũng có người đẩy hàng chục GB lên github."
        )

    # --- 3. Đủ thư mục chuẩn chưa ----------------------------------------
    missing = [d for d in REQUIRED_DIRS if not d.is_dir()]
    if missing:
        errors.append(f"Thiếu {len(missing)}/{len(REQUIRED_DIRS)} thư mục chuẩn. Chạy bootstrap_dirs.")
        for d in missing[:8]:
            errors.append(f"    thiếu: {d}")
    else:
        print(f"[OK] Đủ {len(REQUIRED_DIRS)}/{len(REQUIRED_DIRS)} thư mục chuẩn")

    # --- 4. Không còn tệp .zip sót lại ------------------------------------
    if DATA_ROOT.exists():
        leftovers = list(DATA_ROOT.rglob("*.zip"))
        if leftovers:
            warnings.append(
                f"Còn {len(leftovers)} tệp .zip chưa xóa "
                f"(vd: {leftovers[0].name}) — đang chiếm ổ gấp đôi."
            )

    # --- 5. Dung lượng trống ---------------------------------------------
    if DATA_ROOT.exists():
        free = free_space_gb(DATA_ROOT)
        if free < ERROR_GB:
            errors.append(f"Ổ chứa gốc dữ liệu chỉ còn {free:.1f} GB (dưới {ERROR_GB:.0f} GB).")
        elif free < WARN_GB:
            warnings.append(f"Ổ chứa gốc dữ liệu còn {free:.1f} GB (dưới {WARN_GB:.0f} GB) — sắp thiếu cho Giai đoạn 1.")
        else:
            print(f"[OK] Ổ đĩa còn trống {free:.1f} GB")

    # --- 6. Đếm SỐ VIDEO từng loại ---------------------------------------
    print()
    print("SỐ VIDEO ĐÃ CÓ TRÊN MÁY NÀY (Task 4)")
    print("-" * 62)
    print(f"{'thư mục':<20}{'số video':>10}   {'đầu':<12}{'cuối':<12}")
    print("-" * 62)

    counts: dict[str, int] = {}
    for name, (folder, suffix) in PHASE0_RAW_DIRS.items():
        ids = list_video_ids(folder, suffix)
        counts[name] = len(ids)
        first = ids[0] if ids else "-"
        last = ids[-1] if ids else "-"
        print(f"raw/{name:<16}{len(ids):>10}   {first:<12}{last:<12}")

    print("-" * 62)

    # Ba tệp đầu phải phủ đúng cùng một tập video. media-info được phép thiếu.
    core = {k: v for k, v in counts.items() if k != "media-info"}
    if core and len(set(core.values())) > 1:
        errors.append(
            f"Ba loại dữ liệu cốt lõi không khớp số video: {core}. "
            "Có tệp giải nén thiếu hoặc sai thư mục."
        )
    if counts.get("media-info", 0) and counts.get("map-keyframes", 0):
        thieu = counts["map-keyframes"] - counts["media-info"]
        print(f"Video KHÔNG có media-info: {thieu} "
              f"({counts['media-info']}/{counts['map-keyframes']}) — BTC đã báo trước.")

    print()
    print("So con số 'số video' này với ba máy còn lại. Lệch là có người tải thiếu.")
    print()

    # --- Kết luận ---------------------------------------------------------
    for w in warnings:
        print(f"[CẢNH BÁO] {w}")
    for e in errors:
        print(f"[LỖI] {e}")

    print()
    print("=" * 62)
    if errors:
        print("=> KẾT LUẬN: KHÔNG ĐẠT")
        print("=" * 62)
        return 1
    print("=> KẾT LUẬN: ĐẠT")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
