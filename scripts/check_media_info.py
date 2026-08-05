"""
TASK 5 (phần hai) — soát media-info.

Chạy:  python -m scripts.check_media_info

Trả lời hai câu hỏi của Task 5:
  1. Bao nhiêu video KHÔNG có tệp media-info? (BTC đã báo trước là có thiếu)
  2. Tiếng Việt có dấu đọc lên có đúng không, hay ra ký tự lạ?

Câu 2 quan trọng vì nếu máy đọc sai bảng mã thì tiêu đề và mô tả video
biến thành rác, mà chương trình vẫn chạy bình thường — sẽ đi tìm kiếm
trên dữ liệu rác mà không ai biết.
"""

import json
import unicodedata

from src.aic2026.paths import (
    MEDIA_INFO_DIR,
    list_video_ids,
    map_keyframes_file,
    media_info_file,
)
from src.aic2026.frame_map import available_video_ids

TEXT_FIELDS = ["title", "description", "keywords", "author", "channel_id"]


def has_vietnamese_diacritics(text: str) -> bool:
    """Có ít nhất một chữ cái tiếng Việt mang dấu."""
    for ch in text:
        if ch.isalpha() and ord(ch) > 127:
            decomposed = unicodedata.normalize("NFD", ch)
            if len(decomposed) > 1:
                return True
    return False


def main() -> int:
    all_videos = available_video_ids()
    if not all_videos:
        print("Chưa có map-keyframes trên máy. Làm Task 4 trước.")
        return 1

    have_info = set(list_video_ids(MEDIA_INFO_DIR, ".json"))
    missing = [v for v in all_videos if v not in have_info]

    print("=" * 66)
    print("SOÁT MEDIA-INFO (Task 5)")
    print("=" * 66)
    print(f"Tổng số video (theo map-keyframes) : {len(all_videos)}")
    print(f"Video CÓ media-info                : {len(all_videos) - len(missing)}")
    print(f"Video THIẾU media-info             : {len(missing)}")
    if all_videos:
        print(f"Tỉ lệ thiếu                        : {len(missing) / len(all_videos):.1%}")
    print()

    if missing:
        show = missing[:15]
        print("Vài video thiếu:", ", ".join(show))
        if len(missing) > 15:
            print(f"... và {len(missing) - 15} video nữa")
        print()
        print("=> Chương trình phải chịu được chỗ khuyết này: đọc media-info")
        print("   luôn bọc trong kiểm tra tệp có tồn tại không, không được")
        print("   giả định video nào cũng có.")
        print()

    # --- Kiểm bảng mã tiếng Việt -----------------------------------------
    print("-" * 66)
    print("KIỂM TIẾNG VIỆT CÓ DẤU")
    print("-" * 66)

    checked = 0
    found_diacritics = 0
    broken: list[str] = []
    sample_printed = False

    for vid in all_videos:
        path = media_info_file(vid)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            broken.append(vid)
            continue
        except json.JSONDecodeError:
            broken.append(vid)
            continue

        checked += 1
        blob = " ".join(
            str(data.get(f, "")) for f in TEXT_FIELDS
        )
        if has_vietnamese_diacritics(blob):
            found_diacritics += 1
            if not sample_printed:
                title = str(data.get("title", ""))[:70]
                print(f"Ví dụ đọc được: {vid}")
                print(f"  title: {title}")
                print()
                sample_printed = True
        if checked >= 300:      # đủ mẫu rồi, không cần quét hết
            break

    print(f"Đã mở thử               : {checked} tệp")
    print(f"Có tiếng Việt có dấu    : {found_diacritics} tệp")
    print(f"Mở KHÔNG được           : {len(broken)} tệp")
    print()

    if broken:
        print(f"[LỖI] {len(broken)} tệp không đọc được bằng UTF-8: {broken[:5]}")
        return 1
    if checked and found_diacritics == 0:
        print("[CẢNH BÁO] Không thấy chữ tiếng Việt có dấu nào. Có thể bảng mã sai,")
        print("           hoặc phần dữ liệu này vốn toàn tiếng Anh. Kiểm mắt thường.")
    else:
        print("[ĐẠT] Tiếng Việt có dấu đọc lên bình thường (UTF-8).")

    print()
    print("Ghi con số 'video THIẾU media-info' vào sheet Check list, cột Output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
