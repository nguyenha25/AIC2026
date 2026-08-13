import sys
from pathlib import Path


class FakeReader:
    def __init__(self, lang_list, gpu=False):
        pass

    def readtext(self, img_path):
        p = Path(img_path)
        stem = p.stem

        # 1. Lỗi giả lập theo tên file
        if "bad" in stem.lower() or "corrupt" in stem.lower():
            raise RuntimeError(f"Fake OCR Error on {stem}")

        # 2. Lỗi giả lập theo nội dung file
        if p.exists():
            content = p.read_bytes()

            if len(content) == 0:
                raise RuntimeError(f"Empty image file: {stem}")

            # Không phân biệt chữ hoa/chữ thường
            content_lower = content.lower()

            if any(
                keyword in content_lower
                for keyword in [
                    b"bad",
                    b"corrupt",
                    b"error",
                    b"fail",
                    b"invalid",
                ]
            ):
                raise RuntimeError(f"Corrupt image content in: {stem}")

        # 3. OCR thành công
        return [
            (
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                f"text_{stem}",
                0.99,
            )
        ]


class FakeEasyOCRModule:
    Reader = FakeReader


def install():
    sys.modules["easyocr"] = FakeEasyOCRModule()


def uninstall():
    sys.modules.pop("easyocr", None)