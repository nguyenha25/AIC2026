import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fake_easyocr import install as install_fake_easyocr

install_fake_easyocr()  # phai chay TRUOC khi import aic2026.enrich.ocr

import pytest
from aic2026.enrich.ocr import OCRExtractor, _atomic_write_jsonl


class FakeFrameMap:
    """Gia lap FrameMap.frame_idx_of(n). n=999 gia lap 'khong co trong map'."""

    def frame_idx_of(self, n: int) -> int:
        if n == 999:
            raise KeyError(f"n={n} khong co trong frame_map")
        return n * 100


@pytest.fixture
def img_dir(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    # 3 anh binh thuong
    for n in (1, 2, 3):
        (d / f"{n:03d}.jpg").write_bytes(b"fake")
    # 1 anh se gia lap OCR loi (noi dung = CORRUPT -> readtext raise)
    (d / "004.jpg").write_bytes(b"CORRUPT")
    # 1 anh co n=999 -> frame_idx_of se raise KeyError
    (d / "999.jpg").write_bytes(b"fake")
    return d


def test_one_bad_image_does_not_abort_whole_video(tmp_path, img_dir):
    """Truoc day: 1 anh loi OCR se lam exception bay len va MAT CA VIDEO.
    Sau khi sua: cac anh con lai van duoc xu ly va ghi ra file."""
    img_paths = sorted(img_dir.glob("*.jpg"))
    extractor = OCRExtractor(languages=["vi", "en"], gpu=False)
    out_dir = tmp_path / "derived" / "ocr"
    error_log = tmp_path / "derived" / "ocr" / "_errors.log"

    out_file = extractor.process_video_keyframes(
        video_id="L99_V001",
        img_paths=img_paths,
        output_dir=out_dir,
        frame_map=FakeFrameMap(),
        error_log_path=error_log,
    )

    assert out_file.exists()
    records = [json.loads(l) for l in out_file.read_text(encoding="utf-8").splitlines()]

    # n=1,2,3 OK; n=4 (CORRUPT) van co record nhung text rong + co field error;
    # n=999 bi loai hoan toan vi frame_idx_of loi (khong doan frame_idx).
    by_n = {r["n"]: r for r in records}
    assert set(by_n.keys()) == {1, 2, 3, 4}
    assert 999 not in by_n

    assert by_n[1]["text"] == "text_001"
    assert by_n[4]["text"] == ""
    assert "error" in by_n[4]
    assert by_n[4]["frame_idx"] == 400  # frame_idx_of van chay dung cho anh nay

    # loi phai duoc ghi vao error log de kiem tra thu cong
    assert error_log.exists()
    log_content = error_log.read_text(encoding="utf-8")
    assert "n=999" in log_content
    assert "n=4" in log_content


def test_no_tmp_file_left_behind_on_success(tmp_path, img_dir):
    img_paths = sorted(img_dir.glob("*.jpg"))
    extractor = OCRExtractor(languages=["vi", "en"], gpu=False)
    out_dir = tmp_path / "derived" / "ocr"

    out_file = extractor.process_video_keyframes(
        video_id="L99_V002",
        img_paths=img_paths,
        output_dir=out_dir,
        frame_map=FakeFrameMap(),
    )
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    assert out_file.exists()
    assert not tmp_file.exists()


def test_atomic_write_helper_replaces_only_after_full_write(tmp_path):
    out_file = tmp_path / "x.jsonl"

    def gen():
        yield {"a": 1}
        yield {"a": 2}

    count = _atomic_write_jsonl(gen(), out_file)
    assert count == 2
    assert out_file.exists()
    assert not (tmp_file := out_file.with_suffix(".jsonl.tmp")).exists()


def test_all_frames_fail_frame_idx_raises_and_cleans_up(tmp_path, img_dir):
    """Neu TOAN BO frame trong video deu loi frame_idx_of, khong duoc de lai
    file rong gay hieu nham 'da xong' -- phai raise de nguoi dung biet."""

    class AllFailFrameMap:
        def frame_idx_of(self, n):
            raise KeyError("khong co video nay trong frame_map")

    img_paths = [img_dir / "001.jpg"]
    extractor = OCRExtractor(languages=["vi", "en"], gpu=False)
    out_dir = tmp_path / "derived" / "ocr"

    with pytest.raises(RuntimeError):
        extractor.process_video_keyframes(
            video_id="L99_V003",
            img_paths=img_paths,
            output_dir=out_dir,
            frame_map=AllFailFrameMap(),
        )

    assert not (out_dir / "L99_V003.jsonl").exists()
