"""
Nơi DUY NHẤT trong toàn bộ dự án biết đường dẫn thật trên đĩa.

Quy tắc: không tệp nào khác được viết đường dẫn tuyệt đối.
Cần đường dẫn nào thì import từ đây.

    from src.aic2026.paths import MAP_KEYFRAMES_DIR, map_keyframes_file
"""

import os
import re
import shutil
from collections import deque
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Gốc chương trình (thư mục chứa .git). Suy ra từ vị trí chính tệp này,
#    KHÔNG phụ thuộc thư mục đang đứng khi chạy lệnh.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Nạp .env theo đường dẫn tuyệt đối. Nếu chỉ gọi load_dotenv() suông thì
# python đi tìm từ thư mục đang đứng — chạy từ thư mục con là không thấy.
load_dotenv(PROJECT_ROOT / ".env")

CONFIG_DIR = PROJECT_ROOT / "config"
DOCS_DIR = PROJECT_ROOT / "docs"
SCHEMA_DIR = DOCS_DIR / "schema"

# ---------------------------------------------------------------------------
# 2. Gốc dữ liệu — khai báo trong .env, mỗi máy một đường dẫn khác nhau
# ---------------------------------------------------------------------------
_data_root_env = os.getenv("DATA_ROOT")
if not _data_root_env:
    raise ValueError(
        "Chưa khai báo DATA_ROOT trong tệp .env!\n"
        f"Cách sửa: chép {PROJECT_ROOT / '.env.example'} thành .env "
        "rồi điền đường dẫn gốc dữ liệu của máy mình.\n"
        "Windows nhớ viết gạch chéo xuôi: D:/aic-data"
    )

# Bắt lỗi .env dính dòng TRƯỚC khi nó biến thành FileNotFoundError khó hiểu
# ở tận đáy ngăn xếp.
#
# Chuyện đã xảy ra thật: .env không có ký tự xuống dòng ở cuối, nên
#     Add-Content .env "ANTHROPIC_API_KEY=..."
# nối thẳng vào cuối dòng DATA_ROOT, thành:
#     DATA_ROOT=D:/aic-dataANTHROPIC_API_KEY=sk-ant-...
# Triệu chứng duy nhất là một đường dẫn lạ trong thông báo lỗi ở tệp khác.
if "=" in _data_root_env:
    raise ValueError(
        f"DATA_ROOT có dấu bằng bên trong: {_data_root_env!r}\n"
        "Gần như chắc chắn là .env bị dính hai dòng vào nhau — Add-Content nối "
        "thẳng vào cuối dòng cũ khi tệp không kết thúc bằng ký tự xuống dòng.\n"
        "Sửa bằng PowerShell:\n"
        '    Set-Content .env "DATA_ROOT=D:/aic-data"\n'
        '    Add-Content .env "ANTHROPIC_API_KEY=sk-ant-..."\n'
        "    type .env        # phải thấy HAI dòng riêng biệt"
    )

DATA_ROOT = Path(_data_root_env).expanduser()

# --- Ba tầng ---------------------------------------------------------------
RAW_DIR = DATA_ROOT / "raw"
DERIVED_DIR = DATA_ROOT / "derived"
INDEX_DIR = DATA_ROOT / "index"

# --- raw/ : BTC phát, CHỈ ĐỌC ---------------------------------------------
MAP_KEYFRAMES_DIR = RAW_DIR / "map-keyframes"
CLIP_FEATURES_DIR = RAW_DIR / "clip-features-32"
OBJECTS_DIR = RAW_DIR / "objects"
MEDIA_INFO_DIR = RAW_DIR / "media-info"
KEYFRAMES_DIR = RAW_DIR / "keyframes"
VIDEOS_DIR = RAW_DIR / "videos"

# --- derived/ : nhóm tự sinh, SAO LƯU BẮT BUỘC ----------------------------
THUMBNAILS_DIR = DERIVED_DIR / "thumbnails"
OCR_DIR = DERIVED_DIR / "ocr"
ASR_DIR = DERIVED_DIR / "asr"
AUDIO_DIR = DERIVED_DIR / "audio"
FRAMES_DENSE_DIR = DERIVED_DIR / "frames_dense"
CAPTIONS_DIR = DERIVED_DIR / "captions"

# --- index/ : dựng lại được ------------------------------------------------
FRAME_MAP_PARQUET = INDEX_DIR / "frame_map.parquet"   # TỆP, không phải thư mục
FAISS_DIR = INDEX_DIR / "faiss"
FTS_DIR = INDEX_DIR / "fts"

# --- các thư mục còn lại ---------------------------------------------------
DEV_DIR = DATA_ROOT / "dev"
RUNS_DIR = DATA_ROOT / "runs"
SUBMISSIONS_DIR = DATA_ROOT / "submissions"

# --- eval/ (nhánh 4b — Task 12) --------------------------------------------
# dev/dev_questions.jsonl : bộ câu hỏi tự chấm, xem docs/schema/README.md mục 5
DEV_QUERIES_PATH = DEV_DIR / "dev_questions.jsonl"
# derived/eval/ : kết quả tự chấm, SAO LƯU không bắt buộc — dựng lại được từ
# dev/dev_questions.jsonl + submissions/ bằng: python -m scripts.run_scoring
#
# KHÔNG còn ANSWER_MATCH_CACHE_PATH: bản đầu Task 12 định dùng LLM trả phí làm
# giám khảo ngữ nghĩa nên cần cache. Bản chốt dùng embedding chạy local, kết
# quả deterministic tuyệt đối — cache không còn tác dụng gì.
# Xem docs/decisions/001-cham-diem-task-12.md mục 3.
EVAL_DIR = DERIVED_DIR / "eval"
SCORING_LOG_PATH = EVAL_DIR / "scoring_results.csv"

# ---------------------------------------------------------------------------
# 3. Danh sách chuẩn — bootstrap_dirs và verify_layout dùng CHUNG danh sách
#    này. Sửa một chỗ là hai lệnh cùng đổi theo, không bao giờ lệch nhau.
# ---------------------------------------------------------------------------
REQUIRED_DIRS = [
    RAW_DIR,
    MAP_KEYFRAMES_DIR,
    CLIP_FEATURES_DIR,
    OBJECTS_DIR,
    MEDIA_INFO_DIR,
    KEYFRAMES_DIR,
    VIDEOS_DIR,
    DERIVED_DIR,
    THUMBNAILS_DIR,
    OCR_DIR,
    ASR_DIR,
    AUDIO_DIR,
    FRAMES_DENSE_DIR,
    CAPTIONS_DIR,
    INDEX_DIR,
    FAISS_DIR,
    FTS_DIR,
    DEV_DIR,
    RUNS_DIR,
    SUBMISSIONS_DIR,
    EVAL_DIR,
]

# Bốn thư mục BTC phát ở Giai đoạn 0 — dùng để so số video giữa bốn máy.
PHASE0_RAW_DIRS = {
    "map-keyframes": (MAP_KEYFRAMES_DIR, ".csv"),
    "clip-features-32": (CLIP_FEATURES_DIR, ".npy"),
    "objects": (OBJECTS_DIR, None),
    "media-info": (MEDIA_INFO_DIR, ".json"),
}


# ---------------------------------------------------------------------------
# 4. Hàm dựng đường dẫn theo tên video. Dùng những hàm này thay vì tự nối
#    chuỗi, để lỡ nhóm đổi cách đặt tên thì chỉ sửa ở đây.
# ---------------------------------------------------------------------------
def map_keyframes_file(video_id: str) -> Path:
    """raw/map-keyframes/L21_V001.csv (kể cả khi nằm trong thư mục lồng)"""
    return resolve(MAP_KEYFRAMES_DIR, video_id, ".csv") or MAP_KEYFRAMES_DIR / f"{video_id}.csv"


def clip_features_file(video_id: str) -> Path:
    """raw/clip-features-32/L21_V001.npy"""
    return resolve(CLIP_FEATURES_DIR, video_id, ".npy") or CLIP_FEATURES_DIR / f"{video_id}.npy"


# ---------------------------------------------------------------------------
# Tệp đánh số bên trong thư mục của một video
#
# BTC KHÔNG đệm số thống nhất. Đo thật trên máy nhóm:
#     raw/keyframes/Keyframes_L30/keyframes/L30_V023/001.jpg   <- BA chữ số
# Bản cũ ở đây gõ cứng f"{n:04d}.jpg" nên trả về 0001.jpg và mọi phép tra ảnh
# đều trượt. Triệu chứng rất dễ đọc nhầm: các việc 4, 4b, 10, 11, 12 chỉ báo
# "thiếu ảnh", trông y như chưa tải shard, chứ không báo lỗi gì.
#
# Nên không gõ cứng nữa: quét thư mục MỘT LẦN, lấy số trong tên tệp, rồi tra
# bằng bảng. Cách này chịu được mọi kiểu đệm, kể cả khi BTC đổi giữa chừng.
# ---------------------------------------------------------------------------

_bang_tep_cache: dict = {}


def bang_tep_theo_so(thu_muc: Path, duoi: str) -> dict[int, Path]:
    """{số trong tên tệp: đường dẫn}. Quét một lần rồi giữ trong RAM."""
    khoa = (str(thu_muc), duoi.lower())
    if khoa in _bang_tep_cache:
        return _bang_tep_cache[khoa]

    bang: dict[int, Path] = {}
    if thu_muc.is_dir():
        for item in thu_muc.iterdir():
            if not item.is_file() or item.suffix.lower() != duoi.lower():
                continue
            khop = re.search(r"(\d+)", item.stem)
            if khop:
                bang[int(khop.group(1))] = item

    _bang_tep_cache[khoa] = bang
    return bang


def _tep_theo_so(thu_muc: Path, so: int, duoi: str) -> Path:
    """Tệp mang số `so`. Không thấy thì trả đường dẫn bốn chữ số để báo lỗi."""
    tim_duoc = bang_tep_theo_so(thu_muc, duoi).get(int(so))
    return tim_duoc if tim_duoc is not None else thu_muc / f"{so:04d}{duoi}"


def objects_file(video_id: str, keyframe_n: int) -> Path:
    """raw/objects/L21_V001/0047.json — đệm bao nhiêu chữ số cũng tra được"""
    base = resolve(OBJECTS_DIR, video_id) or OBJECTS_DIR / video_id
    return _tep_theo_so(base, keyframe_n, ".json")


def media_info_file(video_id: str) -> Path:
    """raw/media-info/L21_V001.json — MỘT SỐ VIDEO KHÔNG CÓ TỆP NÀY"""
    return resolve(MEDIA_INFO_DIR, video_id, ".json") or MEDIA_INFO_DIR / f"{video_id}.json"


def keyframe_image(video_id: str, keyframe_n: int) -> Path:
    """raw/keyframes/L21_V001/047.jpg — BTC đệm ba chữ số, KHÔNG phải bốn."""
    base = resolve(KEYFRAMES_DIR, video_id) or KEYFRAMES_DIR / video_id
    return _tep_theo_so(base, keyframe_n, ".jpg")


def video_file(video_id: str) -> Path:
    """raw/videos/L21_V001.mp4"""
    return resolve(VIDEOS_DIR, video_id, ".mp4") or VIDEOS_DIR / f"{video_id}.mp4"


def ocr_file(video_id: str) -> Path:
    return OCR_DIR / f"{video_id}.jsonl"


def asr_file(video_id: str) -> Path:
    return ASR_DIR / f"{video_id}.jsonl"


def captions_file(video_id: str) -> Path:
    return CAPTIONS_DIR / f"{video_id}.jsonl"


def thumbnail_image(video_id: str, keyframe_n: int) -> Path:
    return THUMBNAILS_DIR / video_id / f"{keyframe_n:04d}.jpg"


def run_dir(tag: str) -> Path:
    """runs/2026-08-15_1430_clipb32 — mỗi lần chạy một thư mục, không ghi đè."""
    return RUNS_DIR / tag


# ---------------------------------------------------------------------------
# 5. Tiện ích
# ---------------------------------------------------------------------------
# Tên video BTC đặt: L21_V001. Dùng để phân biệt thư mục dữ liệu thật với
# thư mục bọc ngoài do giải nén sinh ra (map-keyframes-aic25-b1, ...).
VIDEO_ID_RE = re.compile(r"^L\d{2,}_V\d{3,}$")

MAX_SEARCH_DEPTH = 4     # đủ sâu cho mọi kiểu giải nén, không quét lan man

_scan_cache: dict = {}


def _scan(folder: Path, suffix: str | None) -> dict[str, Path]:
    """
    Tìm dữ liệu thật bên trong `folder`, chui qua các tầng bọc do giải nén.

    Tệp BTC phát khi giải nén thường sinh thêm một hai tầng thư mục, ví dụ:
        raw/map-keyframes/map-keyframes-aic25-b1/map-keyframes/L21_V001.csv
    Đó là cấu trúc BTC phát ra, KHÔNG sắp lại bằng tay. Hàm này đi tìm
    tầng nào thật sự chứa dữ liệu rồi dừng ngay tại đó.

    Trả về: {video_id: đường dẫn thật}
    """
    if not folder.exists():
        return {}

    key = (str(folder), suffix)
    if key in _scan_cache:
        return _scan_cache[key]

    found: dict[str, Path] = {}
    queue = deque([(folder, 0)])

    while queue and not found:
        level_dirs = []
        while queue:
            current, depth = queue.popleft()
            if depth > MAX_SEARCH_DEPTH:
                continue
            try:
                children = list(current.iterdir())
            except (PermissionError, OSError):
                continue

            for item in children:
                if item.name.startswith("."):
                    continue
                if item.is_file():
                    if suffix and item.suffix.lower() == suffix.lower():
                        found[item.stem] = item
                elif item.is_dir():
                    if VIDEO_ID_RE.match(item.name):
                        found[item.name] = item      # objects/, keyframes/
                    else:
                        level_dirs.append((item, depth + 1))

        if not found:
            queue.extend(level_dirs)

    _scan_cache[key] = found
    return found


def refresh_scan_cache() -> None:
    """Gọi sau khi vừa giải nén thêm dữ liệu trong cùng một lần chạy."""
    _scan_cache.clear()
    _bang_tep_cache.clear()


def resolve(folder: Path, video_id: str, suffix: str | None = None) -> Path | None:
    """Đường dẫn thật của một video, kể cả khi nằm trong thư mục lồng."""
    return _scan(folder, suffix).get(video_id)


def list_video_ids(folder: Path, suffix: str | None = None) -> list[str]:
    """
    Đếm SỐ VIDEO trong một thư mục, không phải số tệp.

    - map-keyframes/, clip-features-32/, media-info/: mỗi video một TỆP
    - objects/, keyframes/: mỗi video một THƯ MỤC con

    Tự chui qua tầng bọc do giải nén sinh ra. Đây là con số bốn máy phải
    so cho khớp ở Task 4 — đếm theo tệp thì objects/ ra hàng trăm nghìn.
    """
    return sorted(_scan(folder, suffix))


def free_space_gb(path: Path) -> float:
    """Dung lượng còn trống của ổ chứa path, tính bằng GB."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024 ** 3)