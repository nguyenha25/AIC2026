import os
from pathlib import Path
from dotenv import load_dotenv

# Nạp cấu hình từ tệp .env (nằm ở thư mục gốc)
load_dotenv()

# Lấy đường dẫn gốc dữ liệu từ .env
data_root_env = os.getenv("DATA_ROOT")
if not data_root_env:
    raise ValueError("Chưa khai báo DATA_ROOT trong tệp .env!")

# Định nghĩa các đường dẫn chuẩn bằng pathlib
DATA_ROOT = Path(data_root_env)

# 3 tầng dữ liệu cốt lõi
RAW_DIR = DATA_ROOT / "raw"
DERIVED_DIR = DATA_ROOT / "derived"
INDEX_DIR = DATA_ROOT / "index"

# Các thư mục con bên trong gốc dự án (nằm trên GitHub)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"