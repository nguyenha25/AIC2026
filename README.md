# AIC 2026 — YOLO

Kho mã nguồn của nhóm. **Chỉ chứa chương trình, không chứa dữ liệu.**
Dữ liệu nằm ở một thư mục riêng, khai báo trong `.env` của từng máy.

---

## Cài lần đầu (làm một lần)

```powershell
git clone <link kho>
cd AIC2026

copy .env.example .env          # rồi MỞ RA SỬA đường dẫn của máy mình
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m scripts.bootstrap_dirs
python -m scripts.verify_layout
pytest
```

Hai dòng cuối phải ra `KẾT LUẬN: ĐẠT` và `8 passed`.

> **Windows:** trong `.env` viết gạch chéo **xuôi** — `D:/aic-data`, không phải `D:\aic-data`.
> Nếu `Activate.ps1` báo không được phép chạy script:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
>
> Mỗi lần mở terminal mới là phải `.\.venv\Scripts\Activate.ps1` lại.
> Dấu `(.venv)` phải hiện ở đầu dòng lệnh. Tắt bằng `deactivate`.

---

# BỘ LỆNH KIỂM — GIAI ĐOẠN 0

Mỗi người chạy đủ trên máy mình rồi chụp màn hình gửi nhóm chat.

## Task 1 — Chốt cấu trúc thư mục

Không có lệnh máy. Kiểm bằng mắt sheet *Cấu trúc thư mục*: 5 câu hỏi mở đã điền
cột "Nhóm chọn" chưa. Câu 1 (để dữ liệu ở đâu) bắt buộc chốt ngay; bốn câu sau
hoãn được sang Giai đoạn 1.

## Task 2 — Ba lệnh áp dụng cấu trúc

```powershell
python -m scripts.verify_layout
```

**Kiểm:** máy này có đúng chuẩn nhóm không — đã khai `.env`, đủ 20 thư mục, dữ
liệu không nằm nhầm trong thư mục mã nguồn.
**Đạt:** `KẾT LUẬN: ĐẠT`.

## Task 3 — Cấu hình máy

```powershell
Get-PSDrive C, D | Select-Object Name, @{n='Trống(GB)';e={[math]::Round($_.Free/1GB,1)}}
```

**Kiểm:** còn đủ chỗ cho Giai đoạn 1 không.
**Đạt:** trên 30 GB. Dưới 10 GB là phải dọn ổ trước khi làm tiếp.

## Task 4 — Tải và giải nén

Mã kiểm tra tính **khi còn là .zip** (giải nén rồi không tính lại được):

```powershell
certutil -hashfile map-keyframes-aic25-b1.zip SHA256
certutil -hashfile clip-features-32-aic25-b1.zip SHA256
certutil -hashfile objects-aic25-b1.zip SHA256
certutil -hashfile media-info-aic25-b1.zip SHA256
```

Giải nén, xóa .zip, rồi:

```powershell
python -m scripts.verify_layout
```

**Kiểm:** bốn máy có cầm cùng một phiên bản dữ liệu không, và có ai tải thiếu không.
**Đạt:** mã kiểm tra khớp cột trong sheet (so 8 ký tự đầu là đủ) + cả bốn máy ra
**873 video** + không còn tệp `.zip` nào.

> Giải nén ra cấu trúc lồng (`map-keyframes-aic25-b1\map-keyframes\...`) là bình
> thường — đó là cách BTC đóng gói. **Không sắp lại bằng tay.** Chương trình tự
> chui qua các tầng bọc.

## Task 5 — Hiểu bốn cột

```powershell
python -m scripts.peek_data L21_V001
python -m scripts.check_media_info
```

**Kiểm:** cả nhóm hiểu đúng `n` (số thứ tự ảnh) khác `frame_idx` (số đem nộp);
biết trước bao nhiêu video thiếu media-info để code chịu được chỗ khuyết.
**Đạt:** `[ĐẠT] frame_idx khớp với pts_time × fps` + có con số cụ thể ghi vào sheet.

## Task 6 — Đồng bộ môi trường

```powershell
python -c "import numpy, pandas, yaml, dotenv; print('ok')"
pytest
```

**Kiểm:** bốn máy chạy cùng phiên bản thư viện, bộ khung không lỗi.
**Đạt:** `ok` và `8 passed` trên cả bốn máy.

So phiên bản giữa bốn người:

```powershell
python -m pip freeze | Sort-Object > pip_may_toi.txt
```

## Task 7 — Dữ liệu thật

Xem máy mình có video nào:

```powershell
python -c "from src.aic2026.frame_map import available_video_ids as f; print(f()[:20])"
```

Mỗi người chọn **một video khác nhau**:

```powershell
python -m scripts.peek_data L21_V007 --rows 5
```

**Kiểm:** chương trình đọc được dữ liệu thật mà không ai phải sửa đường dẫn
riêng của máy mình vào code.
**Đạt:** chạy được nguyên trạng, cột `lệch` bằng 0 hoặc 1 ở cả 20 dòng của bốn máy.

## Task 8 — Mẫu bốn nhánh và tệp nộp giả

```powershell
python -m scripts.make_dummy_submission
Get-Content docs\schema\README.md -Encoding UTF8
```

**Kiểm:** nhóm hiểu đúng định dạng nộp bài ngay từ đầu, không đợi sát hạn mới
phát hiện sai; mỗi người đọc được mẫu dữ liệu của ba người kia.
**Đạt:** ba tệp `[ĐẠT]`, mỗi tệp 100 dòng, `trùng 0`.

> **Luôn thêm `-Encoding UTF8` khi dùng `Get-Content`.** Windows PowerShell 5.1
> mặc định đọc bằng bảng mã hệ thống, tiếng Việt sẽ hiện thành `Ä'Ã¡p Ã¡n`.
> Tệp không hỏng — chỉ màn hình hiển thị sai. Đừng ghi đè lại.

## Task 9 — Đóng giai đoạn

```powershell
python -m scripts.check_phase0_gates
```

**Kiểm:** soát sáu điều kiện đóng Giai đoạn 0 trong một lệnh.
**Đạt:** `SÁU ĐIỀU KIỆN ĐỀU ĐẠT`.

---

## Hai phép kiểm dữ liệu bổ sung

Bắt buộc trước khi sang Giai đoạn 1. Điều kiện 3 trong `check_phase0_gates` chỉ
quét 20 video mẫu — hai lệnh này quét đủ 873.

### Độ lệch trên toàn kho

```powershell
python -c "from src.aic2026.frame_map import FrameMap, available_video_ids as f; w=0; n=0
for v in f():
    d=FrameMap.load(v).max_drift(); n+=1; w=max(w,d)
print(f'da kiem {n} video, lech lon nhat {w}')"
```

**Kiểm:** `frame_idx` khớp `pts_time × fps` trên **toàn bộ** kho, không chỉ mẫu.
**Đạt:** `da kiem 873 video, lech lon nhat 1`.

### Đặc trưng CLIP khớp bảng đối chiếu

```powershell
python -c "import numpy as np
from src.aic2026.frame_map import FrameMap, available_video_ids as f
from src.aic2026.paths import clip_features_file
bad=[]; tong=0
for v in f():
    a=np.load(clip_features_file(v), mmap_mode='r'); tong+=a.shape[0]
    if a.shape[0]!=len(FrameMap.load(v)): bad.append(v)
print(f'tong vector {tong}'); print('LECH:', bad[:10] if bad else 'khop hoan toan')"
```

**Kiểm:** số vector CLIP có bằng số dòng map-keyframes không. Lệch nghĩa là có
video giải nén thiếu — phát hiện muộn thì phải dựng lại index từ đầu.
**Đạt:** `tong vector 177321` và `khop hoan toan`.

---

## Ba việc người, máy không kiểm được

- Bốn máy dán số **873** vào nhóm chat, phải giống hệt nhau
- Bốn ảnh chụp `peek_data` của **bốn video khác nhau**, đủ 20 dòng lệch ≤ 1
- Chốt bảng phân công bốn nhánh: `index/` · `enrich/` · `rank/` · `submit/`+`eval/`

---

# ĐIỀU DỄ SAI NHẤT TRONG CẢ DỰ ÁN

Trong `raw/map-keyframes/L21_V001.csv` có hai con số **hoàn toàn khác nhau**:

- `n` — tấm ảnh thứ mấy. Ảnh `0005.jpg` thì `n = 5`.
- `frame_idx` — **vị trí khung hình trong video. SỐ NÀY MỚI ĐEM NỘP.**

Ảnh thứ 5 của một video 30 hình/giây có thể nằm ở khung hình thứ 411. Nộp nhầm
số 5 thì BTC chấm 0 điểm, mà kết quả tìm kiếm trên màn hình vẫn trông rất đúng —
không ai phát hiện ra.

Luôn lấy số nộp bài qua `FrameMap.frame_idx_of(n)`, đừng bao giờ dùng thẳng `n`.

---

# SỐ LIỆU KHO DỮ LIỆU (batch 1, đã đo)

| Chỉ số | Giá trị |
|---|---|
| Số video | 873 (L21_V001 → L30_V096) |
| Tổng keyframe | 177.321 (~203 ảnh/video) |
| Vector CLIP | 177.321 × 512, kiểu **float16** |
| Trên đĩa | ~173 MB |
| Trong RAM khi dựng FAISS (float32) | ~346 MB |
| Lệch `frame_idx` lớn nhất | 1, trên đủ 873 video |
| Video thiếu media-info | 0 (batch 1 đủ; batch 2 sẽ có chỗ khuyết) |
| Trùng `frame_idx` | 604 ca (0,34%) |

## Ba điều rút ra cho Giai đoạn 1

**Đặc trưng là `float16`.** FAISS chỉ nhận `float32` liên tục trong bộ nhớ —
phải `.astype('float32')` trước khi nạp, nếu không lỗi ngay.

**Có `frame_idx` trùng nhau.** 604 ca, phần lớn là cặp `n` liền nhau trỏ cùng một
vị trí khung hình (BTC trích hai ảnh cách nhau chưa tới 1/30 giây). Nhánh `rank/`
phải lọc trùng theo `(video_id, frame_idx)` — **lọc theo `n` sẽ để lọt**, và nộp
hai dòng y hệt nhau là phí một suất trong 100.

**346 MB nằm gọn trong RAM cả bốn máy**, kể cả máy 8 GB. Index phẳng không nén là
đủ, chưa cần IVF hay PQ. Đúng tinh thần *baseline trước, nâng cấp sau*.

---

# QUY ƯỚC

## Ba điều cấm

1. **Không ghi gì vào `raw/`** sau khi đã giải nén xong. Cần biến đổi thì đọc ra,
   xử lý, rồi ghi sang `derived/`.
2. **Không viết đường dẫn riêng của máy mình vào chương trình.** Mọi đường dẫn
   lấy từ `src/aic2026/paths.py`.
3. **Không đưa `.env`, gốc dữ liệu, hay `index/` lên github.**

## Bảy quy tắc đặt tên

1. Tên video giữ nguyên chữ hoa như BTC đặt: `L21_V001`.
2. Số thứ tự ảnh giữ đủ bốn chữ số: `0000`, `0047`.
3. Mọi tệp trong `derived/` đặt tên theo video, một tệp một video.
4. Dữ liệu nhiều dòng dùng `.jsonl` — mỗi dòng một bản ghi.
5. Thư mục `runs/` đặt tên theo thời gian và cấu hình: `2026-08-15_1430_clipb32`.
6. Không dấu tiếng Việt, không khoảng trắng trong mọi tên tệp và tên thư mục.
7. Tệp đang viết dở mang đuôi `.partial`, viết xong mới đổi tên.

---

# CẤU TRÚC

```
AIC2026/                     GỐC CHƯƠNG TRÌNH (lên github)
├── .env                     đường dẫn dữ liệu của riêng máy này — KHÔNG lên github
├── .env.example             tệp mẫu để chép ra .env
├── requirements.txt         phiên bản thư viện, Ngân chốt
├── config/
│   ├── settings.yaml        tham số chạy
│   └── shards.yaml          ai giữ phần dữ liệu nặng nào
├── src/aic2026/
│   ├── paths.py             NƠI DUY NHẤT biết đường dẫn thật trên đĩa
│   ├── frame_map.py         đọc bảng đối chiếu ảnh <-> vị trí khung hình
│   ├── index/               nhánh 1: dựng và truy vấn kho tra cứu
│   ├── enrich/              nhánh 2: đọc chữ trên ảnh, chuyển lời nói thành chữ
│   ├── rank/                nhánh 3: xếp hạng, gộp kết quả, lọc trùng
│   ├── submit/              nhánh 4a: xuất tệp nộp
│   └── eval/                nhánh 4b: tự chấm điểm
├── scripts/                 các lệnh chạy tay
├── tests/                   bộ kiểm tra tự động, 8 mục
└── docs/
    ├── layout.md            tài liệu cấu trúc thư mục đã chốt
    ├── decisions/           mỗi quyết định lớn một tệp ngắn
    └── schema/              MẪU DỮ LIỆU BỐN NHÁNH  <-- đọc trước khi code
```

Gốc dữ liệu (`raw/`, `derived/`, `index/`, `dev/`, `runs/`, `submissions/`) nằm ở
chỗ khác, xem `docs/layout.md`.

## Bảng lệnh nhanh

| Lệnh | Làm gì | Task |
|---|---|---|
| `python -m scripts.bootstrap_dirs` | Tạo toàn bộ thư mục rỗng | 2 |
| `python -m scripts.verify_layout` | Kiểm chuẩn, đếm số video | 2, 4 |
| `python -m scripts.peek_data <video>` | Đọc dữ liệu thật, kiểm phép tính | 5, 7 |
| `python -m scripts.check_media_info` | Đếm video thiếu media-info | 5 |
| `python -m scripts.make_dummy_submission` | Tệp nộp giả 100 dòng | 8 |
| `python -m scripts.check_phase0_gates` | Soát sáu điều kiện | 9 |
| `pytest` | Bộ kiểm tra tự động, xem chi tiết 8 mục | 6 |
| `pytest -q` | Như trên, chỉ xem pass hay không | 6 |

Mọi lệnh chạy **từ thư mục gốc dự án** và dùng `python -m`, không phải
`python scripts/xxx.py` — chạy kiểu sau sẽ không import được `src`.