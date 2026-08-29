# N-00 — đóng băng baseline sạch và tập dev

Owner: **Ngân**  
Deadline: **20:00 ngày 29/08/2026**  
Trạng thái hoàn tất chỉ khi lệnh chính in `N-00 ĐẠT`.

## N-00 giải quyết việc gì?

Một lần chạy tạo mốc tham chiếu không sửa được bằng cách:

- kiểm schema toàn bộ `dev_questions.jsonl`;
- chỉ giữ câu có evidence GT thật trên máy;
- chia `dev_tune/dev_holdout` theo `video_id`, không leakage;
- kiểm cấu hình offline-only, từ chối API key và nguồn `llm`;
- chụp snapshot `settings.yaml`, `rrf_weights.yaml`, `n00_baseline.yaml`;
- chạy baseline Q&A local trên `dev_holdout` và ghi manifest có SHA-256.

Q&A/KIS chỉ được gọi là sạch nếu có ảnh keyframe thật trong cửa sổ GT. TRAKE
chỉ sạch khi `frames_dense` có manifest và mỗi event có ít nhất một frame trong
cửa sổ GT. Câu thiếu dữ liệu được ghi vào `dev_excluded.jsonl`, không bị bỏ âm
thầm và không được tính vào baseline.

## Chạy chính thức

Đứng tại `D:\AIC2026`, đã kích hoạt `.venv` và đặt `DATA_ROOT` đúng:

```powershell
python -u -m scripts.dong_bang_n00
```

Script dùng đúng baseline Phase 2 đã khóa:

- QA retrieval: `clip_l=1.0`, các nguồn khác bằng 0;
- `K_source=500`;
- CLIP-L dùng Marian local; reader hiện tại đọc tối đa 5 ứng viên;
- không gọi API và xóa biến API khỏi tiến trình baseline.

Output mong đợi:

```text
Video leakage: 0
API key      : không dùng
Baseline QA   : trần=..., thật=..., khớp=.../...
=> N-00 ĐẠT. Có thể dùng manifest này làm mốc so sánh Gate 1-5.
```

Run được ghi tại:

```text
D:\aic-data\runs\n00_baseline_20260829\
├── manifest.json
├── dev_clean.jsonl
├── dev_tune.jsonl
├── dev_holdout.jsonl
├── dev_excluded.jsonl
├── baseline_qa.json
└── config\
    ├── n00_baseline.yaml
    ├── settings.yaml
    └── rrf_weights.yaml
```

Không sửa các file trong run sau khi tạo. Muốn thử lại phải đổi `--run-id`.

## Xem trước mà chưa chạy model

```powershell
python -u -m scripts.dong_bang_n00 `
  --chi-chuan-bi `
  --run-id n00_preview
```

Lệnh này chỉ dùng để kiểm split/evidence. Dòng cuối phải là `N-00 CHƯA ĐÓNG`;
không được dùng preview làm baseline.

## Điều kiện dừng

Script dừng và không tuyên bố Gate 0 đạt nếu gặp một trong các trường hợp:

- còn biến `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` hoặc
  `GEMINI_API_KEY` trong môi trường;
- `settings.yaml` dùng nguồn `llm` hoặc trọng số QA khác cấu hình N-00;
- tập sạch rỗng, holdout không có câu QA hoặc split rò rỉ `video_id`;
- báo cáo `do_qa.json` không khớp số câu QA trong holdout;
- baseline local chạy lỗi.

## Test trước khi commit

```powershell
pytest -q tests/test_n00_dong_bang.py
```

Kỳ vọng: `7 passed`.
