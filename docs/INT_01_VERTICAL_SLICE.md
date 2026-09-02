# INT-01 — Review Q&A vertical slice

## Mục tiêu

Kiểm đúng một luồng Q&A qua ba bàn giao độc lập:

`QA-S1 QueryPlan -> QA-R1 reader candidates -> QA-V2 reader -> EvidenceAnswer`

Sau khi owner cho phép sửa contract, vertical slice dùng các handoff sau:

- QA-R1 gọi trực tiếp parser QA-S1 cho query Q&A, áp dụng prior CLIP-L và
  `semantic_k_hint`, rồi ghi routing đã dùng vào từng query record. Sàn trọng
  số/pool baseline được giữ để contract mới không tự làm giảm recall gate.
  Recall@300 chính thức dùng ranking baseline đã khóa; routed ranking chỉ cấp
  candidate cho top-50/top-12.
- CandidateRecord xuất `semantic_coverage`, `window` và provenance.
- QA-V2 xuất confidence bằng geometric mean xác suất token sinh ra. Đây là
  model confidence chưa hiệu chuẩn; `confidence_method` luôn được lưu kèm.

## Chạy trên máy có dữ liệu và Qwen trong cache

Trước hết tạo lại QA-R1 profile bằng contract mới trong môi trường chính:

```powershell
cd D:\AIC2026
.\.venv\Scripts\Activate.ps1

python -u -m scripts.candidate_funnel
```

Sau đó kích hoạt môi trường Qwen và chạy một query `hoi_dap` có ảnh candidate
local:

```powershell
cd D:\AIC2026
.\.venv-qwen\Scripts\Activate.ps1

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

python -u -m scripts.int_01_vertical_slice `
  --query-id 12 `
  --qa-r1-profile D:\aic-data\runs\dev_qa_r1_profile.json `
  --fail-on-blocker
```

Report được tạo tại:

`D:\aic-data\runs\<timestamp>_INT-01\report\int_01_vertical_slice.json`

Nếu Qwen chưa chạy được, audit riêng hai handoff đầu bằng:

```powershell
python -u -m scripts.int_01_vertical_slice `
  --query-id 12 `
  --qa-r1-profile D:\aic-data\runs\dev_qa_r1_profile.json `
  --reader skip
```

`--reader skip` chỉ dùng để tìm blocker contract, không tính là end-to-end pass.

## Cách đọc kết quả

- `status=pass`: model đã chạy, không còn issue P0 và `EvidenceAnswer` hợp lệ.
- `status=reviewed_with_blockers`: INT-01 đã review xong nhưng còn blocker cần
  trả về owner.
- `issue_summary.by_owner`: danh sách lỗi đã nhóm theo Nghi, Thi, Nguyên hoặc
  Ngân.
- `acceptance`: năm điều kiện machine-checkable của vertical slice.

Chạy với `--fail-on-blocker` để process trả exit code `2` khi còn P0. Report vẫn
được ghi trước khi thoát.

## Các chênh lệch contract INT-01 đang kiểm

1. QA-S1 phải xuất QueryPlan schema 1.1, modality weights hợp lệ và giữ
   `query_id`.
2. QA-R1 phải chứng minh đã nhận/applied QueryPlan, không chỉ ghép theo
   `query_id`; CandidateRecord phải giữ `video_id`, `n`, `frame_idx`,
   `semantic_coverage`, `window` và provenance.
3. QA-V2 phải trả answer có confidence và `confidence_method` để tạo
   EvidenceAnswer. API `hoi()` trả chuỗi cũ vẫn được giữ để tương thích ngược;
   INT-01 dùng API có cấu trúc `hoi_co_confidence()`.

## Kiểm thử

```powershell
python -m pytest .\tests\test_int_01_vertical_slice.py -v
python -m pytest .\tests\test_n01_semantic_parser.py -v
python -m pytest .\tests\test_qa_r1_candidate_funnel.py -v
python -m pytest .\tests\test_qwen_oracle.py -v
```

Test INT-01 dùng fake reader có cấu trúc, không tải model và không cần dữ liệu
AIC. Smoke thật vẫn phải chạy bằng lệnh ở phần đầu.
