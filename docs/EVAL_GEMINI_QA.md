# EVAL-GEMINI-QA

Bộ đánh giá có hai chế độ candidate tách biệt:

- `r4` (mặc định): đánh giá end-to-end từ QA-R4 đến scorer BTC. Ground truth
  chỉ được mở sau khi sinh submission.
- `oracle_gt`: dùng `gt_video_id` và `gt_frame_range` để chọn keyframe, nhằm đo
  riêng năng lực Reader. **Không truyền `gt_answer` vào Gemini**, nhưng kết quả
  vẫn là `diagnostic_only` và tuyệt đối không được báo cáo như điểm end-to-end.

Script không đụng TRAKE. Ở chế độ `r4`, evaluator giữ nguyên prefix QA-R4 và
không tự chèn frame `n±2`, nên retrieval ceiling phản ánh đúng artifact đầu vào.

## 1. Kiểm tra nhanh không gọi API

```powershell
cd D:\AIC2026
.\.venv\Scripts\Activate.ps1

python -X utf8 -u -m scripts.eval_gemini_qa `
  --dev D:\aic-data\dev\dev_questions.jsonl `
  --r4 D:\aic-data\runs\qa_r4_adaptive_candidates.jsonl `
  --providers none `
  --limit 1
```

`none` là reader OCR/ASR local. Lệnh này kiểm tra schema dev, contract QA-R4,
submission và scorer mà không gọi Gemini.

## 2. Chẩn đoán Reader bằng oracle GT

Đây là bước nên chạy trước khi sửa retrieval. Nó trả lời câu hỏi: *nếu đã đưa
đúng đoạn video cho Gemini, Reader có trả lời đúng không?*

```powershell
python -X utf8 -u -m scripts.eval_gemini_qa `
  --dev "D:\aic-data\dev\dev_questions.jsonl" `
  --frame-map "D:\aic-data\index\frame_map.parquet" `
  --candidate-mode oracle_gt `
  --partition tune `
  --holdout-size 6 `
  --split-seed aic2026-qa-v1 `
  --providers none gemini gemini_rerank `
  --max-images 4 8 12 `
  --delay-seconds 0.5
```

`--r4` không được đọc trong chế độ này. Oracle ưu tiên keyframe nằm trong đoạn
GT và sắp prefix theo độ phủ giữa/đầu/cuối. Nếu đoạn GT không chứa keyframe đã
trích, script dùng keyframe ngữ cảnh gần nhất nhưng `frame_hit` vẫn là `false`.
Chỉ các row Gemini thực sự nhìn thấy mới được tính cho cấu hình Gemini; câu trả
lời local trên các row còn lại không được phép làm tăng điểm oracle.

Đọc `report/ablation.csv`, chọn cấu hình có `mean_final_score` tốt, không
fallback và latency chấp nhận được. Mục tiêu ban đầu: oracle score từ `0.75`.

## 3. Tune end-to-end trên 12 câu

```powershell
python -X utf8 -u -m scripts.eval_gemini_qa `
  --dev D:\aic-data\dev\dev_questions.jsonl `
  --r4 D:\aic-data\runs\qa_r4_adaptive_candidates.jsonl `
  --candidate-mode r4 `
  --partition tune `
  --holdout-size 6 `
  --split-seed aic2026-qa-v1 `
  --providers none gemini gemini_rerank `
  --max-images 4 8 12 `
  --delay-seconds 0.5
```

`gemini` và `gemini_rerank` dùng chung một API response ở cùng mức số ảnh;
vì vậy bật cả hai không nhân đôi số API call. Mặc định mỗi run có cache riêng
để latency và số call là cold-start, không bị cache cũ làm đẹp số liệu.

Chỉ chạy bước này sau khi đã chốt Reader bằng oracle. Nếu `frame_recall` còn
thấp, ưu tiên sửa retrieval thay vì tiếp tục đổi prompt Gemini.

## 4. Chấm holdout sau khi khóa cấu hình

Ví dụ tune chọn `Gemini 12 ảnh + rerank`:

```powershell
python -X utf8 -u -m scripts.eval_gemini_qa `
  --dev D:\aic-data\dev\dev_questions.jsonl `
  --r4 D:\aic-data\runs\qa_r4_adaptive_candidates.jsonl `
  --candidate-mode r4 `
  --partition holdout `
  --holdout-size 6 `
  --split-seed aic2026-qa-v1 `
  --providers none gemini_rerank `
  --max-images 12 `
  --delay-seconds 0.5
```

Không đổi `holdout-size` hoặc `split-seed` giữa hai lệnh. Không xem holdout để
sửa prompt rồi chấm lại; nếu làm vậy holdout đã trở thành tune.

## 5. Output

```text
D:\aic-data\runs\<timestamp>_EVAL-GEMINI-QA_<candidate_mode>_<partition>\
├── manifest.json
├── cache\
├── submissions\
│   └── <config_id>\query-<id>-qa.csv
└── report\
    ├── per_query.jsonl
    ├── failures.jsonl     # bản gọn, không chứa mảng answers
    ├── fallbacks.csv      # quota/JSON/ảnh/API
    ├── ablation.csv
    ├── by_intent.csv
    └── summary.json
```

Đọc `report/ablation.csv` trước:

- `candidate_mode`, `diagnostic_only`: ngăn nhầm oracle với điểm end-to-end.
- `mean_final_score`: điểm trung bình theo công thức BTC; chỉ là end-to-end khi
  `candidate_mode=r4`.
- `mean_final_score_all_status_ok`: số audit cũ có thể chứa fallback và oracle
  gap; không dùng để chọn cấu hình.
- `gemini_valid_queries`: số câu thực sự được Gemini trả lời, không tính local
  fallback.
- `metric_queries`: mẫu thật sự đi vào điểm trung bình sau khi loại fallback và
  `oracle_keyframe_gap`.
- `excluded_oracle_gap_queries`: số câu không có keyframe trích sẵn nằm trong
  khoảng frame GT.
- `evaluation_valid`: chỉ true khi không có query lỗi/fallback và còn mẫu chấm.
- `mean_retrieval_ceiling`: trần điểm nếu giữ ranking nhưng điền GT answer.
- `mean_reader_loss`: phần điểm reader làm mất so với retrieval ceiling.
- `mean_delta_vs_none`: chênh lệch so với OCR/ASR local trên đúng câu đó.
- `video_recall`, `frame_recall`: retrieval có đưa đúng video/frame vào pool.
- `answer_accuracy_given_gt_frame`: độ đúng của reader khi đã có frame GT.
- `fallback_queries`, `json_retry_queries`: độ ổn định Gemini.
- `latency_p95_ms`, `api_calls`: chi phí vận hành.

Đọc `report/failures.jsonl` để sửa đúng tầng:

- `retrieval_video_miss`: chưa có video đúng, không sửa prompt Gemini.
- `retrieval_frame_miss`: có video đúng nhưng chưa có frame trong GT.
- `reader_miss`: đã có frame đúng nhưng câu trả lời sai.
- `provider_fallback=true`: lỗi API/JSON/quota, xem `fallback_reason`.
- `oracle_keyframe_gap`: frame map không có keyframe trong khoảng GT; đây là lỗi
  coverage dữ liệu, không quy kết cho Reader.

Không mở thẳng `failures.jsonl` bằng PowerShell nếu chỉ cần thống kê. Dùng:

```powershell
Import-Csv "$($run.FullName)\report\fallbacks.csv" |
  Select-Object query_id, config_id, provider_failure_category |
  Format-Table -AutoSize
```

## 6. Chạy một câu cụ thể

```powershell
python -X utf8 -u -m scripts.eval_gemini_qa `
  --partition all `
  --query-id 7 `
  --providers none gemini gemini_rerank `
  --max-images 12 `
  --delay-seconds 0
```

Có thể lặp `--query-id` để chạy một nhóm câu. `--variants-per-frame` mặc định
là 1 để so sánh reader rõ ràng; chỉ tăng sau khi đã đo riêng chiến thuật rải
biến thể đáp án.

## 7. Quality gate khuyến nghị

Quality gate cho Reader trên `oracle_gt`:

1. `mean_final_score >= 0.75` trên tune.
2. Không có fallback hoặc query lỗi.
3. Chọn số ảnh nhỏ nhất có điểm gần cấu hình tốt nhất (chênh không quá `0.02`).
4. Khóa prompt/model/số ảnh trước khi xem holdout.

Quality gate cho hệ thống trên `r4`:

Chỉ thay cấu hình thi khi holdout đồng thời thỏa:

1. `mean_final_score` cao hơn baseline `none`.
2. Không giảm `frame_recall`.
3. `fallback_queries = 0`.
4. `json_retry_queries` thấp và không có query lỗi.
5. Latency p95 nằm trong ngân sách vận hành của đội.
