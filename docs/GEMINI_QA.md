# Gemini Cloud cho Q&A

Nhánh này chỉ gửi một nhóm keyframe đã được retrieval cục bộ chọn ra. CLIP-L,
OCR, ASR, KIS và TRAKE không thay đổi. Nếu Gemini lỗi mạng, timeout hoặc hết
quota, UI tự dùng đáp án OCR/ASR local nên không tạo ô trả lời rỗng.

## 1. Khai báo API key

Mở tệp `.env` ở gốc repository và thêm các dòng riêng biệt:

```env
GEMINI_API_KEY=dan-api-key-cua-ban-vao-day
AIC_GEMINI_MODEL=gemini-2.5-flash
AIC_GEMINI_MAX_IMAGES=12
AIC_GEMINI_TIMEOUT_SECONDS=25
AIC_GEMINI_RETRIES=2
```

Không gửi API key cho người khác và không commit `.env`. Tệp `.env` đã nằm
trong `.gitignore`. Không đặt key trong `app.py` hoặc `gemini_qa.py`.

Nếu dùng PowerShell để thêm dòng, kiểm tra `.env` có xuống dòng đúng:

```powershell
Get-Content .env
```

Phải thấy `DATA_ROOT=...` và `GEMINI_API_KEY=...` ở hai dòng khác nhau.

## 2. Kiểm tra kết nối

Không cần cài thêm Google SDK. Mã dùng REST API và thư viện chuẩn của Python.

```powershell
.\.venv\Scripts\Activate.ps1
python -u -m scripts.check_gemini_api
```

Kết quả hợp lệ bắt đầu bằng `OK: model=gemini-2.5-flash`. Lệnh này dùng đúng
một request văn bản nhỏ và không in API key.

## 3. Chạy UI

```powershell
streamlit run src/aic2026/ui/app.py
```

Trong `Tùy chọn tìm kiếm` của Q&A:

- `Gemini Cloud`: chế độ khuyến nghị. Sinh đáp án nhưng giữ ranking retrieval.
- `Gemini Cloud + rerank (thử nghiệm)`: dùng relevance của Gemini để đổi thứ tự
  các frame đã đánh giá. Chỉ dùng sau khi đo tốt hơn trên holdout.
- `BLIP local`: reader cũ, không cần mạng nhưng nặng trên CPU.
- `Chỉ OCR/ASR local`: không dùng VLM.

Mỗi truy vấn Gemini dùng tối đa một API call cho tối đa 12 ảnh: tối đa ba
keyframe ở mỗi trong bốn video đứng đầu. OCR và ASR quanh từng frame được gửi
kèm nhưng đáp án vẫn gắn với chính frame đó.

## 4. Cache và fallback

Kết quả Gemini mặc định được cache tại:

```text
D:/aic-data/runs/gemini_qa_cache
```

Có thể đổi bằng `AIC_GEMINI_CACHE_DIR`. Cache không chứa API key. Cùng câu hỏi,
model và các ảnh không đổi sẽ không gọi API lần nữa.

Các lỗi sau tự fallback về local:

- thiếu `GEMINI_API_KEY`;
- HTTP 429 do quota;
- HTTP 5xx;
- timeout hoặc mất mạng;
- phản hồi JSON không hợp lệ;
- keyframe gốc chưa được tải.

Chẩn đoán Q&A trên UI hiển thị model, số ảnh, cache hit/miss, số API call và lý
do fallback.

## 5. Kiểm thử trước khi thi

```powershell
python -m pytest -q tests/test_gemini_qa.py tests/test_ui_pipelines.py
python -m pytest -q
```

Nên so sánh ba cấu hình trên cùng holdout: local, Gemini answer-only, Gemini
answer + rerank. Chỉ bật rerank trong thi khi điểm BTC end-to-end cao hơn; điểm
answer tốt nhưng retrieval ceiling thấp thì phải tiếp tục sửa retrieval.
