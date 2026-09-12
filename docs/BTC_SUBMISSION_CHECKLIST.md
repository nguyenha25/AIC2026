# Checklist nộp mã nguồn và báo cáo kỹ thuật BTC

## 1. Kiểm thử trước khi đóng gói

```powershell
cd D:\AIC2026
.\.venv\Scripts\Activate.ps1
python -m pytest -q tests\test_gemini_qa.py tests\test_eval_gemini_qa.py tests\test_package_btc_submission.py
```

Không gọi Gemini trong bước này. Unit test dùng transport giả và dữ liệu tạm.

## 2. Tạo ZIP mã nguồn từ repo mới nhất

```powershell
python -X utf8 -u -m scripts.package_btc_submission `
  --root "D:\AIC2026" `
  --output "D:\AIC2026-BTC-source-code.zip"
```

Script tự loại `.env`, API key, `.venv`, cache Python, `runs`, dữ liệu/index mô
hình và các ZIP cũ. Nếu phát hiện chuỗi giống API key nằm trong source, quá
trình đóng gói dừng thay vì tạo một gói không an toàn. ZIP chứa
`SUBMISSION_MANIFEST.json` với SHA-256 của từng file.

## 3. Kiểm tra ZIP

```powershell
Get-FileHash "D:\AIC2026-BTC-source-code.zip" -Algorithm SHA256

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead("D:\AIC2026-BTC-source-code.zip")
$zip.Entries |
  Where-Object { $_.FullName -match '(^|/)(\.env|runs|__pycache__)(/|$)' } |
  Select-Object FullName
$zip.Dispose()
```

Lệnh kiểm tra thứ hai phải không trả dòng nào. `.env.example` được phép xuất
hiện vì chỉ chứa placeholder.

## 4. Báo cáo kỹ thuật

Chèn nội dung `docs/BTC_TECHNICAL_QA_GEMINI.tex` vào phần giải pháp Q&A của báo
cáo LaTeX. Nội dung phân biệt rõ:

- `r4`: đánh giá end-to-end;
- `oracle_gt`: chẩn đoán reader, không phải điểm hệ thống;
- request API fallback và keyframe gap không được tính vào điểm Gemini;
- kết quả Gemini hiện là sơ bộ do quota, không tuyên bố quá mức.

## 5. Không nộp

- `.env` hoặc ảnh chụp có API key;
- thư mục `.venv`, `runs`, cache và model/index dung lượng lớn;
- dữ liệu DEV có ground truth nếu BTC không yêu cầu;
- báo cáo oracle như kết quả end-to-end;
- file submission dùng `n` thay cho `frame_idx`.
