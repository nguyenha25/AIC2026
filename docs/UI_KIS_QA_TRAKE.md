# Giao diện hợp nhất KIS, Q&A và TRAKE

## Chạy giao diện

Từ `D:\AIC2026`, kích hoạt môi trường rồi chạy:

```powershell
.\.venv\Scripts\Activate.ps1
$env:AIC_DATA_ROOT = "D:/aic-data"
streamlit run src/aic2026/ui/app.py
```

Mở địa chỉ Streamlit in trong terminal, thường là `http://localhost:8501`.

Adapter tự xác định thư mục gốc của repository trước khi nạp
`scripts.run_trake_e2`; không cần tự đặt `PYTHONPATH` khi chạy Streamlit.

## Ba dạng truy vấn

- **KIS:** nhập một mô tả khoảnh khắc, xem keyframe/lân cận/ảnh gốc, chọn thủ
  công hoặc tải CSV Top-100.
- **Q&A:** nhập cảnh cần tìm kèm câu hỏi. UI dùng semantic parser, adaptive K,
  sinh đáp án riêng cho từng frame; có thể bật VLM cho 5 candidate đầu và sửa
  câu trả lời trước khi thêm vào giỏ.
- **TRAKE:** nhập mỗi sự kiện trên một dòng. UI chạy TR-R1 profile-union,
  TR-R2 dual-profile và TR-E2 đã đóng băng; mỗi candidate hiển thị toàn bộ path
  frame và được chọn/xuất thành một dòng.

Mỗi task có giỏ riêng dù dùng cùng mã câu. CSV không có header, bỏ trùng và bị
cắt ở 100 dòng bằng formatter submission dùng chung.

## Kiểm tra trước khi chạy

```powershell
python -m py_compile `
  src/aic2026/ui/app.py `
  src/aic2026/ui/pipelines.py

python -m pytest -q `
  tests/test_ui_pipelines.py `
  tests/test_trake_retrieval_tr_r1.py `
  tests/test_tr_r2.py `
  tests/test_trake_e2_boundary.py `
  tests/test_export_trake_ranked.py
```

## Số liệu kỹ thuật

Nếu có `D:\aic-data\runs\trake_complete_offline_metrics.json`, thanh bên sẽ
hiện `Video Accuracy@1` và `Official-style score`. Đây là số liệu trên tập dev
12 query đã dùng để hiệu chỉnh, không phải blind holdout hay public leaderboard.

## Lưu ý vận hành

- TRAKE cần tối thiểu hai sự kiện, tối đa 12 sự kiện.
- Beam 24 là mặc định cân bằng. Beam 72 chậm hơn nhiều và chỉ nên dùng để cứu
  recall cho query khó.
- VLM Q&A mặc định tắt để tránh tăng latency; OCR/ASR/rule fallback vẫn bảo đảm
  ô đáp án không rỗng.
- Giao diện không sửa thuật toán boundary TR-E2.
