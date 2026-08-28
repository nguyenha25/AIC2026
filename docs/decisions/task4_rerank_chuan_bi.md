# Task 4 — chuẩn bị rerank top-100

## Trạng thái hiện tại

Phần triển khai đã sẵn sàng trước khi có dev v2:

- CLIP B/32 và `clip-features-32` vẫn là tầng tìm thô;
- chỉ top-100 ảnh được mã hoá lại bằng `ViT-L-14/laion2b_s32b_b82k`;
- ảnh được mã hoá theo batch, mặc định 8, không gọi mô hình 100 lần rời rạc;
- cache vector ảnh được dùng lại giữa các câu hỏi;
- ảnh thiếu không bị xoá âm thầm và được ghi vào báo cáo;
- có hai cách xếp: `thay` và `rrf`, chưa chốt trước khi đo dev v2;
- có phép nghiệm thu hai lượt độc lập sau khi xoá cache;
- có báo cáo JSON chứa baseline, hai điểm rerank, delta và fingerprint.

Chưa thể hoàn tất duy nhất phần **“điểm tăng trên dev v2”**, vì bộ dev v2 chưa
có. Không dùng bộ dev nhỏ hiện tại để chốt mô hình hoặc `cach_gop`.

## Kiểm tra không cần dev v2

```powershell
python -m pytest -q tests/test_rerank.py
python -m compileall -q src scripts tests
```

Nếu muốn thử đường chạy thật trên 1–2 câu của bộ dev cũ:

```powershell
python -u -m scripts.benchmark_rerank --so-cau 2
```

Kết quả này chỉ là smoke-test tải mô hình, đọc ảnh và chạy hết pipeline; không
được ghi là điểm nghiệm thu.

## Lệnh chạy khi có dev v2

Chạy hai biến thể độc lập:

```powershell
python -u -m scripts.benchmark_rerank `
  --tep D:\aic-data\dev\dev_v2.jsonl `
  --cach-gop thay

python -u -m scripts.benchmark_rerank `
  --tep D:\aic-data\dev\dev_v2.jsonl `
  --cach-gop rrf
```

Nếu Task 5 đã chốt nguồn mở rộng truy vấn, dùng **cùng một câu tiếng Anh** cho
cả tầng B/32 và tầng rerank:

```powershell
python -u -m scripts.benchmark_rerank `
  --tep D:\aic-data\dev\dev_v2.jsonl `
  --cach-gop thay `
  --nguon-mo-rong tu_dien
```

Không bật `--clip-l`: đó là nhánh chỉ mục độc lập của Task 8, không phải phép
đo riêng đóng góp của Task 4.

## Điều kiện chốt Task 4

Một báo cáo chỉ được dùng để nghiệm thu khi đồng thời thỏa:

1. `so_cau_kis_qa >= 40` và chạy toàn bộ dev v2;
2. `so_thieu_anh == 0` trong top-100 của mọi câu;
3. `fingerprint_lan_1 == fingerprint_lan_2`;
4. `diem_rerank_lan_1 == diem_rerank_lan_2`;
5. `delta > 0` so với baseline chạy trên đúng cùng tập ứng viên thô;
6. chọn biến thể (`thay` hoặc `rrf`) bằng điểm dev v2, không chọn bằng cảm giác.

Nếu máy báo thiếu ảnh, không dùng `--cho-phep-thieu-anh` để chốt điểm. Cờ đó
chỉ phục vụ debug; cần bổ sung đủ keyframe top-100 rồi chạy lại.

## Tài nguyên

Mặc định `rerank.kich_thuoc_lo: 8`. Nếu hết VRAM/RAM, hạ trong
`config/settings.yaml` theo thứ tự 4 rồi 2. Thay batch size không được làm đổi
fingerprint; nếu đổi thì phép chạy chưa tất định và chưa được nghiệm thu.
