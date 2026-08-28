# Việc 10 — Caption cho keyframe

## Mục tiêu nghiệm thu

- Mỗi keyframe có đúng một dòng trong `derived/captions/<video_id>.jsonl`.
- Ít nhất 95% dòng có caption không rỗng; dòng lỗi vẫn phải có `status` để
  không làm lệch quan hệ `n → frame_idx → pts_time`.
- Caption không rỗng được nạp vào bảng `caption_fts` trong
  `index/fts/text.sqlite`.
- Đo riêng các câu `loai_truy_van = "mo_ta"` của dev v2; không gộp Q&A hoặc
  TRAKE vào con số này.

## Chạy trên PowerShell

Đo tốc độ và xem thử caption trước:

```powershell
python -u -m scripts.run_caption_batch `
  --thu L30_V023 8 `
  --thiet-bi cuda
```

Sinh cho toàn bộ keyframe đang có trên máy và tự nạp FTS:

```powershell
python -u -m scripts.run_caption_batch `
  --thiet-bi cuda `
  --lo 16
```

Nếu máy không có CUDA, đổi thành `--thiet-bi cpu --lo 4`. Có thể chia việc
theo shard bằng `--shard L23 L24`. Nếu dừng giữa chừng, chạy lại đúng lệnh;
chương trình đọc `.jsonl.partial` và chỉ sinh phần còn thiếu. Chỉ dùng
`--lam-lai` khi thực sự muốn xóa hiệu quả chạy tiếp và sinh lại cả video.

Khi nhận JSONL caption từ máy khác, chép vào `derived/captions/` rồi nạp lại
toàn bộ mà không cần keyframe gốc:

```powershell
python -u -m scripts.run_caption_batch --chi-nap
```

Nghiệm thu schema, độ phủ, FTS và 46 câu thuần thị giác trong dev v2:

```powershell
python -u -m scripts.verify_task10_acceptance `
  --tep-dev "D:\aic-data\dev\dev_questions.jsonl"
```

Đo đúng hai nhánh CLIP và caption trên riêng nhóm `mo_ta`:

```powershell
python -u -m scripts.do_trong_so_rrf `
  --tep "D:\aic-data\dev\dev_questions.jsonl" `
  --chi-mo-ta `
  --chi-clip-caption `
  --nguon-mo-rong marian `
  --nguon-caption marian
```

Báo cáo được ghi vào `runs/task10_caption_mo_ta.json`. Phép đo riêng này cố ý
không sửa `config/rrf_weights.yaml`; chỉ chốt trọng số caption khi mức tăng lớn
hơn nhiễu của bộ dev.

Sau khi kết quả đo xác nhận nhánh gộp tốt hơn CLIP đơn, bật caption trong mạch
sinh bài nộp bằng cờ tường minh (chỉ áp dụng cho KIS):

```powershell
python -u -m scripts.tao_bo_nop `
  --de "D:\aic-data\de\SOTUYEN1" `
  --ra p1-caption `
  --dung-caption
```

## Chia việc giữa các máy

Mỗi máy chạy một hoặc nhiều shard. Tệp JSONL là đơn vị trao đổi; không trao đổi
`text.sqlite` vì nạp lại từ JSONL nhanh và tránh ghi đè chỉ mục của đồng đội.
Tên mô hình mặc định là `Salesforce/blip-image-captioning-base`, beam search 3,
không sampling, nên các máy sinh kết quả tất định với cùng phiên bản phụ thuộc.
