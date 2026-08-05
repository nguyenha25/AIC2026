# Mẫu dữ liệu bốn nhánh trao cho nhau

**Chốt ở Task 8. Sau khi chốt, ai muốn đổi phải báo nhóm trước.**

Mục đích: bốn người làm bốn phần khác nhau CÙNG LÚC mà không phải chờ nhau.
Ai cũng biết trước phần mình nhận vào cái gì và trả ra cái gì, nên tự tạo được
dữ liệu giả để làm ngay, không cần đợi ba người kia làm xong.

Mỗi mẫu dưới đây có một tệp ví dụ ba dòng nằm cùng thư mục này. Mở tệp ví dụ
ra đọc là hiểu, không cần hỏi ai.

---

## Bốn nhánh và ranh giới giữa chúng

| Nhánh | Thư mục mã nguồn | Nhận vào | Trả ra |
|---|---|---|---|
| Tra cứu | `src/aic2026/index/` | `raw/clip-features-32/`, `frame_map` | `index/faiss/`, danh sách ứng viên |
| Làm giàu | `src/aic2026/enrich/` | `raw/keyframes/`, `derived/audio/` | `derived/ocr/`, `derived/asr/` |
| Xếp hạng | `src/aic2026/rank/` | danh sách ứng viên của các nhánh | danh sách đã gộp + lọc trùng |
| Nộp & chấm | `src/aic2026/submit/`, `eval/` | danh sách đã xếp hạng | tệp `.csv` nộp bài, điểm tự chấm |

Ranh giới quan trọng nhất: **nhánh nào cũng chỉ nói chuyện với nhau bằng
`video_id` + `frame_idx`**, không truyền đường dẫn tệp, không truyền số thứ tự
tấm ảnh `n`.

---

## 1. `frame_map` — xương sống, mọi nhánh đều dùng

Không phải tệp nhóm tự sinh; đây là cách ĐỌC `raw/map-keyframes/<video_id>.csv`.

| Cột | Kiểu | Nghĩa |
|---|---|---|
| `n` | số nguyên | số thứ tự tấm ảnh. Ảnh `0001.jpg` thì `n = 1` |
| `pts_time` | số thực | tấm ảnh nằm ở giây thứ mấy |
| `fps` | số thực | video quay bao nhiêu hình một giây |
| `frame_idx` | số nguyên | **VỊ TRÍ KHUNG HÌNH — số này mới đem nộp** |

Kiểm chứng: `frame_idx ≈ round(pts_time × fps)`, lệch cho phép 0 hoặc 1.

`n` và `frame_idx` là hai số hoàn toàn khác nhau. Trong mọi hàm, mọi tệp,
mọi biến — đặt tên rõ ràng cái nào là cái nào. Nộp nhầm `n` thì BTC chấm 0 điểm
mà kết quả trên màn hình vẫn trông rất đúng.

Đọc bằng: `FrameMap.load(video_id)` trong `src/aic2026/frame_map.py`.

---

## 2. `derived/ocr/<video_id>.jsonl` — chữ đọc được trên ảnh

Nhánh **Làm giàu** ghi ra. Nhánh **Tra cứu** và **Xếp hạng** đọc vào.
Mỗi dòng một bản ghi (`.jsonl`), một tệp một video.

| Khóa | Kiểu | Bắt buộc | Nghĩa |
|---|---|---|---|
| `video_id` | chuỗi | có | `L21_V001` |
| `n` | số nguyên | có | tấm ảnh thứ mấy |
| `frame_idx` | số nguyên | có | vị trí khung hình (chép sẵn để khỏi tra lại) |
| `text` | chuỗi | có | chữ đọc được, gộp cả ảnh thành một chuỗi |
| `boxes` | mảng | không | `[{"text":..., "bbox":[x1,y1,x2,y2], "conf":0.0-1.0}]` |
| `engine` | chuỗi | có | tên công cụ đã dùng, vd `easyocr-1.7.1` |

Ảnh không có chữ: vẫn ghi một dòng, `text` để chuỗi rỗng. Bỏ dòng đi thì
người đọc không phân biệt được "chưa chạy" với "chạy rồi mà không có chữ".

Xem `example_ocr.jsonl`.

---

## 3. `derived/asr/<video_id>.jsonl` — lời nói chuyển thành chữ

Nhánh **Làm giàu** ghi ra. Một dòng một đoạn lời nói, không phải một dòng một ảnh.

| Khóa | Kiểu | Bắt buộc | Nghĩa |
|---|---|---|---|
| `video_id` | chuỗi | có | `L21_V001` |
| `start` | số thực | có | giây bắt đầu đoạn |
| `end` | số thực | có | giây kết thúc đoạn |
| `text` | chuỗi | có | lời nói, **tiếng Việt có dấu** |
| `frame_idx_start` | số nguyên | có | `round(start × fps)` |
| `frame_idx_end` | số nguyên | có | `round(end × fps)` |
| `lang` | chuỗi | có | `vi` hoặc `en` |
| `engine` | chuỗi | có | vd `faster-whisper-large-v3` |

Hai cột `frame_idx_*` là để nhánh Xếp hạng khỏi phải mở lại `map-keyframes`.
Người ghi tính sẵn một lần, ba người kia dùng lại.

Xem `example_asr.jsonl`.

---

## 4. `runs/<tên lần chạy>/candidates.jsonl` — kết quả tra cứu

Nhánh **Tra cứu** ghi ra. Nhánh **Xếp hạng** đọc vào. Đây là mẫu quan trọng
nhất vì nó nằm đúng chỗ nối giữa hai người.

| Khóa | Kiểu | Bắt buộc | Nghĩa |
|---|---|---|---|
| `query_id` | chuỗi | có | mã truy vấn, vd `1` |
| `video_id` | chuỗi | có | `L21_V001` |
| `n` | số nguyên | có | tấm ảnh thứ mấy |
| `frame_idx` | số nguyên | có | vị trí khung hình |
| `score` | số thực | có | điểm giống nhau, càng cao càng giống |
| `source` | chuỗi | có | nhánh nào tìm ra: `clip`, `ocr`, `asr`, `object` |
| `rank` | số nguyên | có | thứ hạng trong riêng nguồn đó, đếm từ 1 |

`source` và `rank` bắt buộc phải có vì nhánh Xếp hạng cần chúng để gộp kết quả
nhiều nguồn (RRF). Thiếu hai cột này là không gộp được.

Xem `example_candidates.jsonl`.

---

## 5. `dev/queries.jsonl` — bộ câu hỏi tự nghĩ để tự chấm điểm

Nhánh **Nộp & chấm** dùng. Cả nhóm cùng góp câu hỏi.

| Khóa | Kiểu | Bắt buộc | Nghĩa |
|---|---|---|---|
| `query_id` | chuỗi | có | mã do nhóm tự đặt |
| `task` | chuỗi | có | `kis`, `qa` hoặc `trake` |
| `text` | chuỗi | có | nội dung truy vấn |
| `gt_video_id` | chuỗi | có | đáp án: video nào |
| `gt_frame_range` | mảng 2 số | KIS/QA | `[s, e]` — khoảng khung hình đúng |
| `gt_frame_ids` | mảng số | TRAKE | mỗi khoảnh khắc một số |
| `gt_answer` | chuỗi | Q&A | đáp án chữ |

Xem `example_dev_queries.jsonl`.

---

## 6. Tệp nộp bài — `submissions/query-<id>-<dạng>.csv`

Không có dòng tiêu đề. Thứ tự dòng chính là thứ hạng. Tối đa **100 dòng**.

```
kis    L21_V001,1500
qa     L21_V001,3450,màu xanh
trake  L21_V001,120,340,560,780
```

Ba luật:

1. Tối đa 100 dòng. Dòng 101 trở đi bị bỏ.
2. Không dòng nào trùng dòng nào. Điểm lấy `max` ở từng mốc thứ hạng, nên hai
   dòng trỏ cùng một chỗ chỉ ăn điểm một lần mà lại chiếm mất một suất.
3. TRAKE sai `video_id` là 0 điểm ngay, không xét tiếp khung hình.

Ghi bằng `SubmissionBudget` trong `src/aic2026/submit/formatter.py` — lớp này
tự bỏ trùng và tự cắt ở 100 dòng.

> Quy ước **đặt tên tệp** là nhóm tự chọn, BTC chưa công bố chính thức.
> Khi có thông báo thì sửa hàm `submission_filename()`, chỗ khác không phải đụng.

---

## Ba quy ước chung cho mọi tệp trên

1. **UTF-8, luôn luôn.** Ghi bằng `open(..., encoding="utf-8")`. Tiếng Việt có
   dấu phải giữ nguyên dấu, không được chuyển thành `\uXXXX`.
2. **Một tệp một video.** Gộp nhiều video vào một tệp lớn là hỏng khả năng bốn
   người ghi song song.
3. **Tệp đang viết dở mang đuôi `.partial`**, viết xong mới đổi tên. Tránh cảnh
   người khác đọc phải tệp mới ghi được nửa chừng.
