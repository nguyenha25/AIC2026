# Việc 9 — quyết định trích `frames_dense`

## Kết luận

- Chỉ trích các video nghi là đáp án TRAKE, không trích cả 873 video.
- Lịch mặc định là 0,16 giây/ảnh; với 25/30 fps đều là 4 khung. Bộ dev hiện
  tại có cửa sổ TRAKE ngắn nhất chỉ 4 frame, nên đây là bước lớn nhất vẫn bảo
  đảm mỗi cửa sổ có ít nhất một mẫu, không phụ thuộc pha của lịch lấy mẫu.
  Tiêu chí nghiệm thu là khoảng cách **lớn nhất** trong từng vùng phải nhỏ
  hơn 0,5 giây và mọi cửa sổ dev phải có ít nhất một ảnh.
- Tên ảnh là `frame_idx` sáu chữ số, không phải số thứ tự keyframe `n`.
- Giữ thêm mọi keyframe BTC trong vùng trích. Các ảnh này không làm khoảng
  cách thưa hơn và tạo mốc độc lập để đối chiếu cách đánh số.
- Không dùng `ffmpeg -ss`: tua trước khi giải mã làm bộ đếm `n` bắt đầu lại,
  có thể khiến tên ảnh lệch âm thầm.

## Vì sao không trích mọi khung

Trích 25–30 ảnh/giây không cần thiết. Bản thử lịch 0,4 giây chỉ phủ 44/50 cửa
sổ của `dev_questions.jsonl`: sáu cửa sổ dài 5–8 frame nằm lọt giữa hai mẫu.
Lịch 0,16 giây phủ 50/50 và vẫn giảm số ảnh khoảng 4–7,5 lần so với trích mọi
frame.

## Nghiệm thu

`scripts.verify_task9_acceptance` kiểm bốn điều:

1. thư mục và ảnh tồn tại;
2. mọi tên ảnh là số `frame_idx`;
3. manifest phủ đủ hai đầu mỗi vùng;
4. lỗ lớn nhất giữa hai ảnh liên tiếp nhỏ hơn 0,5 giây;
5. nếu truyền `--tep-dev`, mọi `cac_giai_doan` TRAKE đều chứa ít nhất một ảnh.

Nếu `raw/keyframes` có trên máy, script còn so ảnh dense với ảnh BTC tại cùng
`frame_idx`. Thiếu ảnh BTC chỉ là cảnh báo vì đầu vào bắt buộc của Việc 9 theo
kế hoạch là video gốc và map-keyframes.
