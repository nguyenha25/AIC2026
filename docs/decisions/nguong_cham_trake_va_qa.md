# 001 — Ánh xạ schema dev thật, TRAKE dùng khoảng thật, Q&A chấm KHÔNG TỐN PHÍ (Task 12)
 
**Ngày:** 16/08/2026 (bản cuối — đổi hướng sang chấm Q&A miễn phí, không dùng API trả tiền)
**Người đề xuất:** Nguyên (Task 12) — CẦN CẢ NHÓM XÁC NHẬN, chưa chốt chính thức.
 
## 1. Ánh xạ schema dev thật
 
`dev_question_Nguyen.jsonl` dùng tên trường khác mẫu ở `docs/schema/`:
`id`→`query_id`, `loai_truy_van` (`mo_ta`/`hoi_dap`/`chuoi_su_kien`)→`kis`/
`qa`/`trake`, `frame_start`+`frame_end`→`gt_frame_range`, `cau_tra_loi`→
`gt_answer`, `cac_giai_doan`→`gt_events`. Việc ánh xạ chỉ làm ở MỘT chỗ:
hàm `build_gt()` trong `scripts/run_scoring.py`. `scoring.py` không biết gì
về tên trường gốc.
 
## 2. TRAKE dùng khoảng [s,e] thật, không dùng dung sai tự chế
 
`cac_giai_doan[j]` đã có sẵn `frame_start`/`frame_end` — khoảng đúng thật
của từng khoảnh khắc, khớp chính xác công thức gốc BTC (mục 2.1.3).
`r_score_trake()` so khớp trực tiếp `s_j <= frame_nộp <= e_j`.
 
## 3. Q&A — so khớp ngữ nghĩa KHÔNG GỌI API TRẢ PHÍ
 
**Bản trước (đã bỏ):** dùng Claude API làm giám khảo ngữ nghĩa. Về mặt kỹ
thuật đúng và chính xác nhất, nhưng tốn phí (dù rất nhỏ — vài cent với quy
mô bộ dev) và cần cấu hình thêm `ANTHROPIC_API_KEY`.
 
**Đã đổi sang — hoàn toàn miễn phí:** dùng mô hình embedding câu
(`sentence-transformers`, model `paraphrase-multilingual-MiniLM-L12-v2`,
hỗ trợ tiếng Việt) chạy LOCAL trên máy, so độ tương đồng cosine giữa hai
câu trả lời. Không gọi mạng trả phí, không cần API key.
 
Package `sentence-transformers` **đã có sẵn** trong `requirements.txt`
(dùng cho việc khác ở Giai đoạn 1) — không cần thêm gì, không cần báo
nhóm về package mới.
 
**Chi phí thật sự phát sinh:** mô hình cần TẢI VỀ MÁY đúng một lần (cần
mạng internet lúc chạy lần đầu, tải free từ HuggingFace — không phải trả
phí, chỉ là băng thông/thời gian chờ). Sau lần đầu, chạy hoàn toàn offline.
 
**Ngưỡng độ tương đồng:** `cham_diem.nguong_tuong_dong_qa = 0.80` trong
`config/settings.yaml`. Con số này CHƯA qua kiểm định thực tế trên câu hỏi
Q&A thật của nhóm — **CẦN NHÓM CHẠY THỬ và điều chỉnh** sau khi nhìn vài
cặp answer/GT thật (nếu quá chặt sẽ chấm sai một answer đúng nghĩa nhưng
diễn đạt khác; quá lỏng sẽ chấm đúng cho answer sai nghĩa).
 
**Đánh đổi so với bản LLM:** embedding similarity không "hiểu" tốt bằng
LLM ở các trường hợp phức tạp (ví dụ số đếm viết bằng chữ khác ngôn ngữ,
suy luận gián tiếp) — nhưng với answer ngắn, cụ thể (như "5 người", "màu
xanh") thì đủ tốt và ổn định, đồng thời deterministic tuyệt đối (không cần
cache như bản LLM, vì không có yếu tố ngẫu nhiên/thay đổi giữa các lần gọi
API).