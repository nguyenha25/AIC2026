"""
compute_frame_offset.py  (v2 — cập nhật sau khi có frame_map.parquet thật)
----------------------------------------------------------------------------
ĐÃ XÁC MINH trên toàn bộ 177.321 dòng của frame_map.parquet thật:
 
    frame_idx = floor(pts_time × fps)      (khớp tuyệt đối 100%)
 
fps KHÔNG cố định 25 cho mọi video — mỗi video có fps riêng (25 / 29.97 / 30 /
26.44...), và cột 'fps' trong frame_map.parquet đã ghi sẵn giá trị đúng.
 
=> Không cần dùng PotPlayer để "đoán offset" nữa. Chỉ cần:
   1. Xác định đúng giây (pts_time) của khoảnh khắc — có thể dùng PotPlayer để
      xem, tua từng khung cho chắc mắt, nhưng KHÔNG cần đọc số khung nó hiển thị.
   2. Chạy script này với --video-id và --pts-time để tính đúng frame_idx.
 
CÁCH DÙNG:
 
1) Tính frame_idx cho MỘT thời điểm cụ thể (dùng khi soạn câu hỏi):
       python compute_frame_offset.py --frame-map frame_map.parquet \
           --video-id L26_V436 --pts-time 137.9 --khoang 4
 
   → In ra frame_idx chính xác + gợi ý frame_start/frame_end (khoảng quanh đó).
 
2) Xác minh lại công thức trên toàn bộ (hoặc 1 video) — chỉ cần chạy 1 lần
   để yên tâm, không cần chạy lại mỗi câu hỏi:
       python compute_frame_offset.py --frame-map frame_map.parquet --xac-minh
 
Yêu cầu thư viện: pandas, pyarrow
    pip install pandas pyarrow --break-system-packages
"""
 
import argparse
import math
import sys
 
 
def tinh_frame_idx(pts_time, fps):
    """Công thức đã xác minh 100% trên dữ liệu thật: frame_idx = floor(pts_time * fps)."""
    return math.floor(pts_time * fps)
 
 
def lay_fps_video(df, video_id, cot_video, cot_fps):
    df_video = df[df[cot_video].astype(str) == str(video_id)]
    if df_video.empty:
        print(f"LỖI: không tìm thấy video_id '{video_id}' trong frame_map.parquet")
        sys.exit(1)
    fps_values = df_video[cot_fps].unique()
    if len(fps_values) > 1:
        print(f"CẢNH BÁO: video '{video_id}' có nhiều giá trị fps khác nhau: {fps_values} — dùng giá trị đầu tiên.")
    return float(fps_values[0])
 
 
def tim_cot(df):
    cot_video = cot_giay = cot_frame = cot_fps = None
    for c in df.columns:
        cl = c.lower()
        if cot_video is None and "video" in cl:
            cot_video = c
        if cot_giay is None and ("pts_time" == cl or "giay" in cl or "second" in cl or ("pts" in cl and "time" in cl)):
            cot_giay = c
        if cot_frame is None and cl == "frame_idx":
            cot_frame = c
        if cot_fps is None and cl == "fps":
            cot_fps = c
    return cot_video, cot_giay, cot_frame, cot_fps
 
 
def che_do_tinh(frame_map_path, video_id, pts_time, khoang):
    import pandas as pd
    df = pd.read_parquet(frame_map_path)
    cot_video, cot_giay, cot_frame, cot_fps = tim_cot(df)
 
    if not all([cot_video, cot_fps]):
        print(f"LỖI: không tự nhận diện được cột video_id/fps. Các cột hiện có: {list(df.columns)}")
        sys.exit(1)
 
    fps = lay_fps_video(df, video_id, cot_video, cot_fps)
    frame_idx = tinh_frame_idx(pts_time, fps)
    frame_start = max(0, frame_idx - khoang // 2)
    frame_end = frame_idx + (khoang - khoang // 2)
 
    print("\n" + "=" * 60)
    print(f"KẾT QUẢ TÍNH TOÁN cho video '{video_id}'")
    print("=" * 60)
    print(f"fps thật của video     : {fps}")
    print(f"pts_time bạn nhập      : {pts_time}")
    print(f"frame_idx chính xác    : {frame_idx}")
    print(f"Gợi ý frame_start      : {frame_start}")
    print(f"Gợi ý frame_end        : {frame_end}   (khoảng rộng {frame_end - frame_start} khung)")
    print("=" * 60)
    print("Lưu ý: frame_start/frame_end chỉ là GỢI Ý đối xứng quanh frame_idx.")
    print("Bạn vẫn nên xem lại bằng mắt (PotPlayer/ffmpeg) để đảm bảo khoảng này")
    print("thực sự chứa trọn khoảnh khắc kích hoạt, rồi mới chốt vào file JSONL.")
 
 
def che_do_xac_minh(frame_map_path, video_id_loc=None):
    import pandas as pd
    df = pd.read_parquet(frame_map_path)
    cot_video, cot_giay, cot_frame, cot_fps = tim_cot(df)
 
    if not all([cot_video, cot_giay, cot_frame, cot_fps]):
        print(f"LỖI: không tự nhận diện đủ 4 cột. Các cột hiện có: {list(df.columns)}")
        sys.exit(1)
 
    if video_id_loc:
        df = df[df[cot_video].astype(str) == str(video_id_loc)]
        if df.empty:
            print(f"LỖI: không tìm thấy video_id '{video_id_loc}'")
            sys.exit(1)
 
    du_doan = (df[cot_giay] * df[cot_fps]).apply(math.floor)
    khop = (du_doan == df[cot_frame])
    ty_le = khop.sum() / len(df) * 100
 
    print("\n" + "=" * 60)
    print(f"XÁC MINH CÔNG THỨC frame_idx = floor(pts_time × fps)")
    print("=" * 60)
    print(f"Số dòng kiểm tra : {len(df)}")
    print(f"Số dòng khớp     : {khop.sum()} ({ty_le:.2f}%)")
    if ty_le == 100:
        print("✅ Công thức ĐÚNG TUYỆT ĐỐI trên toàn bộ dữ liệu kiểm tra.")
    else:
        print(f"⚠️  Có {len(df) - khop.sum()} dòng KHÔNG khớp — xem lại trước khi tin dùng công thức.")
        print(df[~khop][[cot_video, cot_giay, cot_fps, cot_frame]].head(10))
    print("=" * 60)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tính frame_idx chính xác từ pts_time (dựa trên frame_map.parquet)")
    parser.add_argument("--frame-map", dest="frame_map", required=True, help="Đường dẫn frame_map.parquet")
    parser.add_argument("--video-id", dest="video_id", default=None, help="video_id cần tính (bắt buộc nếu không dùng --xac-minh)")
    parser.add_argument("--pts-time", dest="pts_time", type=float, default=None, help="Giây cần tính frame_idx tương ứng")
    parser.add_argument("--khoang", dest="khoang", type=int, default=4, help="Độ rộng khoảng frame_start/frame_end gợi ý (mặc định 4)")
    parser.add_argument("--xac-minh", dest="xac_minh", action="store_true", help="Chạy chế độ xác minh công thức trên toàn bộ (hoặc 1 video nếu có --video-id)")
    args = parser.parse_args()
 
    if args.xac_minh:
        che_do_xac_minh(args.frame_map, args.video_id)
    elif args.video_id and args.pts_time is not None:
        che_do_tinh(args.frame_map, args.video_id, args.pts_time, args.khoang)
    else:
        print("Cần cung cấp --xac-minh, HOẶC (--video-id <id> --pts-time <giây>). Xem --help.")
        sys.exit(1)
 