from aic2026.rank.hop_nhat import tim_ung_vien_gop
from aic2026.rank.search import tim_ung_vien_clip

def test_clip_only_fusion_preserves_order():
    query = "Một người mặc áo blouse trắng đeo khẩu trang đang chăm sóc bệnh nhân trên giường bệnh."
    
    print("Đang gọi Direct CLIP...")
    direct = list(tim_ung_vien_clip(query, 500))
    
    print("Đang gọi Fusion CLIP-only...")
    gop = tim_ung_vien_gop(
        dung_clip=True, dung_ocr=False, dung_ocr_fts=False,
        dung_asr=False, dung_object=False, dung_caption=False, dung_clip_l=False
    )
    fused = gop(query, 500)

    # Đảm bảo số lượng bằng nhau
    assert len(fused) == len(direct), f"Lệch số lượng: {len(fused)} vs {len(direct)}"
    
    # Kiểm tra khắt khe 4 trường như yêu cầu
    for i, (f, d) in enumerate(zip(fused, direct)):
        assert f.video_id == d.video_id, f"Lệch video_id ở hạng {i}: {f.video_id} != {d.video_id}"
        assert f.n == d.n, f"Lệch n ở hạng {i}: {f.n} != {d.n}"
        assert f.frame_idx == d.frame_idx, f"Lệch frame_idx ở hạng {i}"
        assert round(float(f.pts_time or 0), 3) == round(float(d.pts_time or 0), 3), f"Lệch pts_time ở hạng {i}"

    print("✅ REGRESSION TEST PASSED: Fusion CLIP-only hoàn toàn khớp với Direct CLIP (video_id, n, frame_idx, pts_time).")

if __name__ == "__main__":
    test_clip_only_fusion_preserves_order()