"""
trake_r2_dp.py — Lõi DP của TR-R2: "đường sự kiện tốt nhất".

Công thức (nguyên văn handbook):
    maximize sum_i S[i, j_i]  subject to  j_1 < j_2 < ... < j_N

Prefix maximum giúp tính trong O(N*F) thay vì O(N*F^2) — đây chính là lý do
checklist cảnh báo "Bùng nổ O(N^2)" nếu làm ẩu (vd duyệt mọi cặp (i,j) x
(i-1,j') không dùng prefix max).

min_gap: "khoảng cách tối thiểu là tham số cấu hình" — mặc định 1 (chỉ cần
j tăng nghiêm ngặt, không cần cách nhau bao nhiêu frame).
"""
from __future__ import annotations

NEG_INF = float("-inf")


def solve_strict_increasing_path(
    S: list[list[float]], min_gap: int = 1
) -> tuple[list[int], float]:
    """
    S[i][j]: độ phù hợp của event i với frame thứ j (đã chuẩn hóa [0,1]).
    Mọi event phải có cùng số frame F (cùng một lưới dense frame trong 1 video).

    Trả (chosen_frame_indices, total_score). chosen_frame_indices[i] = chỉ số
    frame được chọn cho event i, đảm bảo tăng nghiêm ngặt và cách nhau
    >= min_gap.

    Raises:
        ValueError nếu S rỗng không đều hàng, hoặc bài toán infeasible
        (không đủ frame để xếp N event cách nhau >= min_gap) — fail loud,
        không tự động nới lỏng ràng buộc.
    """
    n_events = len(S)
    if n_events == 0:
        return [], 0.0

    if min_gap < 1:
        raise ValueError(
            "TRAKE yêu cầu min_gap >= 1 để frame tăng nghiêm ngặt"
        )

    n_frames = len(S[0])
    for row in S:
        if len(row) != n_frames:
            raise ValueError("Mọi event phải có cùng số frame F trong ma trận S")

    if n_frames < 1 + (n_events - 1) * min_gap:
        raise ValueError(
            f"Infeasible: {n_events} event cần tối thiểu "
            f"{1 + (n_events - 1) * min_gap} frame (min_gap={min_gap}), "
            f"nhưng chỉ có {n_frames} frame."
        )

    dp = [[NEG_INF] * n_frames for _ in range(n_events)]
    back = [[-1] * n_frames for _ in range(n_events)]

    for j in range(n_frames):
        dp[0][j] = S[0][j]

    for i in range(1, n_events):
        # prefix_max[j] = max(dp[i-1][0..j]), prefix_arg[j] = argmax tương ứng.
        prefix_max = [NEG_INF] * n_frames
        prefix_arg = [-1] * n_frames
        best, best_idx = NEG_INF, -1
        for j in range(n_frames):
            if dp[i - 1][j] > best:
                best, best_idx = dp[i - 1][j], j
            prefix_max[j] = best
            prefix_arg[j] = best_idx

        for j in range(n_frames):
            prev_limit = j - min_gap
            if prev_limit < 0 or prefix_max[prev_limit] == NEG_INF:
                continue
            candidate = S[i][j] + prefix_max[prev_limit]
            if candidate > dp[i][j]:
                dp[i][j] = candidate
                back[i][j] = prefix_arg[prev_limit]

    best_total, best_end = NEG_INF, -1
    for j in range(n_frames):
        if dp[n_events - 1][j] > best_total:
            best_total, best_end = dp[n_events - 1][j], j

    if best_end == -1:
        raise ValueError("Infeasible: không tìm được chuỗi frame hợp lệ nào.")

    chosen = [0] * n_events
    j = best_end
    for i in range(n_events - 1, -1, -1):
        chosen[i] = j
        j = back[i][j] if i > 0 else -1

    return chosen, best_total
