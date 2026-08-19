"""Script nghiệm thu Task 10: So sánh Visual only vs OCR only vs Visual+OCR (RRF).

CÁCH CHẠY (từ thư mục gốc D:\\AIC2026):
    python -u -m scripts.benchmark_rrf

Nhờ đoạn "bootstrap sys.path" ngay dưới, script cũng chạy được khi gọi trực
tiếp `python scripts\\benchmark_rrf.py` mà KHÔNG cần `pip install -e .`, vì nó
tự thêm thư mục src/ vào sys.path để `import aic2026` luôn tìm thấy.

VÌ SAO TRƯỚC ĐÂY "KHÔNG HIỆN GÌ": bản cũ import theo `src.aic2026`. Khi gọi
`python scripts\\benchmark_rrf.py`, Python chỉ để thư mục scripts/ lên sys.path
(không có gốc repo), nên `import src` gãy ngay dòng đầu — main() chưa kịp chạy,
màn hình trống. Bản này thống nhất về `aic2026` (đúng tên gói trong pyproject).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap sys.path — PHẢI đặt TRƯỚC mọi import aic2026.
# parents[1] = gốc repo (scripts/ nằm ngay dưới gốc). Thêm cả gốc và src/ để:
#   - `import aic2026`      chạy dù chưa pip install -e .   (nhờ src/)
#   - các module còn lỡ dùng `from src.aic2026...` vẫn resolve (nhờ gốc)
# Khi cả nhóm đã đổi hết sang `aic2026`, có thể bỏ dòng thêm gốc repo.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

# In ra ngay (line_buffering) và ép UTF-8 cho stdout/stderr. Trên Windows, khi
# output bị chuyển hướng/pipe, Python lùi về cp1252 và chết khi in chữ có dấu
# (UnicodeEncodeError trên 'Ự', 'Ấ'...). reconfigure UTF-8 để chạy được cả khi
# `... > log.txt` hay bị script khác gọi qua subprocess.
for _luong in (sys.stdout, sys.stderr):
    try:
        _luong.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:  # noqa: BLE001 — môi trường lạ thì thôi, đã có -u dự phòng
        pass

# ---------------------------------------------------------------------------
# NẠP torch TRƯỚC pandas — bắt buộc, đừng đổi thứ tự.
#
# pandas kéo numpy dùng runtime MKL/OpenMP riêng. Nếu numpy-MKL nạp TRƯỚC torch,
# thì khi torch rồi faiss nạp OpenMP của chúng vào sẽ trùng libiomp5md.dll →
# Python chết ngay lúc import với mã 0xC0000005 (-1073741819), KHÔNG traceback,
# màn hình trống. Đã đo: `import torch, ... ` chạy tốt; `import pandas, torch, ...`
# thì chết. Nạp torch đầu tiên để runtime của nó chiếm chỗ trước là hết.
#
# KMP_DUPLICATE_LIB_OK KHÔNG cứu được vì đây là lỗi lúc nạp DLL, không phải runtime.
# ---------------------------------------------------------------------------
import torch  # noqa: F401 — PHẢI đứng trước pandas / open_clip / faiss

import pandas as pd

from aic2026.index.encode.clip_encoder import ClipEncoder
from aic2026.index.faiss_index import search_by_vector
from aic2026.paths import DERIVED_DIR, DEV_QUERIES_PATH, SUBMISSIONS_DIR
from aic2026.rank.ocr_rerank import OCRReranker
from aic2026.rank.search import reciprocal_rank_fusion
from aic2026.rank.dedupe import deduplicate_temporal
from aic2026.frame_map import load_frame_map
from aic2026.submit import KIS, QA, TRAKE, submission_filename

clip_encoder = ClipEncoder()


def doc_tep_cau_hoi(duong_dan: Path) -> list[dict]:
    if not duong_dan.exists():
        raise FileNotFoundError(
            f"Không thấy bộ câu hỏi dev: {duong_dan}\n"
            "Kiểm tra DEV_QUERIES_PATH trong paths.py và file thật trên đĩa."
        )
    cau_hoi = []
    with open(duong_dan, "r", encoding="utf-8") as f:
        for so_dong, dong in enumerate(f, start=1):
            if not dong.strip():
                continue
            data = json.loads(dong)
            text = (data.get("cau_hoi") or data.get("cau") or data.get("query") or "").strip()
            q_id = str(data.get("id") or so_dong)
            loai = str(data.get("loai_truy_van") or data.get("dang") or data.get("type") or "kis")
            if loai in {"mo_ta", "kis"}:
                loai = KIS
            elif loai in {"hoi_dap", "qa"}:
                loai = QA
            elif loai in {"chuoi_su_kien", "trake"}:
                loai = TRAKE
            data["id"] = q_id
            data["query_id"] = q_id
            data["text"] = text
            data["task"] = loai
            data["gt_answer"] = data.get("cau_tra_loi")
            cau_hoi.append(data)
    return cau_hoi


def run_experiment(mode: str, danh_sach: list[dict], ocr_engine: OCRReranker) -> tuple[float, float]:
    """Chạy tìm kiếm theo từng mode: 'visual', 'ocr', hoặc 'rrf'."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for p in SUBMISSIONS_DIR.glob("*.csv"):
        p.unlink()

    # Nạp frame_map MỘT LẦN cho cả mode. Không bọc try/except nuốt lỗi ở đây:
    # nếu frame_map hỏng thì phải biết ngay, đừng lặng lẽ bỏ lọc trùng.
    frame_map = load_frame_map()

    for item in danh_sach:
        q_id = item["query_id"]
        task = item["task"]
        query_text = item["text"]

        visual_hits = []
        ocr_hits = []

        if mode in {"visual", "rrf"}:
            q_vec = clip_encoder.encode_text(query_text)
            hits = search_by_vector(q_vec, top_k=200)
            visual_hits = [
                {"video_id": h.video_id, "frame_idx": h.frame_idx, "score": float(h.score)}
                for h in hits
            ]

        if mode in {"ocr", "rrf"}:
            ocr_hits = ocr_engine.search_ocr(query_text, top_k=200)

        if mode == "visual":
            ket_qua = visual_hits
        elif mode == "ocr":
            ket_qua = ocr_hits
        else:  # Adaptive RRF Fusion (Task 10)
            if len(ocr_hits) > 0:
                modality_results = {"ocr": ocr_hits, "visual": visual_hits}
                # RRF Bất đối xứng (Asymmetric RRF)
                # Thay vì 1000.0, ta dùng 0.45 để OCR làm Tie-breaker chứ không lật đổ Visual
                weights = {"ocr": 0.45, "visual": 1.0}
            else:
                modality_results = {"visual": visual_hits}
                weights = {"visual": 1.0}

            ket_qua = reciprocal_rank_fusion(modality_results, weights=weights, k_rrf=60)

        # LỌC TRÙNG (Deduplicate) thay vì bốc đại top 100.
        # KHÔNG bọc try/except câm: nếu dedup lỗi, in cảnh báo rõ RÀNG rồi mới
        # lùi về [:100], để không âm thầm so lệch điểm giữa các mode.
        try:
            ket_qua_sach = deduplicate_temporal(
                ket_qua, frame_map=frame_map, window_seconds=2, limit=100
            )
        except Exception as loi:  # noqa: BLE001
            print(
                f"    [CẢNH BÁO] dedup lỗi ở câu {q_id} ({type(loi).__name__}: {loi}); "
                "tạm lùi về top-100 chưa lọc trùng.",
                file=sys.stderr,
            )
            ket_qua_sach = ket_qua[:100]

        # Xuất CSV nộp bài
        fname = submission_filename(q_id, task)
        out_csv = SUBMISSIONS_DIR / fname
        if ket_qua_sach:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                for r in ket_qua_sach:
                    if task == QA:
                        ans = item.get("gt_answer") or ""
                        f.write(f"{r['video_id']},{r['frame_idx']},{ans}\n")
                    else:
                        f.write(f"{r['video_id']},{r['frame_idx']}\n")

    # Chạy tự chấm điểm. KHÔNG nuốt output: nếu run_scoring lỗi (exit != 0) thì
    # dừng và in stderr, đừng để hàm trả 0.0 khiến tưởng mode này 'thua'.
    # Ép tiến trình con dùng UTF-8: nếu không, run_scoring in tiếng Việt qua
    # pipe sẽ chết cp1252 (UnicodeEncodeError) và thoát mã 1. encoding+errors ở
    # đây để phía cha đọc lại cũng bằng UTF-8, không vỡ vì một byte lạ.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    kq = subprocess.run(
        [sys.executable, "-m", "scripts.run_scoring"],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    if kq.returncode != 0:
        print("    [LỖI] scripts.run_scoring thất bại:", file=sys.stderr)
        print(kq.stdout, file=sys.stderr)
        print(kq.stderr, file=sys.stderr)
        raise RuntimeError(f"run_scoring exit {kq.returncode} ở mode '{mode}'")

    # Đọc kết quả từ CSV score.
    # run_scoring LUÔN ghi đè scoring_results.csv, nên phải chép ra bản riêng cho
    # từng mode NGAY tại đây — nếu không, chạy xong mode sau là mất bản mode trước
    # và không còn cách nào quy điểm về từng câu để biết fusion giúp ở đâu.
    score_file = DERIVED_DIR / "eval" / "scoring_results.csv"
    if score_file.exists():
        df = pd.read_csv(score_file)
        ban_sao = DERIVED_DIR / "eval" / f"scoring_results_{mode}.csv"
        df.to_csv(ban_sao, index=False)

        col = "final_score" if "final_score" in df.columns else ("score" if "score" in df.columns else df.columns[-1])
        tong_diem = float(df[col].sum())
        dtb_32 = tong_diem / 32.0
        return tong_diem, dtb_32
    print(f"    [CẢNH BÁO] không thấy {score_file} sau khi chấm.", file=sys.stderr)
    return 0.0, 0.0


def doi_chieu_diem_tung_cau() -> None:
    """
    So điểm TỪNG CÂU giữa ba mode, chỉ ra chính xác fusion được/mất ở câu nào.

    Đây là bằng chứng nghiệm thu Task 10: chênh lệch tổng điểm phải quy được về
    những câu cụ thể, nếu không thì con số chỉ là nhiễu không truy được nguồn.
    """
    thu_muc = DERIVED_DIR / "eval"
    bang = {}
    for mode in ("visual", "ocr", "rrf"):
        f = thu_muc / f"scoring_results_{mode}.csv"
        if not f.exists():
            print(f"  [THIẾU] {f.name} — không đối chiếu được.", file=sys.stderr)
            return
        bang[mode] = pd.read_csv(f)

    col = "final_score" if "final_score" in bang["visual"].columns else bang["visual"].columns[-1]

    gop = bang["visual"][["query_id", "task", col]].rename(columns={col: "visual"})
    gop = gop.merge(
        bang["ocr"][["query_id", col]].rename(columns={col: "ocr"}), on="query_id", how="outer"
    )
    gop = gop.merge(
        bang["rrf"][["query_id", col]].rename(columns={col: "rrf"}), on="query_id", how="outer"
    )
    gop = gop.fillna(0.0).sort_values("query_id")
    gop["chenh"] = gop["rrf"] - gop["visual"]

    khac = gop[gop["chenh"].abs() > 1e-9]

    print("\n" + "=" * 100)
    print("[ĐỐI CHIẾU] ĐIỂM TỪNG CÂU: VISUAL vs OCR vs RRF")
    print("=" * 100)
    print(f"{'CÂU':<6} | {'LOẠI':<6} | {'VISUAL':>8} | {'OCR':>8} | {'RRF':>8} | {'CHÊNH':>8}")
    print("-" * 100)
    for _, r in gop.iterrows():
        danh_dau = "  <<<" if abs(r["chenh"]) > 1e-9 else ""
        print(
            f"{str(r['query_id']):<6} | {str(r.get('task', '')):<6} | "
            f"{r['visual']:>8.2f} | {r['ocr']:>8.2f} | {r['rrf']:>8.2f} | {r['chenh']:>+8.2f}{danh_dau}"
        )
    print("-" * 100)
    print(
        f"TỔNG: visual={gop['visual'].sum():.3f}  ocr={gop['ocr'].sum():.3f}  "
        f"rrf={gop['rrf'].sum():.3f}  chênh={gop['chenh'].sum():+.3f}"
    )
    print("-" * 100)

    if khac.empty:
        print("  Fusion KHÔNG đổi điểm câu nào. Chênh lệch tổng (nếu có) là do làm tròn.")
    else:
        tang = khac[khac["chenh"] > 0]
        giam = khac[khac["chenh"] < 0]
        print(f"  Số câu fusion LÀM TĂNG điểm:  {len(tang)}  -> câu {list(tang['query_id'])}")
        print(f"  Số câu fusion LÀM GIẢM điểm:  {len(giam)}  -> câu {list(giam['query_id'])}")
        print(
            f"\n  QUY VỀ NGUỒN: toàn bộ chênh {gop['chenh'].sum():+.3f} điểm nằm ở "
            f"{len(khac)}/{len(gop)} câu."
        )
        print(
            "  LƯU Ý ĐỌC SỐ: bộ dev 32 câu có sai số chuẩn ~±9 điểm phần trăm. Chênh lệch\n"
            "  vài phần mười điểm nằm trong nhiễu — thứ đáng tin là CƠ CHẾ truy được ở từng\n"
            "  câu, không phải độ lớn con số. Cần 60–80 câu mới kết luận được về độ lớn."
        )
    print("=" * 100)


def _day_ket_qua(danh_sach_hit: list[dict], frame_map, limit: int = 100) -> list[dict]:
    """Lọc trùng một danh sách hit, trả về top-limit (giữ nguyên khoá 'ranks' nếu có)."""
    try:
        return deduplicate_temporal(
            danh_sach_hit, frame_map=frame_map, window_seconds=2, limit=limit
        )
    except Exception:  # noqa: BLE001
        return danh_sach_hit[:limit]


def phan_ra_dong_gop_ocr(danh_sach: list[dict], ocr_engine: OCRReranker, top_diem: int = 10) -> None:
    """
    So RRF-có-OCR với RRF-chỉ-Visual THEO TỪNG CÂU để đo ĐÚNG đóng góp của OCR.

    QUAN TRỌNG — vì sao baseline là 'RRF chỉ-visual', KHÔNG phải 'visual thô':
    reciprocal_rank_fusion gộp key trùng bằng rrf_scores[key] += ... Nếu visual_hits
    chứa cùng (video_id, frame_idx) hai lần (keyframe kề nhau làm tròn cùng frame_idx),
    frame đó bị CỘNG DỒN điểm → nhảy lên trên frame đơn hạng cao hơn → RRF tự đảo thứ
    tự các frame visual VỚI NHAU dù OCR không làm gì. Lấy baseline là visual thô sẽ bắt
    nhầm sự đảo này thành 'OCR đổi'. Dùng RRF-chỉ-visual thì artifact đó triệt tiêu ở
    cả hai vế, phần chênh còn lại mới THẬT là do OCR.
    """
    frame_map = load_frame_map()

    n_ocr_rong = 0        # OCR không trả gì
    n_khong_doi = 0       # OCR có trả nhưng không đổi gì so với RRF-chỉ-visual
    n_doi_ngoai_top = 0   # OCR đổi, nhưng chỗ đổi đầu tiên ngoài top_diem
    n_doi_trong_top = 0   # OCR đổi, và chạm vào top_diem (vùng scoring quan tâm)
    n_top1_doi = 0        # OCR đổi cả frame hạng 1 (đã trừ artifact trùng-key)
    tong_ocr_only_trong_top = 0
    n_rrf_lech_visual = 0  # số câu RRF-chỉ-visual KHÁC visual thô -> dấu vết bug trùng-key

    print(
        f"\n{'CÂU':<8} | {'LOẠI':<5} | {'#OCR':>5} | {'#OCR-only@100':>13} | "
        f"{'hạng OCR cao nhất':>17} | {'top1(OCR)':>9} | TRẠNG THÁI (do OCR)"
    )
    print("-" * 100)

    for item in danh_sach:
        q_id = item["query_id"]
        task = item["task"]
        query_text = item["text"]

        q_vec = clip_encoder.encode_text(query_text)
        vhits = search_by_vector(q_vec, top_k=200)
        visual_hits = [
            {"video_id": h.video_id, "frame_idx": h.frame_idx, "score": float(h.score)}
            for h in vhits
        ]
        ocr_hits = ocr_engine.search_ocr(query_text, top_k=200)

        # (a) visual thô — chỉ để đo artifact trùng-key của RRF
        raw_vis_seq = [
            (str(r["video_id"]), int(r["frame_idx"]))
            for r in _day_ket_qua(visual_hits, frame_map)
        ]

        # (b) BASELINE ĐÚNG: RRF chỉ-visual (cùng pipeline, bỏ nhánh OCR)
        base = reciprocal_rank_fusion(
            {"visual": visual_hits}, weights={"visual": 1.0}, k_rrf=60
        )
        base_seq = [
            (str(r["video_id"]), int(r["frame_idx"])) for r in _day_ket_qua(base, frame_map)
        ]

        # (c) TREATMENT: RRF có OCR (đúng nhánh adaptive như mode 'rrf')
        if len(ocr_hits) > 0:
            with_ = reciprocal_rank_fusion(
                {"ocr": ocr_hits, "visual": visual_hits},
                weights={"ocr": 0.45, "visual": 1.0},
                k_rrf=60,
            )
        else:
            with_ = base  # OCR rỗng -> giống hệt baseline
        with_dedup = _day_ket_qua(with_, frame_map)
        with_seq = [(str(r["video_id"]), int(r["frame_idx"])) for r in with_dedup]

        # Dấu vết bug trùng-key: RRF-chỉ-visual có khác visual thô không?
        if raw_vis_seq != base_seq:
            n_rrf_lech_visual += 1

        # Frame OCR-only trong top-100: provenance chỉ có 'ocr'
        ocr_only_pos = [
            i + 1
            for i, r in enumerate(with_dedup)
            if isinstance(r.get("ranks"), dict) and "visual" not in r["ranks"]
        ]
        so_ocr_only = len(ocr_only_pos)
        hang_ocr_cao_nhat = min(ocr_only_pos) if ocr_only_pos else None
        tong_ocr_only_trong_top += sum(1 for p in ocr_only_pos if p <= top_diem)

        # Vị trí ĐẦU TIÊN base khác with -> chỗ cao nhất OCR THẬT SỰ chạm tới
        vi_tri_khac = None
        for i in range(min(len(base_seq), len(with_seq))):
            if base_seq[i] != with_seq[i]:
                vi_tri_khac = i + 1
                break
        if vi_tri_khac is None and len(base_seq) != len(with_seq):
            vi_tri_khac = min(len(base_seq), len(with_seq)) + 1

        top1_doi = bool(base_seq[:1] != with_seq[:1])
        if top1_doi:
            n_top1_doi += 1

        if len(ocr_hits) == 0:
            trang_thai = "OCR rỗng"
            n_ocr_rong += 1
        elif vi_tri_khac is None:
            trang_thai = "OCR không đổi gì"
            n_khong_doi += 1
        elif vi_tri_khac <= top_diem:
            trang_thai = f"ĐỔI ở hạng {vi_tri_khac} (trong top-{top_diem})"
            n_doi_trong_top += 1
        else:
            trang_thai = f"đổi ở hạng {vi_tri_khac} (ngoài top-{top_diem})"
            n_doi_ngoai_top += 1

        print(
            f"{q_id:<8} | {task:<5} | {len(ocr_hits):>5} | {so_ocr_only:>13} | "
            f"{(str(hang_ocr_cao_nhat) if hang_ocr_cao_nhat else '-'):>17} | "
            f"{('có' if top1_doi else '—'):>9} | {trang_thai}"
        )

    tong = len(danh_sach)
    n_co_doi = n_doi_trong_top + n_doi_ngoai_top

    print("=" * 100)
    print("TÓM TẮT ĐÓNG GÓP CỦA OCR (so với RRF-chỉ-Visual — đã trừ artifact trùng-key):")
    print(f"  Tổng số câu:                                {tong}")
    print(f"  OCR không trả kết quả nào:                  {n_ocr_rong}")
    print(f"  OCR có trả nhưng KHÔNG đổi gì:              {n_khong_doi}")
    print(f"  OCR làm ĐỔI danh sách:                      {n_co_doi}")
    print(f"     ├─ đổi CHẠM vào top-{top_diem} (được chấm):     {n_doi_trong_top}")
    print(f"     └─ đổi chỉ NGOÀI top-{top_diem} (ít/không ăn điểm): {n_doi_ngoai_top}")
    print(f"  Số câu OCR làm đổi cả frame hạng 1:         {n_top1_doi}")
    print(f"  Tổng frame OCR-only lọt vào top-{top_diem} (toàn bộ): {tong_ocr_only_trong_top}")
    print("-" * 100)
    print("CỜ RIÊNG (không phải OCR — bug pipeline cần báo Nghi/Ngân):")
    print(f"  Số câu RRF-chỉ-visual KHÁC visual thô:      {n_rrf_lech_visual}")
    if n_rrf_lech_visual > 0:
        print(
            "  -> reciprocal_rank_fusion đang CỘNG DỒN điểm cho (video_id, frame_idx) trùng,\n"
            "     tự đảo thứ tự frame visual dù không có OCR. Sửa: trong mỗi nhánh, khử trùng\n"
            "     key giữ hạng nhỏ nhất TRƯỚC khi fuse (hoặc dedupe theo pts_time trước RRF)."
        )
    print("-" * 100)
    if n_doi_trong_top == 0:
        print(
            f"  KẾT LUẬN: OCR CHƯA BAO GIỜ chạm top-{top_diem}. Nhưng gốc rễ KHÔNG phải trọng số:\n"
            f"  OCR rỗng ở {n_ocr_rong}/{tong} câu vì extract_ocr_keywords chỉ moi từ khoá khi câu có\n"
            "  chữ trong ngoặc kép / IN HOA / từ mồi ('tên','hiệu'...). Câu tả cảnh không có →\n"
            "  OCR im. Chỉnh trọng số 0.45 là vô nghĩa khi nhánh rỗng. Việc cần làm: (1) bộ dev\n"
            "  cần câu neo-chữ để OCR có đất diễn; (2) xem lại cách sinh câu truy vấn OCR."
        )
    else:
        print(
            f"  KẾT LUẬN: OCR chạm top-{top_diem} ở {n_doi_trong_top} câu. Đối chiếu với điểm từng câu\n"
            "  (scoring_results.csv) để xem chạm đó có thành điểm không."
        )
    print("=" * 100)


def main():
    print("=" * 75)
    print("NGHIỆM THU TASK 10: ĐỐI SOÁT ĐIỂM RRF FUSION TRÊN BỘ DEV (32 CÂU)")
    print("===========================================================================\n")

    danh_sach = doc_tep_cau_hoi(DEV_QUERIES_PATH)
    ocr_engine = OCRReranker()

    print("[1/3] Đang chạy kiểm thử chế độ: Visual Only...")
    t0 = time.time()
    sum_vis, avg_vis = run_experiment("visual", danh_sach, ocr_engine)
    print(f" -> Tổng điểm: {sum_vis:.3f} | ĐTB (32 câu): {avg_vis:.4f} ({time.time() - t0:.1f}s)")

    print("\n[2/3] Đang chạy kiểm thử chế độ: OCR Only...")
    t0 = time.time()
    sum_ocr, avg_ocr = run_experiment("ocr", danh_sach, ocr_engine)
    print(f" -> Tổng điểm: {sum_ocr:.3f} | ĐTB (32 câu): {avg_ocr:.4f} ({time.time() - t0:.1f}s)")

    print("\n[3/3] Đang chạy kiểm thử chế độ: Visual + OCR (Adaptive RRF Fusion)...")
    t0 = time.time()
    sum_rrf, avg_rrf = run_experiment("rrf", danh_sach, ocr_engine)
    print(f" -> Tổng điểm: {sum_rrf:.3f} | ĐTB (32 câu): {avg_rrf:.4f} ({time.time() - t0:.1f}s)")

    print("\n" + "=" * 75)
    print(f"{'CHẾ ĐỘ TÌM KIẾM':<32} | {'TỔNG ĐIỂM':<15} | {'ĐTB (32 CÂU DEV)':<20}")
    print("-" * 75)
    print(f"{'1. Chỉ dùng Hình ảnh (Visual)':<32} | {sum_vis:<15.3f} | {avg_vis:<20.4f}")
    print(f"{'2. Chỉ dùng Văn bản (OCR)':<32} | {sum_ocr:<15.3f} | {avg_ocr:<20.4f}")
    print(f"{'3. Gộp RRF (Visual + OCR)':<32} | {sum_rrf:<15.3f} | {avg_rrf:<20.4f}")
    print("=" * 75)

    if sum_rrf >= max(sum_vis, sum_ocr) and sum_rrf > 0:
        print("\n[ĐẠT YÊU CẦU NGHIỆM THU TASK 10] Điểm RRF >= max(Visual, OCR).")
    else:
        print("\n[CHƯA ĐẠT]")

    # Đối chiếu điểm từng câu — bằng chứng quy nguồn cho chênh lệch tổng.
    doi_chieu_diem_tung_cau()

    # -----------------------------------------------------------------------
    # PHÂN RÃ: OCR có bao giờ đổi kết quả so với Visual không, và đổi ở đâu?
    # Chạy thêm một lượt qua 32 câu (chỉ encode + tra, không chấm điểm).
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("[PHÂN RÃ] ĐÓNG GÓP THỰC CỦA OCR — SO RRF VỚI VISUAL THEO TỪNG CÂU")
    print("=" * 100)
    t0 = time.time()
    phan_ra_dong_gop_ocr(danh_sach, ocr_engine, top_diem=10)
    print(f"(phân rã xong trong {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()