"""
Nghiệm thu Task 10 — so ba chế độ: chỉ CLIP / chỉ OCR / gộp RRF.

CÁCH CHẠY (từ gốc D:\\AIC2026):
    python -u -m scripts.benchmark_rrf

KHÁC BẢN TRƯỚC Ở CHỖ NÀO — VÀ VÌ SAO QUAN TRỌNG
------------------------------------------------
Bản trước tự ghi CSV bằng tay:

    f.write(f"{r['video_id']},{r['frame_idx']}\\n")      # MỖI MỐC MỘT DÒNG

Định dạng TRAKE bắt buộc là `video_id,frame_1,frame_2,...,frame_n` — NHIỀU MỐC
TRÊN CÙNG MỘT DÒNG (xem submit/formatter.py, hàm Answer.to_row). Ghi mỗi mốc một
dòng là sai cấu trúc, nên cả 8 câu TRAKE của bộ dev chắc chắn 0 điểm bất kể truy
hồi tốt đến đâu. Đó KHÔNG phải vấn đề keyframe thưa như tưởng ban đầu.

Bản này đi qua `run_queries()` của Ngân nên dùng lại toàn bộ phần đã đúng:

  * `_gom_trake()`       gom đúng số mốc vào một dòng, xếp theo thời gian tăng dần
  * `loc_trung()`        lọc trùng theo `pts_time` THẬT, không giả định fps cứng
  * `SubmissionBudget`   khử trùng dòng, giữ trần 100 dòng
  * `tim_cap_qua_gan()`  soát lại cổng thoát số 5 bằng phép kiểm độc lập

Ba chế độ đi qua ĐÚNG MỘT đường ống, chỉ khác nguồn ứng viên — nên chênh lệch đo
được là chênh lệch của nguồn, không lẫn khác biệt về cách ghi tệp.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# --- Bootstrap sys.path: chạy được cả khi chưa `pip install -e .` -----------
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

# --- stdout: UTF-8 + line-buffered ------------------------------------------
# Trên Windows, output bị pipe/redirect thì Python lùi về cp1252 và chết khi in
# chữ có dấu (UnicodeEncodeError trên 'Ự', 'Ấ'...). Line-buffered để dòng đã in
# không kẹt trong buffer nếu tiến trình chết giữa chừng.
for _luong in (sys.stdout, sys.stderr):
    try:
        _luong.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:  # noqa: BLE001
        pass

# --- torch TRƯỚC pandas -----------------------------------------------------
# pandas kéo numpy dùng runtime MKL/OpenMP riêng. numpy-MKL nạp trước torch thì
# khi torch rồi faiss nạp OpenMP của chúng vào sẽ trùng libiomp5md.dll và tiến
# trình chết ngay lúc import với 0xC0000005 (-1073741819): không traceback, màn
# hình trống. KMP_DUPLICATE_LIB_OK không cứu được vì đây là lỗi lúc nạp DLL.
import torch  # noqa: F401,E402

import pandas as pd  # noqa: E402

from aic2026.paths import DERIVED_DIR, DEV_QUERIES_PATH, SUBMISSIONS_DIR  # noqa: E402
from aic2026.index.fts_index import TextSearchIndex  # noqa: E402
from aic2026.paths import DATA_ROOT  # noqa: E402
from aic2026.rank.hop_nhat import TRONG_SO_MAC_DINH, tim_ung_vien_gop  # noqa: E402
from aic2026.rank.ocr_rerank import OCRReranker  # noqa: E402
from aic2026.rank.search import run_queries  # noqa: E402

# Dùng lại bộ đọc câu hỏi của Ngân trong scripts/run_search.py thay vì tự viết.
# Nó phân giải SỐ MỐC TRAKE theo bốn tầng ưu tiên: khai thẳng trong record ->
# len(cac_giai_doan) -> đếm '(1) (2) (3)' trong câu chữ -> hằng số mặc định.
#
# Bản benchmark trước tự đọc và chỉ tìm khoá 'so_moc_trake' — khoá KHÔNG có
# trong lược đồ bộ dev (lược đồ thật dùng 'cac_giai_doan'), nên mọi câu đều rơi
# về mặc định 4 mốc. Câu 07 chỉ có 2 mốc, câu 08 có 5 -> sai số cột trên mỗi
# dòng nộp -> 0 điểm bảo đảm, bất kể truy hồi tốt đến đâu.
from scripts.run_search import doc_tep_cau_hoi  # noqa: E402

# Mỗi chế độ = một tổ hợp nguồn. Bốn chế độ đầu là NGUỒN ĐƠN (để biết mỗi nguồn
# tự nó làm được gì), hai chế độ sau là tổ hợp.
CAU_HINH_CHE_DO = {
    # --- nguồn đơn: mỗi nguồn tự nó làm được gì ---
    "clip":    dict(dung_clip=True),
    "ocr":     dict(dung_ocr=True),
    "ocr_fts": dict(dung_ocr_fts=True),
    "asr":     dict(dung_asr=True),
    # --- tổ hợp ---
    "rrf2":    dict(dung_clip=True, dung_ocr=True),
    "rrf3":    dict(dung_clip=True, dung_ocr=True, dung_asr=True),
    # Thay nhánh OCR khớp-từ-khoá bằng nhánh OCR BM25. Lý do: đo trên bộ dev,
    # ocr_fts (4.400) hơn ocr (3.800) dù kho chữ FTS đang ít bản ghi hơn.
    "rrf3b":   dict(dung_clip=True, dung_ocr_fts=True, dung_asr=True),
    # Cả bốn nguồn, để biết giữ CẢ HAI nhánh OCR có hơn chọn một hay không.
    "rrf4":    dict(dung_clip=True, dung_ocr=True, dung_ocr_fts=True, dung_asr=True),
}

CAC_CHE_DO = tuple(CAU_HINH_CHE_DO)

NHAN_CHE_DO = {
    "clip":    "Chỉ Hình ảnh (CLIP)",
    "ocr":     "Chỉ OCR (OCRReranker)",
    "ocr_fts": "Chỉ OCR (FTS/BM25)",
    "asr":     "Chỉ Lời nói (ASR)",
    "rrf2":    "CLIP + OCR",
    "rrf3":    "CLIP + OCR + ASR",
    "rrf3b":   "CLIP + OCR_FTS + ASR",
    "rrf4":    "CLIP + OCR + OCR_FTS + ASR",
}

# Chế độ dùng làm mốc so sánh trong bảng đối chiếu từng câu.
CHE_DO_MOC = "clip"


def chay_mot_che_do(che_do: str, danh_sach: list[dict], ocr_engine, kho_chu) -> tuple[float, float]:
    """Chạy trọn bộ câu hỏi ở một chế độ, ghi tệp nộp rồi chấm điểm."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for p in SUBMISSIONS_DIR.glob("*.csv"):
        p.unlink()

    nguon = tim_ung_vien_gop(
        ocr_engine=ocr_engine,
        kho_chu=kho_chu,
        trong_so=TRONG_SO_MAC_DINH,
        **CAU_HINH_CHE_DO[che_do],
    )

    ket_qua = run_queries(danh_sach, ghi_tep=True, tim_ung_vien=nguon)

    # Cổng thoát số 5 do chính mạch của Ngân soát. Câu nào trượt thì phải biết
    # ngay, đừng để lẫn vào điểm thấp rồi tưởng do truy hồi kém.
    truot = [kq for kq in ket_qua if not kq.dat_cong_5]
    if truot:
        print(
            f"    [CẢNH BÁO] {len(truot)} câu TRƯỢT cổng thoát 5: "
            f"{[kq.query_id for kq in truot]}",
            file=sys.stderr,
        )

    for kq in ket_qua:
        for cb in kq.canh_bao:
            print(f"    [{kq.query_id}] {cb}", file=sys.stderr)

    # --- chấm điểm ---
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    chay = subprocess.run(
        [sys.executable, "-m", "scripts.run_scoring"],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        env=env,
        encoding="utf-8",
        errors="replace",
    )

    if chay.returncode != 0:
        print("    [LỖI] scripts.run_scoring thất bại:", file=sys.stderr)
        print(chay.stdout, file=sys.stderr)
        print(chay.stderr, file=sys.stderr)
        raise RuntimeError(f"run_scoring exit {chay.returncode} ở chế độ '{che_do}'")

    tep_diem = DERIVED_DIR / "eval" / "scoring_results.csv"
    if not tep_diem.exists():
        print(f"    [CẢNH BÁO] không thấy {tep_diem} sau khi chấm.", file=sys.stderr)
        return 0.0, 0.0

    df = pd.read_csv(tep_diem)

    # run_scoring LUÔN ghi đè scoring_results.csv. Phải chép ra bản riêng NGAY,
    # nếu không chạy xong chế độ sau là mất bản chế độ trước và không còn cách
    # nào quy điểm về từng câu.
    df.to_csv(DERIVED_DIR / "eval" / f"scoring_results_{che_do}.csv", index=False)

    cot = "final_score" if "final_score" in df.columns else df.columns[-1]
    tong = float(df[cot].sum())

    return tong, tong / max(len(danh_sach), 1)


def doi_chieu_tung_cau(so_cau: int) -> None:
    """So điểm từng câu giữa ba chế độ — bằng chứng quy nguồn cho chênh lệch."""
    thu_muc = DERIVED_DIR / "eval"
    bang = {}

    for che_do in CAC_CHE_DO:
        f = thu_muc / f"scoring_results_{che_do}.csv"
        if not f.exists():
            print(f"  [THIẾU] {f.name} — không đối chiếu được.", file=sys.stderr)
            return
        bang[che_do] = pd.read_csv(f)

    cot = "final_score" if "final_score" in bang[CHE_DO_MOC].columns else bang[CHE_DO_MOC].columns[-1]

    gop = bang[CHE_DO_MOC][["query_id", "task", cot]].rename(columns={cot: CHE_DO_MOC})
    for che_do in CAC_CHE_DO[1:]:
        gop = gop.merge(
            bang[che_do][["query_id", cot]].rename(columns={cot: che_do}),
            on="query_id",
            how="outer",
        )

    gop = gop.fillna(0.0).sort_values("query_id")
    gop["chenh"] = gop[CAC_CHE_DO[-1]] - gop[CHE_DO_MOC]

    print("\n" + "=" * 100)
    print("[ĐỐI CHIẾU] ĐIỂM TỪNG CÂU: CLIP vs OCR vs RRF")
    print("=" * 120)
    dau_cot = " | ".join(f"{m:>8}" for m in CAC_CHE_DO)
    print(f"{'CÂU':<6} | {'LOẠI':<6} | {dau_cot} | {'CHÊNH':>8}")
    print("-" * 120)

    for _, r in gop.iterrows():
        dau = "  <<<" if abs(r["chenh"]) > 1e-9 else ""
        o = " | ".join(f"{r[m]:>8.2f}" for m in CAC_CHE_DO)
        print(
            f"{str(r['query_id']):<6} | {str(r.get('task', '')):<6} | {o} | "
            f"{r['chenh']:>+8.2f}{dau}"
        )

    print("-" * 120)

    # Điểm theo từng dạng — 8 câu TRAKE từng bằng 0 do sai định dạng tệp nộp,
    # nên tách riêng để thấy ngay dạng nào còn trắng điểm.
    if "task" in gop.columns:
        print("Theo dạng câu hỏi:")
        for task, nhom in gop.groupby("task"):
            print(
                f"  {str(task):<8}: "
                + "  ".join(f"{m} {nhom[m].sum():>5.2f}" for m in CAC_CHE_DO)
                + f"   / {len(nhom)} câu"
            )
        print("-" * 120)

    khac = gop[gop["chenh"].abs() > 1e-9]
    print(
        "TỔNG: "
        + "  ".join(f"{m}={gop[m].sum():.3f}" for m in CAC_CHE_DO)
        + f"  chênh={gop['chenh'].sum():+.3f}"
    )

    if khac.empty:
        print("  Gộp KHÔNG đổi điểm câu nào.")
    else:
        tang = khac[khac["chenh"] > 0]
        giam = khac[khac["chenh"] < 0]
        print(f"  Gộp LÀM TĂNG điểm: {len(tang)} câu -> {list(tang['query_id'])}")
        print(f"  Gộp LÀM GIẢM điểm: {len(giam)} câu -> {list(giam['query_id'])}")

    print(
        f"\n  LƯU Ý ĐỌC SỐ: bộ dev {so_cau} câu có sai số chuẩn ~±9 điểm phần trăm.\n"
        "  Chênh lệch vài phần mười điểm nằm trong nhiễu — thứ đáng tin là CƠ CHẾ\n"
        "  truy được ở từng câu, không phải độ lớn con số. Cần 60–80 câu mới kết\n"
        "  luận được về độ lớn."
    )
    print("=" * 120)


def main() -> None:
    print("=" * 75)
    print("NGHIỆM THU TASK 10 — ĐỐI SOÁT ĐIỂM RRF TRÊN BỘ DEV")
    print("=" * 75)

    danh_sach = doc_tep_cau_hoi(DEV_QUERIES_PATH)

    # doc_tep_cau_hoi() giữ lại câu lỗi kèm khoá __normalize_error__ để các câu
    # khác vẫn chạy. Ở đây phải loại ra, nhưng PHẢI in tên câu — nuốt lặng thì
    # tổng điểm tụt mà không ai biết vì sao.
    loi = [m for m in danh_sach if m.get("__normalize_error__")]
    for m in loi:
        print(
            f"    [BỎ] câu {m['query_id']}: {m['__normalize_error__']}",
            file=sys.stderr,
        )
    danh_sach = [m for m in danh_sach if not m.get("__normalize_error__")]

    print(f"Đọc được {len(danh_sach)} câu từ {DEV_QUERIES_PATH.name}")

    # Số mốc TRAKE lấy từ đâu — in ra để đối chiếu, đây là chỗ từng sai âm thầm.
    for m in danh_sach:
        if m.get("task") == "trake":
            print(
                f"    TRAKE câu {m['query_id']}: {m.get('so_moc_trake')} mốc "
                f"({m.get('nguon_so_moc', '?')})"
            )
    print()

    ocr_engine = OCRReranker()
    print(
        f"OCRReranker: {len(ocr_engine._index)} khung hình  |  "
        f"box giữ {ocr_engine.so_box_giu} / bỏ {ocr_engine.so_box_bo} "
        f"(ngưỡng conf {ocr_engine.nguong_conf})"
    )

    # Kho chữ FTS phục vụ hai nguồn: ocr_fts và asr. Thiếu nó thì hai chế độ đó
    # ra rỗng — phải BÁO RÕ, đừng để lặng lẽ 0 điểm rồi tưởng thuật toán kém.
    tep_fts = DATA_ROOT / "index" / "fts" / "text.sqlite"
    kho_chu = None

    if tep_fts.exists():
        try:
            kho_chu = TextSearchIndex(db_path=tep_fts)
            import sqlite3

            with sqlite3.connect(tep_fts) as _c:
                so_ocr = _c.execute("select count(*) from ocr_fts").fetchone()[0]
                so_asr = _c.execute("select count(*) from asr_fts").fetchone()[0]

            print(f"Kho chữ FTS: ocr_fts {so_ocr} bản ghi | asr_fts {so_asr} bản ghi")

            if so_asr == 0:
                print(
                    "    [CẢNH BÁO] asr_fts RỖNG — chế độ ASR sẽ ra 0 điểm vì thiếu\n"
                    "    dữ liệu, KHÔNG phải vì thuật toán kém. Nạp bằng:\n"
                    "    python -m scripts.nap_asr_vao_fts",
                    file=sys.stderr,
                )
            if so_ocr == 0:
                print(
                    "    [CẢNH BÁO] ocr_fts RỖNG — chế độ ocr_fts sẽ ra 0 điểm vì thiếu\n"
                    "    dữ liệu. Nạp bằng build_index_from_jsonl_dir(DERIVED_DIR/'ocr').",
                    file=sys.stderr,
                )
        except Exception as loi:  # noqa: BLE001
            print(f"    [LỖI] không mở được kho chữ: {loi}", file=sys.stderr)
    else:
        print(f"    [CẢNH BÁO] không thấy {tep_fts} — bỏ qua ocr_fts và ASR.", file=sys.stderr)

    print()

    diem: dict[str, tuple[float, float]] = {}

    for i, che_do in enumerate(CAC_CHE_DO, start=1):
        print(f"[{i}/{len(CAC_CHE_DO)}] Đang chạy: {NHAN_CHE_DO[che_do]}...")
        moc = time.time()
        diem[che_do] = chay_mot_che_do(che_do, danh_sach, ocr_engine, kho_chu)
        tong, dtb = diem[che_do]
        print(f" -> Tổng điểm: {tong:.3f} | ĐTB: {dtb:.4f} ({time.time() - moc:.1f}s)\n")

    print("=" * 75)
    print(f"{'CHẾ ĐỘ TÌM KIẾM':<32} | {'TỔNG ĐIỂM':<15} | {'ĐTB':<20}")
    print("-" * 75)
    for che_do in CAC_CHE_DO:
        tong, dtb = diem[che_do]
        print(f"{NHAN_CHE_DO[che_do]:<32} | {tong:<15.3f} | {dtb:<20.4f}")
    print("=" * 75)

    NGUON_DON = ("clip", "ocr", "ocr_fts", "asr")
    TO_HOP = ("rrf2", "rrf3", "rrf3b", "rrf4")

    ten_don_tot = max(NGUON_DON, key=lambda m: diem[m][0])
    ten_to_hop_tot = max(TO_HOP, key=lambda m: diem[m][0])

    tot_nhat_don = diem[ten_don_tot][0]
    tong_rrf = diem[ten_to_hop_tot][0]

    print(f"\nNguồn đơn tốt nhất : {NHAN_CHE_DO[ten_don_tot]} = {tot_nhat_don:.3f}")
    print(f"Tổ hợp tốt nhất    : {NHAN_CHE_DO[ten_to_hop_tot]} = {tong_rrf:.3f}")

    if tong_rrf >= tot_nhat_don and tong_rrf > 0:
        print("\n[ĐẠT] Điểm RRF >= max(CLIP, OCR).")
    else:
        print("\n[CHƯA ĐẠT]")

    print(
        "  Lưu ý: tiêu chí này còn yếu — một phép gộp hoàn toàn vô hiệu cũng thoả.\n"
        "  Đọc kèm bảng đối chiếu bên dưới để biết gộp có thật sự đổi gì không."
    )

    doi_chieu_tung_cau(len(danh_sach))


if __name__ == "__main__":
    main()