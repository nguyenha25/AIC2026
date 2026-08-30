from src.aic2026.paths import DATA_ROOT, DERIVED_DIR, RAW_DIR
import sqlite3

p = lambda d, pat: len(list(d.glob(pat))) if d.exists() else 0

print('--- DU LIEU THO (BTC phat) ---')
print(f'  keyframes (thu muc lo) : {p(RAW_DIR/"keyframes", "*")}')
print(f'  map-keyframes (*.csv)  : {p(RAW_DIR/"map-keyframes", "*.csv")}')
print(f'  clip-features (*.npy)  : {p(RAW_DIR/"clip-features-32", "*.npy")}')

print('--- DU LIEU DAN XUAT (tu sinh) ---')
print(f'  frame_map.parquet      : {(DERIVED_DIR/"frame_map.parquet").exists()}')
print(f'  OCR (*.jsonl)          : {p(DERIVED_DIR/"ocr", "*.jsonl")}')
print(f'  ASR (*.jsonl)          : {p(DERIVED_DIR/"asr", "*.jsonl")}')
print(f'  thumbnails (thu muc)   : {p(DERIVED_DIR/"thumbnails", "*")}')

print('--- CHI MUC ---')

faiss = DATA_ROOT/'index'/'faiss'/'clip_b32.index'
print(f'  FAISS clip_b32.index   : {faiss.exists()}')

fts = DATA_ROOT/'index'/'fts'/'text.sqlite'
print(f'  text.sqlite            : {fts.exists()}')

if fts.exists():
    c = sqlite3.connect(fts)

    for t in ('ocr_fts', 'asr_fts'):
        try:
            n = c.execute(f'select count(*) from {t}').fetchone()[0]
            v = c.execute(f'select count(distinct video_id) from {t}').fetchone()[0]
            print(f'    {t:<8}: {n} ban ghi / {v} video')
        except Exception as e:
            print(f'    {t:<8}: LOI {e}')

    c.close()