import sys; sys.path.insert(0, "src")
from aic2026.index.fts_index import TextSearchIndex
from aic2026.paths import FTS_DIR
kho = TextSearchIndex(FTS_DIR / "text.sqlite")
cau = ("Tuong mau do cua mot vi quan dung tren be co bang ghi Mac Cuu 1655-1735, "
       "phia sau la gian trung bay")
for q in [cau, "Mac Cuu 1655 1735", "Mac Cuu"]:
    r = kho.search_text(q, top_k=300)
    hang = next((i for i, x in enumerate(r, 1) if x["video_id"] == "L27_V007"), None)
    print(f"{len(q):>4} ky tu | {len(r):>3} ket qua | L27_V007 o hang: {hang}")
    print("        ", repr(q[:60]))
