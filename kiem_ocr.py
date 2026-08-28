import sys; sys.path.insert(0, "src")
from aic2026.index.fts_index import TextSearchIndex
from aic2026.paths import FTS_DIR
kho = TextSearchIndex(FTS_DIR / "text.sqlite")
for tu in ["Mạc Cửu", "Mac Cuu", "1655", "1735"]:
    r = kho.search_text(tu, top_k=5)
    print(f"{tu!r:12} -> {len(r)} ket qua")
    for x in r[:3]:
        print("   ", x["video_id"], "n =", x.get("n"), "|", str(x.get("text",""))[:60])
