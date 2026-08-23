import sys, json; sys.path.insert(0, "src")
from aic2026.frame_map import load_frame_map
from aic2026.paths import DEV_QUERIES_PATH, FTS_DIR
from aic2026.index.fts_index import TextSearchIndex

q = next(json.loads(l) for l in open(DEV_QUERIES_PATH, encoding="utf-8")
         if l.strip() and json.loads(l)["id"] == "11")
vid, a, b = q["video_id"], int(q["frame_start"]), int(q["frame_end"])
print("cau 11:", vid, "cua so", a, "-", b)

bang = load_frame_map()
nhom = bang[(bang.video_id == vid) & (bang.frame_idx >= a) & (bang.frame_idx <= b)]
print("keyframe trong cua so:", [(int(r.n), int(r.frame_idx)) for r in nhom.itertuples()])
n_dung = {int(r.n) for r in nhom.itertuples()}

kho = TextSearchIndex(FTS_DIR / "text.sqlite")
for ten, truy in [("ca cau", q["cau_hoi"]), ("Mac Cuu", "Mac Cuu")]:
    r = kho.search_text(truy, top_k=1000)
    trung = [(i, x) for i, x in enumerate(r, 1)
             if x["video_id"] == vid and int(x.get("n", -1)) in n_dung]
    print(f"\nOCR [{ten}]: {len(r)} ket qua")
    print("   khung DUNG o hang:", [i for i, _ in trung] or "KHONG co")
    print("   5 hit dau cua video nay:",
          [(i, int(x.get("n", -1))) for i, x in enumerate(r, 1)
           if x["video_id"] == vid][:5])
