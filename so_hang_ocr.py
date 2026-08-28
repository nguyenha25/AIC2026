import sys; sys.path.insert(0, "src")
import json
from aic2026.index.fts_index import TextSearchIndex
from aic2026.paths import DEV_QUERIES_PATH, FTS_DIR
from aic2026.frame_map import load_frame_map

kho = TextSearchIndex(FTS_DIR / "text.sqlite")
bang = load_frame_map()

print(f"{'cau':<5}{'tho':>7}{'hiem':>7}  video")
for l in open(DEV_QUERIES_PATH, encoding="utf-8"):
    if not l.strip():
        continue
    q = json.loads(l)
    if q.get("loai_truy_van") != "mo_ta":
        continue
    vid, a, b = q["video_id"], int(q["frame_start"]), int(q["frame_end"])
    nhom = bang[(bang.video_id == vid) & (bang.frame_idx >= a) & (bang.frame_idx <= b)]
    n_dung = {int(r.n) for r in nhom.itertuples()}

    def hang(loc):
        r = kho.search_text(q["cau_hoi"], top_k=500, loc_hiem=loc)
        return next((i for i, x in enumerate(r, 1)
                     if x["video_id"] == vid and int(x.get("n", -1)) in n_dung), None)

    t, h = hang(False), hang(True)
    dau = " <<<" if (t is None) != (h is None) or (t and h and abs(t - h) > 20) else ""
    print(f"{q['id']:<5}{str(t):>7}{str(h):>7}  {vid}{dau}")
