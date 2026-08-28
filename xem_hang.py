import json
for t in ["dev_09", "dev_10", "dev_11"]:
    g = json.load(open(rf"D:\aic-data\derived\goi_soi\{t}\du_lieu.json", encoding="utf-8"))
    d = [k for k in g["khung"] if k.get("la_dap_an")]
    hang = [k["hang"] for k in d] if d else "KHONG co trong 300"
    print(t, "|", g["so_khung"], "/", g.get("so_khung_yeu_cau"), "khung co anh | hang:", hang)
    for k in d[:3]:
        print("   ", k["video_id"], "frame", k["frame_idx"], "| nguon", k["nguon"], "| n =", k["n"])
