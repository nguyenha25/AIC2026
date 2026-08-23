"""
Việc 3b — DỰNG BẢNG SOI HTML từ gói soi. Không cần cài gì.

VÌ SAO CÓ ĐƯỜNG THỨ HAI
-----------------------
FiftyOne trên Colab hay vỡ vì xung đột thư viện — đã gặp: nó kéo về một bản
Pillow lệch với bản Colab dựng sẵn, và `import fiftyone` chết ngay ở dòng đầu.
Sửa được, nhưng phải cài lại rồi khởi động lại máy ảo, và lần sau Colab đổi
bản dựng sẵn thì lại vỡ.

Bảng HTML này dùng đúng gói soi đó, sinh ra một tệp .html mở bằng trình duyệt.
Không cài gì, không phụ thuộc phiên bản nào, chạy được cả khi không có mạng.
Đổi lại: không lọc tương tác được như FiftyOne, chỉ xem và đánh dấu.

CÁCH CHẠY
    python -u -m scripts.dung_bang_soi --goi D:/aic-data/derived/goi_soi/dev_25
    python -u -m scripts.dung_bang_soi --goi ... --nhung-anh    # tệp tự chứa

`--nhung-anh` nhúng ảnh vào thẳng tệp HTML (base64) để gửi qua chat cho người
khác xem mà không phải gửi kèm thư mục ảnh. Tệp nặng gấp rưỡi.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; background:#14161a; color:#e6e8eb;
       font:14px/1.5 "Segoe UI",system-ui,sans-serif; }
header { position:sticky; top:0; z-index:9; background:#1c1f24;
         border-bottom:1px solid #2b3038; padding:14px 20px; }
h1 { margin:0 0 4px; font-size:17px; font-weight:600; }
.cau { color:#9aa4b2; font-size:13px; max-width:80ch; }
.thanh { display:flex; gap:16px; align-items:center; margin-top:10px;
         flex-wrap:wrap; }
input[type=search] { background:#0f1114; border:1px solid #333a44; color:#e6e8eb;
                     padding:6px 10px; border-radius:6px; width:280px; font-size:13px; }
label { color:#9aa4b2; font-size:13px; user-select:none; cursor:pointer; }
#dem { color:#6f7885; font-size:13px; }
main { display:grid; gap:12px; padding:18px 20px;
       grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); }
figure { margin:0; background:#1c1f24; border:1px solid #262b33; border-radius:8px;
         overflow:hidden; }
figure.dung { border-color:#3fb950; box-shadow:0 0 0 1px #3fb95055; }
figure.chon { border-color:#58a6ff; box-shadow:0 0 0 1px #58a6ff55; }
.anh { position:relative; line-height:0; cursor:pointer; }
.anh img { width:100%; display:block; }
.hop { position:absolute; border:1.5px solid #f0883e; }
.hop span { position:absolute; top:-17px; left:-1.5px; background:#f0883e;
            color:#14161a; font-size:10px; padding:0 4px; white-space:nowrap;
            border-radius:2px 2px 0 0; font-weight:600; }
figcaption { padding:8px 10px; font-size:12px; }
.hang { color:#6f7885; }
.fid { font-weight:600; color:#e6e8eb; }
.meta { color:#9aa4b2; margin-top:2px; }
.ocr { color:#d29922; margin-top:4px; word-break:break-word;
       max-height:3em; overflow:hidden; }
.nhan { color:#7ee787; margin-top:3px; font-size:11px; }
.dau-dung { color:#3fb950; font-weight:600; }
#ketqua { position:fixed; right:0; bottom:0; width:420px; max-height:46vh;
          background:#0f1114; border-top:1px solid #2b3038;
          border-left:1px solid #2b3038; padding:10px 14px; overflow:auto;
          font-family:Consolas,monospace; font-size:12px; display:none; }
#ketqua.hien { display:block; }
#ketqua h2 { margin:0 0 6px; font-size:12px; color:#9aa4b2;
             font-family:"Segoe UI",sans-serif; }
#ketqua pre { margin:0; white-space:pre-wrap; color:#7ee787; }
button { background:#21262d; border:1px solid #363b42; color:#e6e8eb;
         padding:5px 12px; border-radius:6px; cursor:pointer; font-size:13px; }
button:hover { background:#2b3038; }
"""

JS = """
const luoi = document.getElementById('luoi');
const o = document.getElementById('ketqua');
const pre = o.querySelector('pre');

function capNhat() {
  const chon = [...luoi.querySelectorAll('figure.chon')];
  if (!chon.length) { o.classList.remove('hien'); return; }
  pre.textContent = chon
    .map(f => `${f.dataset.video},${f.dataset.fid}`)
    .join('\\n');
  o.querySelector('h2').textContent =
    `${chon.length} khung đã chọn — video_id,frame_idx`;
  o.classList.add('hien');
}

luoi.addEventListener('click', e => {
  const a = e.target.closest('.anh');
  if (!a) return;
  a.closest('figure').classList.toggle('chon');
  capNhat();
});

document.getElementById('tim').addEventListener('input', e => {
  const t = e.target.value.toLowerCase();
  luoi.querySelectorAll('figure').forEach(f => {
    f.style.display = !t || f.dataset.tim.includes(t) ? '' : 'none';
  });
});

document.getElementById('chi-dung').addEventListener('change', e => {
  luoi.querySelectorAll('figure').forEach(f => {
    f.style.display = !e.target.checked || f.classList.contains('dung') ? '' : 'none';
  });
});

document.getElementById('chep').addEventListener('click', () => {
  navigator.clipboard.writeText(pre.textContent);
});
"""


def _the_anh(k: dict, goc: Path, nhung: bool, thu_muc_anh: str = "anh") -> str:
    duong_dan = goc / "anh" / k["anh"]
    if nhung:
        try:
            b64 = base64.b64encode(duong_dan.read_bytes()).decode()
            src = f"data:image/jpeg;base64,{b64}"
        except OSError:
            src = ""
    else:
        # Đường dẫn TƯƠNG ĐỐI TỪ CHỖ ĐẶT TỆP HTML, không phải từ thư mục gói.
        # Bản đầu luôn ghi "anh/..." trong khi tệp HTML nằm ở thư mục CHA của
        # gói, nên trình duyệt tìm ảnh ở goi_soi/anh/ và không thấy gì —
        # trang hiện đủ chữ, đủ hộp, chỉ trống ảnh.
        src = f"{thu_muc_anh}/{k['anh']}"

    hop = "".join(
        '<div class="hop" style="left:{:.2%};top:{:.2%};width:{:.2%};height:{:.2%}">'
        '<span>{} {:.0%}</span></div>'.format(
            v["hop"][0], v["hop"][1], v["hop"][2], v["hop"][3],
            html.escape(v["nhan"]), v["diem"],
        )
        for v in (k.get("vat_the") or [])
    )
    return f'<div class="anh"><img loading="lazy" src="{src}" alt="">{hop}</div>'


def dung_html(goc: Path, nhung_anh: bool = False, dat_tai: Path | None = None) -> str:
    """dat_tai: nơi sẽ ĐẶT tệp HTML, để tính đường dẫn ảnh cho đúng."""
    with (goc / "du_lieu.json").open("r", encoding="utf-8") as f:
        goi = json.load(f)

    if nhung_anh:
        thu_muc_anh = ""
    else:
        import os

        goc_html = (dat_tai or goc).resolve()
        thu_muc_anh = os.path.relpath(
            (goc / "anh").resolve(), goc_html
        ).replace(os.sep, "/")

    o = []
    for k in goi["khung"]:
        nhan = sorted({v["nhan"] for v in (k.get("vat_the") or [])})
        chu = " | ".join(
            b["chu"] for b in (k.get("ocr") or []) if str(b.get("chu", "")).strip()
        )
        tim = " ".join(
            [k["video_id"], str(k["frame_idx"]), k.get("nguon", ""), chu] + nhan
        ).lower()

        lop = "dung" if k.get("la_dap_an") else ""
        o.append(
            f'<figure class="{lop}" data-video="{html.escape(k["video_id"])}" '
            f'data-fid="{k["frame_idx"]}" data-tim="{html.escape(tim)}">'
            + _the_anh(k, goc, nhung_anh, thu_muc_anh)
            + '<figcaption>'
            f'<span class="hang">#{k["hang"]}</span> '
            f'<span class="fid">{k["frame_idx"]}</span>'
            + (' <span class="dau-dung">ĐÁP ÁN</span>' if k.get("la_dap_an") else "")
            + f'<div class="meta">{html.escape(k["video_id"])} · n={k["n"]} · '
            f'{k["pts_time"]:.1f}s · {html.escape(k.get("nguon", ""))}</div>'
            + (f'<div class="nhan">{html.escape(", ".join(nhan[:6]))}</div>' if nhan else "")
            + (f'<div class="ocr">{html.escape(chu[:160])}</div>' if chu else "")
            + '</figcaption></figure>'
        )

    co_dap_an = any(k.get("la_dap_an") for k in goi["khung"])
    cau = html.escape(goi.get("cau_hoi") or "")

    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<title>Soi ứng viên — {html.escape(goi.get('ten') or '')}</title>
<style>{CSS}</style></head><body>
<header>
  <h1>Soi ứng viên — {html.escape(goi.get('ten') or '')}</h1>
  {'<div class="cau">' + cau + '</div>' if cau else ''}
  <div class="thanh">
    <input type="search" id="tim" placeholder="lọc theo video, nhãn, chữ OCR…">
    {'<label><input type="checkbox" id="chi-dung"> chỉ khung ĐÁP ÁN</label>'
     if co_dap_an else '<label style="opacity:.4"><input type="checkbox" id="chi-dung" disabled> không có mốc đáp án</label>'}
    <span id="dem">{len(goi['khung'])} khung</span>
    <button id="chep">Chép danh sách đã chọn</button>
  </div>
</header>
<main id="luoi">{''.join(o)}</main>
<div id="ketqua"><h2></h2><pre></pre></div>
<script>{JS}</script>
</body></html>"""


def main() -> int:
    p = argparse.ArgumentParser(description="Việc 3b — bảng soi HTML")
    p.add_argument("--goi", required=True, help="thư mục gói soi")
    p.add_argument("--nhung-anh", action="store_true",
                   help="nhúng ảnh vào tệp HTML để gửi đi một mình")
    p.add_argument("--ra", default=None)
    args = p.parse_args()

    goc = Path(args.goi)
    if not (goc / "du_lieu.json").exists():
        print(f"Không thấy {goc / 'du_lieu.json'}. Chạy scripts.xuat_goi_soi trước.")
        return 1

    # Mặc định đặt HTML NGAY TRONG gói, cạnh thư mục anh/. Đặt ở thư mục cha
    # thì đường dẫn ảnh phải lùi một cấp, và chỉ cần ai đó di chuyển tệp là hỏng.
    dich = Path(args.ra) if args.ra else goc / "bang_soi.html"
    dich.parent.mkdir(parents=True, exist_ok=True)
    dich.write_text(
        dung_html(goc, args.nhung_anh, dat_tai=dich.parent), encoding="utf-8"
    )

    print(f"Đã dựng {dich}  ({dich.stat().st_size / 1e6:.1f} MB)")
    if not args.nhung_anh:
        print(
            f"Mở bằng trình duyệt. Ảnh đọc từ {goc / 'anh'} theo đường dẫn tương "
            "đối —\ndi chuyển tệp HTML đi chỗ khác là mất ảnh. Cần gửi đi một "
            "mình thì thêm --nhung-anh."
        )
    print("\nBấm ảnh để chọn; danh sách 'video_id,frame_idx' hiện ở góc dưới phải.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
