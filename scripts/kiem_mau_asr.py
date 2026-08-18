"""
Việc 15 — CỔNG THOÁT: "nghe thử 10 đoạn 30 giây lấy từ cả bốn phần, đối chiếu
chữ, mốc thời gian lệch không quá 2 giây."

Lệnh này KHÔNG tự chấm được — chỉ có tai người mới đối chiếu được lời nói với
chữ. Việc của nó là chuẩn bị sẵn mọi thứ để việc nghe mất 15 phút chứ không
mất một buổi:

  1. bốc mẫu NGẪU NHIÊN NHƯNG LẶP LẠI ĐƯỢC (có --seed), rải đều các mã video
  2. cắt sẵn 10 tệp .wav dài 30 giây
  3. in sẵn phiếu kiểm: mỗi đoạn, chữ ASR kèm mốc TÍNH TỪ ĐẦU ĐOẠN CẮT
     (không phải từ đầu video) — nghe tới giây thứ mấy thì nhìn thẳng vào
     cột đó, không phải trừ nhẩm

Cách chạy (PowerShell):
    python -m scripts.kiem_mau_asr                 # 10 đoạn, seed mặc định
    python -m scripts.kiem_mau_asr --so-mau 10 --seed 2026
    python -m scripts.kiem_mau_asr --tien-to L23 L27 L30

Sau khi nghe: điền cột "lệch thực tế" trong phiếu, rồi dán kết quả vào sheet
Nhật ký. Đoạn nào lệch quá 2 giây thì GHI RÕ video nào — đó là dấu hiệu tệp
.wav của video đó có luồng tiếng bắt đầu trễ, phải xử lý riêng.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    from src.aic2026.paths import ASR_DIR, AUDIO_DIR
except ImportError as e:
    print("Không import được aic2026. Đã chạy 'pip install -e .' chưa?")
    print(f"Chi tiết: {e}")
    sys.exit(1)

DAI_DOAN_GIAY = 30.0
# Lùi lại một chút trước câu được bốc, để người nghe bắt được đầu câu
LUI_TRUOC_GIAY = 2.0
THU_MUC_MAU = "_mau_kiem"


def doc_cac_doan(duong_dan: Path) -> list[dict]:
    """Đọc một tệp .jsonl, chỉ lấy dòng có chữ."""
    cac_doan = []
    with duong_dan.open("r", encoding="utf-8") as f:
        for dong in f:
            if not dong.strip():
                continue
            ban_ghi = json.loads(dong)
            if (ban_ghi.get("text") or "").strip():
                cac_doan.append(ban_ghi)
    return cac_doan


def ma_bo(video_id: str) -> str:
    """L23_V001 -> L23. Dùng để rải mẫu đều các mã, không dồn vào một mã."""
    return video_id.split("_")[0]


def cat_doan(wav_goc: Path, bat_dau: float, dai: float, dich: Path) -> None:
    """Cắt một đoạn .wav bằng ffmpeg. -ss đặt TRƯỚC -i để cắt nhanh."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{bat_dau:.3f}", "-t", f"{dai:.3f}",
        "-i", str(wav_goc),
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-f", "wav", str(dich),
    ]
    kq = subprocess.run(cmd, capture_output=True)
    if kq.returncode != 0:
        loi = kq.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg lỗi: {loi[-1] if loi else 'không rõ'}")


def main():
    p = argparse.ArgumentParser(description="Việc 15 — bốc mẫu kiểm mốc thời gian ASR")
    p.add_argument("--so-mau", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026,
                   help="Cùng seed thì bốc ra đúng cùng bộ mẫu — để hai người kiểm chéo được.")
    p.add_argument("--tien-to", nargs="+", default=None,
                   help="Chỉ bốc trong các mã này, vd L23 L27 L30.")
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    if not ASR_DIR.exists():
        print(f"Không có {ASR_DIR}. Chạy Việc 15 trước.")
        sys.exit(1)

    cac_tep = [x for x in sorted(ASR_DIR.glob("*.jsonl")) if not x.name.startswith("_")]
    if args.tien_to:
        tien_to = tuple(args.tien_to)
        cac_tep = [x for x in cac_tep if x.stem.startswith(tien_to)]

    # Chỉ bốc trong những video CÒN tệp .wav — không có tiếng thì không nghe được
    co_wav = [x for x in cac_tep if (AUDIO_DIR / f"{x.stem}.wav").exists()]
    thieu_wav = len(cac_tep) - len(co_wav)
    if not co_wav:
        print("Không video nào vừa có .jsonl vừa còn .wav trong derived/audio/.")
        print("Cổng thoát này bắt buộc phải NGHE, nên phải còn tệp tiếng.")
        sys.exit(1)

    rng = random.Random(args.seed)

    # Rải đều theo mã bộ (L23, L27, ...) rồi mới bốc trong từng nhóm
    theo_ma: dict[str, list[Path]] = defaultdict(list)
    for x in co_wav:
        theo_ma[ma_bo(x.stem)].append(x)

    cac_ma = sorted(theo_ma)
    chon: list[Path] = []
    i = 0
    while len(chon) < args.so_mau:
        nhom = theo_ma[cac_ma[i % len(cac_ma)]]
        con_lai = [x for x in nhom if x not in chon]
        if con_lai:
            chon.append(rng.choice(con_lai))
        elif all(all(x in chon for x in theo_ma[m]) for m in cac_ma):
            break            # đã lấy hết video có thể lấy
        i += 1

    thu_muc = ASR_DIR / THU_MUC_MAU
    thu_muc.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"BỐC {len(chon)} ĐOẠN 30 GIÂY (seed={args.seed}) — mã: {', '.join(cac_ma)}")
    if thieu_wav:
        print(f"({thieu_wav} video có .jsonl nhưng không còn .wav — bỏ qua khi bốc mẫu)")
    print("=" * 70 + "\n")

    dong_phieu: list[str] = [
        "# Phiếu kiểm mốc thời gian ASR — Việc 15",
        "",
        f"- Bốc mẫu bằng: `python -m scripts.kiem_mau_asr --so-mau {args.so_mau} "
        f"--seed {args.seed}` (cùng seed thì ra cùng bộ mẫu)",
        "- Cổng thoát: **mốc thời gian lệch không quá 2 giây**",
        "- Cách kiểm: mở tệp .wav, nghe. Câu trong bảng phải vang lên đúng vào "
        "khoảng giây ghi ở cột *Mốc trong đoạn cắt*.",
        "- Nghe xong điền cột **Lệch thực tế**. Quá 2 giây thì ghi rõ video nào.",
        "",
    ]

    so_cat = 0
    for thu_tu, tep in enumerate(chon, start=1):
        video_id = tep.stem
        cac_doan = doc_cac_doan(tep)
        if not cac_doan:
            print(f"[{thu_tu}] {video_id}: không có đoạn nào có chữ, bỏ qua")
            continue

        moc = rng.choice(cac_doan)
        bat_dau = max(0.0, float(moc["start"]) - LUI_TRUOC_GIAY)
        ket_thuc = bat_dau + DAI_DOAN_GIAY

        ten_clip = f"{video_id}_{int(bat_dau):06d}s.wav"
        dich = thu_muc / ten_clip
        try:
            cat_doan(AUDIO_DIR / f"{video_id}.wav", bat_dau, DAI_DOAN_GIAY, dich)
        except (RuntimeError, OSError) as e:
            print(f"[{thu_tu}] {video_id}: {e}")
            continue

        so_cat += 1
        trong_doan = [
            d for d in cac_doan
            if float(d["end"]) > bat_dau and float(d["start"]) < ket_thuc
        ]

        print(f"[{thu_tu}] {ten_clip}  ({bat_dau:.1f}s–{ket_thuc:.1f}s, "
              f"{len(trong_doan)} câu)")

        dong_phieu += [
            f"## {thu_tu}. `{ten_clip}`",
            "",
            f"Video `{video_id}`, cắt từ giây {bat_dau:.1f} đến {ket_thuc:.1f} của video.",
            "",
            "| Mốc trong đoạn cắt | Mốc trong video | Chữ ASR | Lệch thực tế |",
            "|---|---|---|---|",
        ]
        for d in trong_doan:
            s, e = float(d["start"]), float(d["end"])
            chu = (d["text"] or "").replace("|", "\\|")
            dong_phieu.append(
                f"| {max(0.0, s - bat_dau):.1f}–{min(DAI_DOAN_GIAY, e - bat_dau):.1f}s "
                f"| {s:.2f}–{e:.2f}s | {chu} |  |"
            )
        dong_phieu.append("")

    duong_dan_phieu = thu_muc / "phieu_kiem.md"
    duong_dan_phieu.write_text("\n".join(dong_phieu), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Đã cắt {so_cat} tệp .wav vào: {thu_muc}")
    print(f"Phiếu kiểm                 : {duong_dan_phieu}")
    print("=" * 70)
    print("Mở phiếu, nghe từng tệp, điền cột 'Lệch thực tế'.")
    print("Toàn bộ dưới 2 giây thì Việc 15 qua cổng thoát.")


if __name__ == "__main__":
    main()
