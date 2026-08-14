"""
VIỆC 4 — Mạch tìm kiếm đầu–cuối. Chủ tệp: Ngân.

    câu chữ → dãy số → tra kho → LỌC TRÙNG → cắt 100 → ghi tệp nộp

Năm bước, mỗi bước gọi đúng một nhánh của người khác. Tệp này KHÔNG tự tính
toán gì cả — nó chỉ nối và đo. Khi có lỗi, người giữ tệp này là người duy nhất
nhìn ra lỗi thuộc nhánh nào, nên mỗi bước đều ghi lại số liệu riêng.

    bước 1  câu chữ → dãy số      ClipEncoder.encode_text()      Thi, Việc 3
    bước 2  dãy số  → ứng viên    faiss_index.search_by_vector() Thi, Việc 2
    bước 3  lọc trùng             dedupe.loc_trung()             Ngân, đã xong
    bước 4  cắt 100 dòng          submit.SubmissionBudget        Ngân, đã xong
    bước 5  ghi tệp               SubmissionBudget.write()       Ngân, đã xong

BA ĐIỀU DỄ SAI, ĐÃ CHẶN SẴN Ở ĐÂY
---------------------------------
1. NHẦM n VỚI frame_idx. Tệp này KHÔNG bao giờ tự ghi `hit.n` vào tệp nộp.
   Số duy nhất được ghi là `hit.frame_idx`, mà giá trị đó do
   frame_map.lookup() trả về. Không có đường tắt nào khác trong mạch này.

2. LẤY ĐÚNG 100 ỨNG VIÊN RỒI MỚI LỌC TRÙNG. Lọc xong sẽ còn ba bốn chục dòng
   mà không ai thấy, vì tệp nộp vẫn hợp lệ. Ở đây lấy `so_ung_vien_moi_nguon`
   (mặc định 500) rồi mới lọc, và báo cáo in ra con số còn lại sau khi lọc —
   nếu con số đó dưới 100 thì có cảnh báo.

3. TIN VÀO CHÍNH HÀM LỌC TRÙNG. Sau khi lọc xong, hàm ở đây gọi
   dedupe.tim_cap_qua_gan() — một phép kiểm viết bằng cách khác (so mọi cặp) —
   để soát lại kết quả của chính bước 3. Đó chính là cổng thoát số 5.

CHỖ CHỜ NGƯỜI KHÁC
------------------
Việc 10 của Nghi (gộp hai bảng xếp hạng) sẽ cắm vào tham số `tim_ung_vien`
của run_query(): truyền một hàm nhận câu chữ, trả về danh sách Hit đã xếp
hạng. Mạch từ bước 3 trở đi không phải sửa gì.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, TYPE_CHECKING

from ..paths import SUBMISSIONS_DIR
from ..submit import KIS, QA, TRAKE, Answer, SubmissionBudget, submission_filename
from .config import (
    canh_bao_cau_hinh,
    cua_so_giay,
    so_anh_toi_da_moi_video,
    so_dong_toi_da,
    so_ung_vien_moi_nguon,
    tham_so_da_dung,
)
from .dedupe import BaoCaoLocTrung, loc_trung, tim_cap_qua_gan

if TYPE_CHECKING:  # chỉ để gợi ý kiểu, không nạp faiss lúc chạy thật
    from ..index.faiss_index import Hit


# Số mốc mặc định của một dòng TRAKE (settings.yaml: nop_bai.so_moc_trake).
SO_MOC_TRAKE_MAC_DINH = 4


# ---------------------------------------------------------------------------
# Bước 1 — câu chữ thành dãy số
# ---------------------------------------------------------------------------

# ClipEncoder nạp mô hình mất vài giây. Nạp lại mỗi truy vấn thì năm câu thử
# mất nửa phút, và trong vòng thi thì không dùng được. Giữ một bản duy nhất.
_ENCODER: Any = None


def lay_encoder() -> Any:
    """
    Trả về ClipEncoder dùng chung, nạp mô hình đúng một lần cho cả tiến trình.

    Import nằm TRONG hàm chứ không ở đầu tệp: bộ test của bước 3, 4, 5 nhờ vậy
    chạy được trên máy chưa cài open-clip-torch và faiss.
    """
    global _ENCODER

    if _ENCODER is None:
        from ..index.encode.clip_encoder import ClipEncoder

        _ENCODER = ClipEncoder()

    return _ENCODER


def ma_hoa_cau(cau_hoi: str):
    """Câu chữ → vector CLIP 512 chiều đã chuẩn hoá."""
    if not cau_hoi or not cau_hoi.strip():
        raise ValueError("Câu truy vấn rỗng.")

    return lay_encoder().encode_text(cau_hoi.strip())


# ---------------------------------------------------------------------------
# Bước 2 — tra kho
# ---------------------------------------------------------------------------


def tim_ung_vien_clip(cau_hoi: str, so_ung_vien: int) -> list["Hit"]:
    """
    Nguồn ứng viên mặc định: chỉ CLIP.

    Việc 10 của Nghi sẽ thay hàm này bằng bản gộp CLIP + chữ. Chữ ký giữ
    nguyên (câu chữ, số ứng viên) → (danh sách Hit đã xếp hạng, tốt nhất
    đứng đầu) thì chỗ cắm không phải sửa.

    THỨ TỰ HAI DÒNG DƯỚI LÀ CỐ Ý — ĐỪNG GỘP LẠI CHO GỌN.

    Trên Windows, nạp faiss vào tiến trình TRƯỚC torch thì tiến trình chết
    ngay với 0xC0000005 (lỗi truy cập bộ nhớ). Không traceback, không thông
    báo, màn hình chỉ dừng giữa chừng rồi về dấu nhắc — nhìn y như chạy xong.
    Hai thư viện mỗi bên mang một bản OpenMP riêng, bản nạp sau giẫm lên bản
    nạp trước.

    Nên phải mã hoá câu chữ (kéo theo torch) XONG rồi mới import faiss.

    Đây là chỗ đầu tiên trong cả dự án nạp hai thư viện đó cùng một tiến
    trình: Việc 2 chỉ dùng faiss, Việc 3 chỉ dùng torch. Đo trên máy Ngân —
    Python 3.11, torch 2.6.0+cu124, faiss-cpu 1.8.0.
    """
    vector = ma_hoa_cau(cau_hoi)                       # torch vào trước

    from ..index.faiss_index import search_by_vector   # faiss vào sau

    return search_by_vector(vector, top_k=so_ung_vien)


# ---------------------------------------------------------------------------
# Kết quả một truy vấn
# ---------------------------------------------------------------------------


@dataclass
class KetQuaTruyVan:
    """Toàn bộ số liệu của một lần chạy — để in ra và để ghi vào runs/."""

    query_id: str
    task: str
    cau_hoi: str

    so_ung_vien: int                      # trước lọc trùng
    ket_qua: list                         # sau lọc trùng, đã cắt
    bao_cao_loc_trung: BaoCaoLocTrung
    vi_pham_cong_5: list                  # rỗng nghĩa là ĐẠT
    so_dong_nop: int
    bao_cao_nop: str
    thoi_gian_ms: dict[str, float]
    tham_so: dict[str, Any]
    canh_bao: list[str] = field(default_factory=list)
    duong_dan_nop: Path | None = None

    @property
    def dat_cong_5(self) -> bool:
        return not self.vi_pham_cong_5

    def tom_tat(self) -> str:
        tong = self.thoi_gian_ms.get("tong", 0.0)
        dong = " | ".join(
            [
                f"[{self.query_id}/{self.task}] {self.cau_hoi[:48]}",
                f"{self.so_ung_vien} ứng viên → {self.bao_cao_loc_trung.ra} sau lọc"
                f" → {self.so_dong_nop} dòng nộp",
                f"{tong:.0f} ms",
                "cổng 5: ĐẠT" if self.dat_cong_5 else
                f"cổng 5: TRƯỢT ({len(self.vi_pham_cong_5)} cặp quá gần)",
            ]
        )
        return dong

    def bao_cao_day_du(self) -> str:
        dong = [
            f"Truy vấn : {self.query_id}  ({self.task})",
            f"Câu chữ  : {self.cau_hoi}",
            "",
            f"  bước 1+2  tra kho      : {self.so_ung_vien} ứng viên"
            f"  ({self.thoi_gian_ms.get('tra_kho', 0):.0f} ms)",
            f"  bước 3    lọc trùng    : {self.bao_cao_loc_trung}"
            f"  ({self.thoi_gian_ms.get('loc_trung', 0):.0f} ms)",
            f"  bước 4    cắt {self.tham_so.get('so_dong_toi_da', 100)} dòng   :"
            f" {self.bao_cao_nop}",
            f"  bước 5    ghi tệp      : {self.duong_dan_nop or '(không ghi)'}",
            "",
            f"  cổng thoát 5           : "
            + ("ĐẠT — không có hai dòng cùng video cách nhau dưới "
               f"{self.tham_so.get('cua_so_giay', 10):.1f} giây"
               if self.dat_cong_5
               else f"TRƯỢT — {len(self.vi_pham_cong_5)} cặp quá gần"),
            f"  tổng thời gian         : {self.thoi_gian_ms.get('tong', 0):.0f} ms",
        ]

        for i, j, video_id, khoang_cach in self.vi_pham_cong_5[:5]:
            dong.append(
                f"      dòng {i} và {j} cùng {video_id}, cách {khoang_cach:.2f} giây"
            )

        return "\n".join(dong)


# ---------------------------------------------------------------------------
# Bước 4 — biến Hit thành dòng nộp
# ---------------------------------------------------------------------------


def _gom_kis_qa(
    hits: Sequence["Hit"],
    dap_an: str | None,
    ngan_sach: SubmissionBudget,
) -> list:
    """
    Mỗi ảnh một dòng. frame_ids LẤY TỪ hit.frame_idx, KHÔNG phải hit.n.

    Trả về danh sách Hit THẬT SỰ được nhận vào tệp nộp. Không dựng danh sách
    rồi cắt bằng [:100]: SubmissionBudget còn tự bỏ dòng trùng, nên hai danh
    sách sẽ lệch nhau ở đúng chỗ đó và phép soát cổng 5 bên dưới sẽ đi kiểm
    một dòng không hề nằm trong tệp.
    """
    nhan: list = []

    for h in hits:
        if ngan_sach.is_full():
            break

        duoc_nhan = ngan_sach.add(
            Answer(
                video_id=h.video_id,
                frame_ids=[h.frame_idx],
                answer=dap_an,
            )
        )

        if duoc_nhan:
            nhan.append(h)

    return nhan


def _gom_trake(hits: Sequence["Hit"], so_moc: int) -> tuple[list[Answer], list[str]]:
    """
    Gom ảnh thành dòng TRAKE: mỗi video một dòng, gồm `so_moc` mốc thời gian
    xếp theo thứ tự thời gian tăng dần.

    Thứ tự VIDEO giữ theo thứ hạng: video có ảnh điểm cao nhất đứng trước.
    Thứ tự MỐC trong một dòng thì theo thời gian, vì chuỗi sự kiện phải đúng
    trình tự chứ không phải đúng điểm.

    CHỖ CẦN NGUYÊN KIỂM (Việc 12): danh sách vào đây là danh sách ĐÃ LỌC TRÙNG,
    nên hai mốc liên tiếp của cùng một video luôn cách nhau ít nhất
    `cua_so_giay`. Chuỗi sự kiện diễn ra nhanh hơn 10 giây sẽ bị mất mốc giữa.
    Chưa đổi ở đây vì luật lọc trùng là luật chung; chờ con số từ bộ câu hỏi
    tự chấm rồi Ngân với Nguyên chốt lại cửa sổ.
    """
    theo_video: dict[str, list] = {}
    thu_tu_video: list[str] = []

    for h in hits:
        if h.video_id not in theo_video:
            theo_video[h.video_id] = []
            thu_tu_video.append(h.video_id)
        theo_video[h.video_id].append(h)

    answers: list[Answer] = []
    thieu_moc = 0

    for video_id in thu_tu_video:
        nhom = theo_video[video_id]

        if len(nhom) < so_moc:
            thieu_moc += 1
            continue

        chon = sorted(nhom[:so_moc], key=lambda h: h.pts_time)
        answers.append(
            Answer(
                video_id=video_id,
                frame_ids=[h.frame_idx for h in chon],
            )
        )

    canh_bao: list[str] = []

    if thieu_moc:
        canh_bao.append(
            f"TRAKE: bỏ {thieu_moc} video vì không đủ {so_moc} mốc cách nhau "
            f"{cua_so_giay():.1f} giây. Còn {len(answers)} dòng."
        )

    return answers, canh_bao


# ---------------------------------------------------------------------------
# Mạch đầu–cuối
# ---------------------------------------------------------------------------


def run_query(
    cau_hoi: str,
    query_id: str = "demo",
    task: str = KIS,
    dap_an: str | None = None,
    so_moc_trake: int = SO_MOC_TRAKE_MAC_DINH,
    ghi_tep: bool = True,
    thu_muc_nop: Path | None = None,
    tim_ung_vien: Callable[[str, int], list] | None = None,
) -> KetQuaTruyVan:
    """
    Chạy trọn năm bước cho MỘT câu truy vấn.

    Tham số:
        cau_hoi       : câu chữ người dùng gõ.
        query_id      : dùng để đặt tên tệp nộp (query-<id>-<task>.csv).
        task          : KIS / QA / TRAKE.
        dap_an        : bắt buộc khi task=QA, bỏ qua ở hai dạng kia.
        ghi_tep       : False thì chạy hết mạch nhưng không đụng vào đĩa —
                        dùng khi chấm điểm hàng loạt.
        thu_muc_nop   : mặc định <gốc dữ liệu>/submissions/.
        tim_ung_vien  : chỗ cắm cho Việc 10 của Nghi. None nghĩa là dùng CLIP.

    Trả về:
        KetQuaTruyVan — có sẵn con số của từng bước, KHÔNG in ra màn hình.
        Việc in là của scripts/run_search.py.
    """
    if task not in (KIS, QA, TRAKE):
        raise ValueError(f"Dạng truy vấn lạ: {task}")

    if task == QA and not dap_an:
        raise ValueError(
            "Dạng Q&A bắt buộc phải có đáp án. Mạch tìm kiếm chỉ tìm ra ẢNH; "
            "câu trả lời do người ngồi máy đọc ảnh rồi gõ vào, hoặc do phần "
            "hỏi–đáp của Giai đoạn 2 sinh ra."
        )

    tham_so = tham_so_da_dung()
    tham_so["so_moc_trake"] = so_moc_trake
    canh_bao = list(canh_bao_cau_hinh())

    thoi_gian: dict[str, float] = {}
    bat_dau = time.perf_counter()

    # --- bước 1 + 2: câu chữ → dãy số → tra kho ----------------------------
    nguon = tim_ung_vien or tim_ung_vien_clip
    moc = time.perf_counter()
    ung_vien = list(nguon(cau_hoi, so_ung_vien_moi_nguon()))
    thoi_gian["tra_kho"] = (time.perf_counter() - moc) * 1000

    # --- bước 3: lọc trùng --------------------------------------------------
    moc = time.perf_counter()
    da_loc, bao_cao_loc = loc_trung(
        ung_vien,
        cua_so_giay=cua_so_giay(),
        so_anh_toi_da_moi_video=so_anh_toi_da_moi_video(),
    )
    thoi_gian["loc_trung"] = (time.perf_counter() - moc) * 1000

    # --- bước 4: cắt 100 dòng ----------------------------------------------
    gioi_han = so_dong_toi_da()
    ngan_sach = SubmissionBudget(task=task, limit=gioi_han)

    # `ket_qua` là danh sách Hit tương ứng ĐÚNG những dòng nằm trong tệp nộp.
    # Phép soát cổng 5 bên dưới kiểm trên danh sách này, nên nó phải khớp tệp
    # chứ không phải khớp danh sách trước khi cắt.
    if task == TRAKE:
        answers, canh_bao_trake = _gom_trake(da_loc, so_moc_trake)
        canh_bao.extend(canh_bao_trake)
        ngan_sach.extend(answers)
        video_da_nop = {a.video_id for a in ngan_sach.answers}
        ket_qua = [h for h in da_loc if h.video_id in video_da_nop]
    else:
        ket_qua = _gom_kis_qa(
            da_loc,
            dap_an if task == QA else None,
            ngan_sach,
        )

    if len(ngan_sach) < gioi_han:
        canh_bao.append(
            f"Chỉ ra {len(ngan_sach)}/{gioi_han} dòng. Bỏ phí "
            f"{gioi_han - len(ngan_sach)} suất — tăng "
            f"tra_cuu.so_ung_vien_moi_nguon trong settings.yaml "
            f"(đang là {so_ung_vien_moi_nguon()})."
        )

    # --- soát cổng thoát số 5 ----------------------------------------------
    # Cố ý gọi hàm kiểm viết bằng cách khác (so mọi cặp) chứ không tin bước 3.
    vi_pham = tim_cap_qua_gan(ket_qua, cua_so_giay=cua_so_giay())

    # --- bước 5: ghi tệp ----------------------------------------------------
    duong_dan = None

    if ghi_tep:
        thu_muc = thu_muc_nop or SUBMISSIONS_DIR
        duong_dan = ngan_sach.write(
            Path(thu_muc) / submission_filename(query_id, task)
        )

    thoi_gian["tong"] = (time.perf_counter() - bat_dau) * 1000

    return KetQuaTruyVan(
        query_id=query_id,
        task=task,
        cau_hoi=cau_hoi,
        so_ung_vien=len(ung_vien),
        ket_qua=ket_qua,
        bao_cao_loc_trung=bao_cao_loc,
        vi_pham_cong_5=vi_pham,
        so_dong_nop=len(ngan_sach),
        bao_cao_nop=ngan_sach.report(),
        thoi_gian_ms=thoi_gian,
        tham_so=tham_so,
        canh_bao=canh_bao,
        duong_dan_nop=duong_dan,
    )


def run_queries(
    cau_hoi_list: Sequence[dict],
    ghi_tep: bool = True,
    thu_muc_nop: Path | None = None,
    tim_ung_vien: Callable[[str, int], list] | None = None,
) -> list[KetQuaTruyVan]:
    """
    Chạy một danh sách truy vấn. Mỗi phần tử là một dict:

        {"query_id": "dev-001", "task": "kis", "text": "..."}

    Khoá nhận được: query_id, task, text (hoặc text_en), gt_answer, so_moc_trake.
    Đúng khoá của docs/schema/example_dev_queries.jsonl để chạy thẳng bộ câu
    hỏi tự chấm của Nguyên mà không phải đổi định dạng.

    Câu nào lỗi thì ghi lại lỗi rồi chạy tiếp, KHÔNG dừng cả mẻ — chạy 32 câu
    mà chết ở câu thứ ba là mất luôn 29 câu sau.
    """
    ket_qua: list[KetQuaTruyVan] = []

    for i, muc in enumerate(cau_hoi_list, start=1):
        cau_hoi = muc.get("text_en") or muc.get("text") or ""
        query_id = str(muc.get("query_id") or f"q{i}")
        task = str(muc.get("task") or KIS).lower()

        ket_qua.append(
            run_query(
                cau_hoi=cau_hoi,
                query_id=query_id,
                task=task,
                dap_an=muc.get("gt_answer"),
                so_moc_trake=int(muc.get("so_moc_trake") or SO_MOC_TRAKE_MAC_DINH),
                ghi_tep=ghi_tep,
                thu_muc_nop=thu_muc_nop,
                tim_ung_vien=tim_ung_vien,
            )
        )

    return ket_qua


def clear_cache() -> None:
    """Thả ClipEncoder. Gọi khi đổi bản mô hình trong cùng một tiến trình."""
    global _ENCODER
    _ENCODER = None