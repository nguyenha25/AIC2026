from __future__ import annotations

from aic2026.qa_answer import HitQALanCan, tra_loi_theo_hang
from scripts.adaptive_k import AdaptiveKResult
from scripts.answer_consensus import choose_consensus

def run_qa_v4(
    cau_hoi: str,
    result: AdaptiveKResult,
    *,
    bo_doc_anh=None,
    dung_vlm: bool = True,
):
    reader_outputs = run_reader_from_r4(
        cau_hoi=cau_hoi,
        result=result,
        bo_doc_anh=bo_doc_anh,
        dung_vlm=dung_vlm,
    )

    dap_an_candidates = [
        dap_an
        for _, dap_an in reader_outputs
    ]

    consensus = choose_consensus(
        question=cau_hoi,
        candidates=dap_an_candidates,
    )

    return consensus

def run_reader_from_r4(
    cau_hoi: str,
    result: AdaptiveKResult,
    *,
    bo_doc_anh=None,
    dung_vlm: bool = True,
):
    hits = r4_to_reader_hits(result)
    validate_r4_reader_contract(result, hits)

    return tra_loi_theo_hang(
        cau_hoi=cau_hoi,
        hits=hits,
        so_dong=len(hits),
        so_hang_vlm=len(hits),
        bo_doc_anh=bo_doc_anh,
        dung_vlm=dung_vlm,
        mo_rong_lan_can=False,
    )

def r4_to_reader_hits(
    result: AdaptiveKResult,
) -> list[HitQALanCan]:
    """Chuyển output QA-R4 thành input Reader, giữ nguyên thứ tự và budget."""
    hits: list[HitQALanCan] = []

    for candidate in result.selected_candidates:
        frame_idx = int(
            candidate.get(
                "frame_idx",
                candidate["frame_id"],
            )
        )

        hits.append(
            HitQALanCan(
                video_id=str(candidate["video_id"]),
                n=int(candidate["n"]),
                score=float(candidate["score"]),
                frame_idx=frame_idx,
                pts_time=float(candidate.get("pts_time", 0.0)),
                source="qa_r4",
            )
        )

    return hits


def validate_r4_reader_contract(
    result: AdaptiveKResult,
    hits: list[HitQALanCan],
) -> None:
    """Bảo đảm adapter không làm thay đổi contract của QA-R4."""
    if len(hits) != result.k_effective:
        raise ValueError(
            "Số hit Reader không bằng k_effective của QA-R4: "
            f"{len(hits)} != {result.k_effective}"
        )

    if len(hits) > result.reader_k_requested:
        raise ValueError(
            "Reader nhận nhiều frame hơn budget QA-R4: "
            f"{len(hits)} > {result.reader_k_requested}"
        )

    if len(hits) != len(result.selected_candidates):
        raise ValueError(
            "Số hit Reader không bằng số selected_candidates."
        )

    for candidate, hit in zip(
        result.selected_candidates,
        hits,
        strict=True,
    ):
        if str(candidate["video_id"]) != hit.video_id:
            raise ValueError("Adapter làm thay đổi thứ tự video_id.")

        if int(candidate["n"]) != hit.n:
            raise ValueError("Adapter làm thay đổi n.")

        expected_frame_idx = int(
            candidate.get(
                "frame_idx",
                candidate["frame_id"],
            )
        )

        if expected_frame_idx != hit.frame_idx:
            raise ValueError("Adapter làm thay đổi frame_idx/frame_id.")
