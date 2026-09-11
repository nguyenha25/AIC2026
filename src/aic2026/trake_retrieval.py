"""
TR-R1 — COARSE TEMPORAL RETRIEVAL BẰNG CLIP-L
==============================================

Mục đích
--------
Từ một chuỗi sự kiện:

    event tiếng Việt
        ↓
    Marian query expansion (chỉ cho CLIP)
        ↓
    CLIP-L text encoder
        ↓
    FAISS CLIP-L Top-K
        ↓
    frame candidate fusion
        ↓
    temporal grouping
        ↓
    coarse temporal regions
        ↓
    R2

TR-R1 KHÔNG làm:
    - dense/local retrieval
    - temporal alignment / DP
    - OCR / ASR
    - chọn best frame cuối cùng
    - submission

Public contract
---------------
tim_vung_tho(event, config=..., retriever=...)
    -> TRR1Result

tim_nhieu_su_kien(events, config=..., retriever=...)
    -> tuple[TRR1Result, ...]

CoarseRegion chỉ có:
    video_id
    start_time
    end_time
    score
    hits

Không có best_frame_idx.

Thiết kế scoring
----------------
TR-R1 tối ưu coarse temporal recall, vì vậy region không được
xếp hạng chỉ bằng peak frame score.

Region score gồm:

    peak_score
        : hit mạnh nhất trong region.

    support_score
        : mức hỗ trợ của các hit còn lại.

    density_score
        : mức tập trung temporal của các hit.

Các thành phần được normalize về [0, 1] trước khi fusion.

Mặc định:

    peak_weight    = 0.50
    support_weight = 0.30
    density_weight = 0.20

Đây là ranking heuristic có thể benchmark/tune ở tầng benchmark.
Không dùng số frame hit để cộng cosine thô trực tiếp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable


# ============================================================================
# CONFIG
# ============================================================================


@dataclass(frozen=True)
class TRR1Config:
    """
    Cấu hình TR-R1.

    Retrieval
    ---------
    top_k:
        Top-K frame cho MỖI query variant.

    Temporal grouping
    -----------------
    max_region_duration_seconds:
        Độ dài tối đa của một coarse region.

    region_merge_gap_seconds:
        Khoảng cách tối đa giữa hai hit liên tiếp để tiếp tục
        cùng một region.

    region_padding_seconds:
        Padding mỗi phía của region sau khi grouping.

        Padding chỉ dùng để biểu diễn temporal interval.
        Không tạo thêm hit.

    min_region_duration_seconds:
        Độ dài tối thiểu của region. Một hit đơn lẻ không được biểu diễn
        bằng interval suy biến ``[t, t]`` vì temporal IoU của interval đó
        luôn bằng 0, kể cả khi hit nằm đúng trong GT.

    Ranking
    -------
    max_regions_per_event:
        Số coarse regions trả ra.

    min_hits_per_region:
        Số hit tối thiểu để region được coi là candidate.
        Giá trị 1 giữ recall cao nhất.

    peak_weight:
        Trọng số peak hit.

    support_weight:
        Trọng số support của các hit còn lại.

    density_weight:
        Trọng số temporal density.

    Query expansion
    ----------------
    use_query_expansion:
        Có dùng Marian hay không.

    max_query_variants:
        Số query variant tối đa.

    video_consensus_weight:
        Trọng số prior video dùng khi tìm nhiều event của cùng một query
        TRAKE. Prior chỉ được tính từ retrieval hits của toàn chuỗi, không
        dùng GT. Đặt 0 để giữ hành vi xếp hạng từng event độc lập.

    video_rrf_k:
        Hằng số RRF khi hợp nhất best rank của mỗi video giữa các event.

    video_sequence_weight:
        Trọng số trộn prior consensus hiện tại với prior coverage có thứ tự
        thời gian trên raw hits. Mặc định 0 để giữ nguyên production baseline.

    video_sequence_span_weight:
        Trọng số compactness của path có ordered coverage lớn nhất. Mặc định
        0 để sequence prior gốc và production baseline không đổi.

    video_sequence_span_scale_seconds:
        Thang thời gian của compactness ``1 / (1 + span / scale)``.

    consensus_rescue_videos:
        Số video có query-level consensus cao nhất được giữ ít nhất một region
        nếu video đó có raw hit trong event. Mặc định 0 để giữ nguyên baseline.
    """

    top_k: int = 500

    max_region_duration_seconds: float = 3.0
    region_merge_gap_seconds: float = 0.5
    region_padding_seconds: float = 0.1
    min_region_duration_seconds: float = 0.16

    max_regions_per_event: int = 10
    min_hits_per_region: int = 1

    peak_weight: float = 0.50
    support_weight: float = 0.30
    density_weight: float = 0.20

    use_query_expansion: bool = True
    max_query_variants: int = 4

    video_consensus_weight: float = 0.45
    video_rrf_k: float = 60.0
    video_sequence_weight: float = 0.0
    video_sequence_span_weight: float = 0.0
    video_sequence_span_scale_seconds: float = 60.0
    consensus_rescue_videos: int = 0

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(
                "top_k phải > 0."
            )

        if self.max_region_duration_seconds <= 0:
            raise ValueError(
                "max_region_duration_seconds phải > 0."
            )

        if self.region_merge_gap_seconds < 0:
            raise ValueError(
                "region_merge_gap_seconds phải >= 0."
            )

        if self.region_padding_seconds < 0:
            raise ValueError(
                "region_padding_seconds phải >= 0."
            )

        if self.min_region_duration_seconds <= 0:
            raise ValueError(
                "min_region_duration_seconds phải > 0."
            )

        if (
            self.min_region_duration_seconds
            > self.max_region_duration_seconds
        ):
            raise ValueError(
                "min_region_duration_seconds không được lớn hơn "
                "max_region_duration_seconds."
            )

        if self.max_regions_per_event <= 0:
            raise ValueError(
                "max_regions_per_event phải > 0."
            )

        if self.min_hits_per_region <= 0:
            raise ValueError(
                "min_hits_per_region phải > 0."
            )

        weights = (
            self.peak_weight,
            self.support_weight,
            self.density_weight,
        )

        if any(
            weight < 0
            for weight in weights
        ):
            raise ValueError(
                "Các region score weights phải >= 0."
            )

        if sum(weights) <= 0:
            raise ValueError(
                "Tổng region score weights phải > 0."
            )

        if self.max_query_variants <= 0:
            raise ValueError(
                "max_query_variants phải > 0."
            )

        if not 0.0 <= self.video_consensus_weight <= 1.0:
            raise ValueError(
                "video_consensus_weight phải nằm trong [0, 1]."
            )

        if self.video_rrf_k < 0:
            raise ValueError(
                "video_rrf_k phải >= 0."
            )

        if not 0.0 <= self.video_sequence_weight <= 1.0:
            raise ValueError(
                "video_sequence_weight phải nằm trong [0, 1]."
            )

        if not 0.0 <= self.video_sequence_span_weight <= 1.0:
            raise ValueError(
                "video_sequence_span_weight phải nằm trong [0, 1]."
            )

        if self.video_sequence_span_scale_seconds <= 0:
            raise ValueError(
                "video_sequence_span_scale_seconds phải > 0."
            )

        if self.consensus_rescue_videos < 0:
            raise ValueError(
                "consensus_rescue_videos phải >= 0."
            )


# ============================================================================
# PUBLIC RESULT OBJECTS
# ============================================================================


@dataclass(frozen=True)
class CoarseRegion:
    """
    Một coarse temporal region.

    CỐ Ý không có:
        - frame_idx
        - best_frame_idx
        - best_frame

    Frame provenance chỉ nằm trong `hits`.
    """

    video_id: str
    start_time: float
    end_time: float
    score: float
    hits: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TRR1Result:
    """
    Kết quả TR-R1 cho một event.
    """

    event_id: str
    text: str
    relation: str | None
    regions: tuple[CoarseRegion, ...]


# ============================================================================
# TEST / INTERNAL HIT BUILDER
# ============================================================================


def _build_test_hit(
    video_id: str,
    pts_time: float,
    score: float,
) -> dict[str, Any]:
    """
    Tạo một hit tối thiểu cho unit test.

    Production hit có thể chứa thêm:
        n
        frame_idx
        rank
        source
    """

    return {
        "video_id": str(video_id),
        "pts_time": float(pts_time),
        "score": float(score),
    }


# ============================================================================
# QUERY EVENT VALIDATION
# ============================================================================


def _parse_event(
    event: Any,
) -> tuple[str, str, str | None]:
    """
    Parse một event theo contract TR-R1.

    Bắt buộc:
        event_id
        text

    Không chấp nhận:
        description thay cho text.
    """

    if not isinstance(event, dict):
        raise ValueError(
            "TR-R1 event phải là dict."
        )

    if "event_id" not in event:
        raise ValueError(
            "TR-R1 event thiếu trường 'event_id'."
        )

    if "text" not in event:
        raise ValueError(
            "TR-R1 event bắt buộc có trường 'text'."
        )

    event_id = event["event_id"]
    text = event["text"]

    if event_id is None:
        raise ValueError(
            "event_id không được là None."
        )

    if not isinstance(text, str):
        raise ValueError(
            "event.text phải là string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "event.text không được rỗng."
        )

    relation = event.get("relation")

    if relation is not None:
        relation = str(relation).strip()

    return (
        str(event_id),
        text,
        relation,
    )


# ============================================================================
# QUERY EXPANSION
# ============================================================================


_TRR1_VISUAL_ACTION_TERMS: tuple[tuple[str, str], ...] = (
    ("hai con rồng vàng", "two golden dragons"),
    ("người biểu diễn lân", "lion dance performers"),
    ("bốn chân hoàn toàn chạm đất", "landing on all four feet"),
    ("4 chân hoàn toàn chạm đất", "landing on all four feet"),
    ("chào ban giám khảo", "bowing to judges"),
    ("cây sả", "lemongrass stalk"),
    ("cây xả", "lemongrass stalk"),
    ("kẻng đồng", "bronze gong"),
    ("thanh trụ", "acrobatic poles"),
    ("cột trụ", "acrobatic poles"),
    ("cử động đầu", "moving its head"),
    ("múa lân", "Chinese lion dance"),
    ("con lân", "Chinese lion dance"),
    ("rồng vàng", "golden dragon"),
    ("con rồng", "dragon"),
    ("cắt rời", "cutting apart"),
    ("chạm vào", "touching"),
    ("xoay vòng", "spinning around"),
    ("quay vòng", "spinning around"),
    ("xoay người", "spinning"),
    ("tiếp đất", "landing"),
    ("dùi", "mallet"),
    ("dao", "knife"),
    ("chào", "bowing"),
    ("lân", "Chinese lion"),
)


def _trr1_visual_action_variants(
    text: str,
    *,
    max_variants: int = 1,
) -> list[str]:
    """Tạo cụm CLIP ngắn từ hành động/vật thể nhìn thấy được.

    Bảng này chỉ chứa các khái niệm hình ảnh có nghĩa tổng quát. Thuật toán
    chọn match dài nhất khi các cụm chồng lấn, giữ thứ tự xuất hiện trong câu
    và không dùng query id, video id hay GT.
    """

    if max_variants <= 0:
        return []

    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return []

    matches: list[tuple[int, int, str]] = []
    for vietnamese, english in _TRR1_VISUAL_ACTION_TERMS:
        start = normalized.find(vietnamese)
        while start >= 0:
            end = start + len(vietnamese)
            left_ok = start == 0 or not normalized[start - 1].isalnum()
            right_ok = end == len(normalized) or not normalized[end].isalnum()
            if left_ok and right_ok:
                matches.append((start, end, english))
            start = normalized.find(vietnamese, start + 1)

    selected: list[tuple[int, int, str]] = []
    for candidate in sorted(matches, key=lambda row: (row[0], -(row[1] - row[0]))):
        start, end, _english = candidate
        if any(start < chosen_end and end > chosen_start for chosen_start, chosen_end, _ in selected):
            continue
        selected.append(candidate)

    concepts: list[str] = []
    for _start, _end, english in sorted(selected):
        if english not in concepts:
            concepts.append(english)

    if len(concepts) < 2:
        return []

    phrase = " ".join(concepts)
    variants = [phrase, f"a video frame of {phrase}"]
    return variants[:max_variants]


def _mo_rong_trr1(
    text: str,
    *,
    use_query_expansion: bool = True,
    max_query_variants: int = 4,
) -> list[str]:

    text = (text or "").strip()

    if not text:
        return []

    if not use_query_expansion:
        return [text]

    try:
        from aic2026.query_expand import mo_rong

        output: list[str] = []

        for nguon in (
            "marian",
            "tu_dien",
        ):
            try:
                result = mo_rong(
                    text,
                    nguon=nguon,
                )
            except Exception:
                continue

            variants = getattr(
                result,
                "cum_tieng_anh",
                None,
            )

            if not variants:
                continue

            for variant in variants:
                variant = str(
                    variant or ""
                ).strip()

                if (
                    variant
                    and variant not in output
                ):
                    output.append(
                        variant
                    )

                if (
                    len(output)
                    >= max_query_variants
                ):
                    return output

        return output or [text]

    except Exception:
        return [text]


def _query_variants(
    text: str,
    *,
    use_query_expansion: bool,
    max_query_variants: int,
) -> list[str]:
    """
    Chuẩn hóa danh sách query variants.
    """

    text = (text or "").strip()

    if not text:
        return []

    variants = _mo_rong_trr1(
        text,
        use_query_expansion=use_query_expansion,
        max_query_variants=max_query_variants,
    )

    output: list[str] = []

    for variant in variants:
        variant = str(
            variant or ""
        ).strip()

        if not variant:
            continue

        if variant not in output:
            output.append(variant)

        if len(output) >= max_query_variants:
            break

    return output or [text]


# ============================================================================
# CLIP-L RUNTIME
# ============================================================================


_CLIP_L_RUNTIME: (
    tuple[Any, Any, Any, Any, Any] | None
) = None


def _get_trr1_clip_l_runtime():
    """
    Load/cache:

        model
        tokenizer
        device
        FAISS index
        ID dataframe
    """

    global _CLIP_L_RUNTIME

    if _CLIP_L_RUNTIME is not None:
        return _CLIP_L_RUNTIME

    from aic2026.index import clip_l_index

    model, tokenizer, device = (
        clip_l_index._encoder()
    )

    index, ids = clip_l_index.nap_chi_muc()

    _CLIP_L_RUNTIME = (
        model,
        tokenizer,
        device,
        index,
        ids,
    )

    return _CLIP_L_RUNTIME


# ============================================================================
# RAW CLIP-L RETRIEVAL
# ============================================================================


def _tim_hit_clip_l_mot_query(
    text: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Một query text -> raw CLIP-L Top-K frame hits.
    """

    text = (text or "").strip()

    if not text:
        return []

    if top_k <= 0:
        return []

    import numpy as np
    import torch

    model, tokenizer, device, index, ids = (
        _get_trr1_clip_l_runtime()
    )

    tokens = tokenizer([text])

    if hasattr(tokens, "to"):
        tokens = tokens.to(device)

    elif isinstance(tokens, dict):
        tokens = {
            key: value.to(device)
            if hasattr(value, "to")
            else value
            for key, value in tokens.items()
        }

    with torch.inference_mode():
        query = model.encode_text(tokens)

    query = (
        query
        / query.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
    )

    query_np = (
        query.detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    k = min(
        int(top_k),
        int(index.ntotal),
    )

    if k <= 0:
        return []

    scores, positions = index.search(
        query_np,
        k,
    )

    scores = scores[0]
    positions = positions[0]

    hits: list[dict[str, Any]] = []

    for rank, (
        score,
        position,
    ) in enumerate(
        zip(scores, positions),
        start=1,
    ):
        position = int(position)

        if position < 0:
            continue

        try:
            row = ids.iloc[position]
        except (
            IndexError,
            KeyError,
        ):
            continue

        try:
            video_id = str(
                row["video_id"]
            )

            n = int(
                row["n"]
            )

            frame_idx = int(
                row["frame_idx"]
            )

            pts_time = float(
                row["pts_time"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        hits.append(
            {
                "video_id": video_id,
                "n": n,
                "frame_idx": frame_idx,
                "pts_time": pts_time,
                "score": float(score),
                "rank": int(rank),
                "source": "clip_l",
            }
        )

    return hits


# ============================================================================
# HIT HELPERS
# ============================================================================


def _safe_score(
    hit: dict[str, Any],
) -> float:
    """
    Lấy score an toàn.
    """

    try:
        return float(hit["score"])
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return float("-inf")


def _safe_time(
    hit: dict[str, Any],
) -> float | None:
    """
    Lấy timestamp an toàn.
    """

    try:
        value = float(
            hit["pts_time"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None

    if value < 0:
        return None

    return value


def _safe_video_id(
    hit: dict[str, Any],
) -> str:
    """
    Lấy video_id an toàn.
    """

    try:
        return str(
            hit["video_id"]
        ).strip()
    except Exception:
        return ""


# ============================================================================
# MULTI-VARIANT MERGE
# ============================================================================


def _hit_key(
    hit: dict[str, Any],
) -> tuple[Any, ...]:
    """
    Key ổn định để dedup cùng một frame.

    Ưu tiên:
        video_id + frame_idx
        video_id + n
        video_id + rounded pts_time
    """

    video_id = _safe_video_id(hit)

    if hit.get("frame_idx") is not None:
        try:
            return (
                video_id,
                "frame_idx",
                int(hit["frame_idx"]),
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    if hit.get("n") is not None:
        try:
            return (
                video_id,
                "n",
                int(hit["n"]),
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    timestamp = _safe_time(hit)

    if timestamp is None:
        timestamp = 0.0

    return (
        video_id,
        "pts_time",
        round(
            timestamp,
            3,
        ),
    )


def _merge_query_variant_hits(
    hits_by_variant: Iterable[
        list[dict[str, Any]]
    ],
    *,
    rrf_k: float = 60.0,
) -> list[dict[str, Any]]:

    fused: dict[
        tuple[Any, ...],
        dict[str, Any],
    ] = {}

    rrf_scores: dict[
        tuple[Any, ...],
        float,
    ] = {}

    variant_ranks: dict[
        tuple[Any, ...],
        list[int],
    ] = {}

    for variant_idx, hits in enumerate(
        hits_by_variant
    ):
        for rank, hit in enumerate(
            hits,
            start=1,
        ):
            if not isinstance(
                hit,
                dict,
            ):
                continue

            key = _hit_key(
                hit
            )

            # -------------------------------------------------
            # RRF: mỗi variant đóng góp bằng rank.
            # -------------------------------------------------

            rrf_scores[key] = (
                rrf_scores.get(
                    key,
                    0.0,
                )
                + 1.0
                / (
                    rrf_k
                    + float(rank)
                )
            )

            variant_ranks.setdefault(
                key,
                [],
            ).append(
                int(rank)
            )

            # -------------------------------------------------
            # Giữ hit gốc có cosine mạnh nhất
            # để bảo toàn provenance.
            # -------------------------------------------------

            previous = fused.get(
                key
            )

            candidate = dict(
                hit
            )

            if (
                previous is None
                or _safe_score(
                    candidate
                )
                > _safe_score(
                    previous
                )
            ):
                fused[key] = candidate

    output: list[
        dict[str, Any]
    ] = []

    for key, hit in fused.items():
        candidate = dict(
            hit
        )

        candidate[
            "clip_score"
        ] = float(
            _safe_score(hit)
        )

        candidate[
            "score"
        ] = float(
            rrf_scores[key]
        )

        candidate[
            "variant_ranks"
        ] = tuple(
            variant_ranks[key]
        )

        candidate[
            "source"
        ] = "clip_l_rrf"

        output.append(
            candidate
        )

    output.sort(
        key=lambda hit: (
            -float(
                hit["score"]
            ),
            _safe_video_id(
                hit
            ),
            _safe_time(hit)
            if _safe_time(hit)
            is not None
            else float("inf"),
        )
    )

    return output


def tim_hit_clip_l(
    text: str,
    top_k: int = 500,
    *,
    use_query_expansion: bool = True,
    max_query_variants: int = 2,
) -> list[dict[str, Any]]:
    """
    Query tiếng Việt -> raw CLIP-L hits.

    top_k là Top-K cho MỖI variant.

    Sau đó:
        variant hits
            ↓
        frame dedup
            ↓
        best cosine giữ lại
    """

    text = (text or "").strip()

    if not text:
        return []

    if top_k <= 0:
        return []

    variants = _query_variants(
        text,
        use_query_expansion=use_query_expansion,
        max_query_variants=max_query_variants,
    )

    if not variants:
        return []

    hits_by_variant: list[
        list[dict[str, Any]]
    ] = []

    for variant in variants:
        hits_by_variant.append(
            _tim_hit_clip_l_mot_query(
                variant,
                top_k,
            )
        )

    return _merge_query_variant_hits(
        hits_by_variant
    )


def _video_consensus_scores(
    hits_by_event: Iterable[list[dict[str, Any]]],
    *,
    rrf_k: float = 60.0,
) -> dict[str, float]:
    """Tính prior video từ toàn bộ event trong một query TRAKE.

    Mỗi video chỉ đóng góp best rank một lần cho mỗi event. Cách này tránh
    bias về video dài (có nhiều frame trong index) và tránh cộng trực tiếp
    cosine của các event/query variant vốn không cùng calibration.

    Điểm cuối kết hợp:
        - 65% RRF best-rank qua các event;
        - 35% tỉ lệ event có ít nhất một hit của video.

    Kết quả được chuẩn hóa về [0, 1] và hoàn toàn không dùng GT.
    """

    event_hits = list(hits_by_event)

    if not event_hits:
        return {}

    rrf_scores: dict[str, float] = {}
    event_support: dict[str, int] = {}

    for hits in event_hits:
        best_rank: dict[str, int] = {}

        for rank, hit in enumerate(hits, start=1):
            if not isinstance(hit, dict):
                continue

            video_id = _safe_video_id(hit)

            if not video_id or video_id in best_rank:
                continue

            best_rank[video_id] = rank

        for video_id, rank in best_rank.items():
            rrf_scores[video_id] = (
                rrf_scores.get(video_id, 0.0)
                + 1.0 / (rrf_k + float(rank))
            )
            event_support[video_id] = (
                event_support.get(video_id, 0) + 1
            )

    if not rrf_scores:
        return {}

    max_rrf = max(rrf_scores.values())
    num_events = len(event_hits)
    output: dict[str, float] = {}

    for video_id, raw_rrf in rrf_scores.items():
        rrf_norm = raw_rrf / max_rrf if max_rrf > 0 else 0.0
        support_norm = event_support[video_id] / num_events
        output[video_id] = float(
            0.65 * rrf_norm + 0.35 * support_norm
        )

    return output


def _video_sequence_scores(
    hits_by_event: Iterable[list[dict[str, Any]]],
    *,
    rrf_k: float = 60.0,
    max_hits_per_video_event: int = 8,
    span_weight: float = 0.0,
    span_scale_seconds: float = 60.0,
) -> dict[str, float]:
    """Chấm prior video theo coverage và thứ tự thời gian của toàn chuỗi.

    Mỗi video giữ tối đa ``max_hits_per_video_event`` hit tốt nhất/event rồi
    dùng DP tìm chuỗi timestamp tăng nghiêm ngặt dài nhất, cho phép bỏ qua
    event không có bằng chứng. Điểm gồm ordered coverage, raw event coverage
    và chất lượng best rank. Không dùng GT, duration hay tên video.

    Hàm này chỉ có tác dụng khi ``TRR1Config.video_sequence_weight > 0``;
    production mặc định bằng 0 để baseline không đổi.
    """

    if rrf_k < 0:
        raise ValueError("rrf_k phải >= 0")
    if max_hits_per_video_event <= 0:
        raise ValueError("max_hits_per_video_event phải > 0")
    if not 0.0 <= span_weight <= 1.0:
        raise ValueError("span_weight phải nằm trong [0, 1]")
    if span_scale_seconds <= 0:
        raise ValueError("span_scale_seconds phải > 0")

    event_hits = list(hits_by_event)
    if not event_hits:
        return {}

    candidates: dict[str, dict[int, list[tuple[float, int]]]] = {}

    for event_index, hits in enumerate(event_hits):
        seen_times: dict[str, set[float]] = {}

        for rank, hit in enumerate(hits, start=1):
            if not isinstance(hit, dict):
                continue

            video_id = _safe_video_id(hit)
            timestamp = _safe_time(hit)
            if not video_id or timestamp is None:
                continue

            video_event_hits = candidates.setdefault(video_id, {}).setdefault(
                event_index,
                [],
            )
            if len(video_event_hits) >= max_hits_per_video_event:
                continue

            event_seen = seen_times.setdefault(video_id, set())
            rounded_time = round(timestamp, 6)
            if rounded_time in event_seen:
                continue

            event_seen.add(rounded_time)
            video_event_hits.append((timestamp, rank))

    num_events = len(event_hits)
    raw_scores: dict[str, float] = {}

    for video_id, by_event in candidates.items():
        # State: event index, timestamp, ordered event count, RRF path quality.
        states: list[tuple[int, float, int, float]] = []
        best_count = 0
        best_path_quality = 0.0

        for event_index in range(num_events):
            current_states: list[tuple[int, float, int, float]] = []

            for timestamp, rank in by_event.get(event_index, []):
                node_quality = 1.0 / (rrf_k + float(rank))
                count = 1
                path_quality = node_quality

                for previous_event, previous_time, previous_count, previous_quality in states:
                    if previous_event >= event_index or previous_time >= timestamp:
                        continue

                    candidate_count = previous_count + 1
                    candidate_quality = previous_quality + node_quality
                    if (candidate_count, candidate_quality) > (count, path_quality):
                        count = candidate_count
                        path_quality = candidate_quality

                current_states.append(
                    (event_index, timestamp, count, path_quality)
                )
                if (count, path_quality) > (best_count, best_path_quality):
                    best_count = count
                    best_path_quality = path_quality

            states.extend(current_states)

        support_count = len(by_event)
        best_ranks = [
            min(rank for _timestamp, rank in event_candidates)
            for event_candidates in by_event.values()
        ]
        rank_quality = sum(
            1.0 / math.log2(float(rank) + 1.0)
            for rank in best_ranks
        ) / max(1, support_count)

        ordered_coverage = best_count / num_events
        support_coverage = support_count / num_events
        base_score = float(
            0.60 * ordered_coverage
            + 0.30 * support_coverage
            + 0.10 * rank_quality
        )

        # Tìm span ngắn nhất trong các path có ordered coverage lớn nhất.
        # Với một điểm bắt đầu cố định, chọn timestamp hợp lệ sớm nhất ở mỗi
        # event sau không thể làm giảm số event còn nối được.
        compact_count = 0
        compact_span = float("inf")
        for start_event, start_candidates in by_event.items():
            for start_time, _start_rank in start_candidates:
                current_time = start_time
                count = 1

                for event_index in range(start_event + 1, num_events):
                    next_times = [
                        timestamp
                        for timestamp, _rank in by_event.get(event_index, [])
                        if timestamp > current_time
                    ]
                    if not next_times:
                        continue

                    current_time = min(next_times)
                    count += 1

                span = max(0.0, current_time - start_time)
                if count > compact_count or (
                    count == compact_count and span < compact_span
                ):
                    compact_count = count
                    compact_span = span

        compactness = 0.0
        if compact_count > 1 and math.isfinite(compact_span):
            compactness = 1.0 / (
                1.0 + compact_span / span_scale_seconds
            )
        compact_score = ordered_coverage * compactness
        raw_scores[video_id] = float(
            (1.0 - span_weight) * base_score
            + span_weight * compact_score
        )

    if not raw_scores:
        return {}

    max_score = max(raw_scores.values())
    if max_score <= 0:
        return {video_id: 0.0 for video_id in raw_scores}

    return {
        video_id: float(score / max_score)
        for video_id, score in raw_scores.items()
    }


# ============================================================================
# TEMPORAL REGION BUILDER
# ============================================================================


@dataclass
class _RegionBuilder:
    video_id: str
    start_time: float
    end_time: float
    hits: list[dict[str, Any]]


def _can_merge(
    region: _RegionBuilder,
    hit: dict[str, Any],
    config: TRR1Config,
) -> bool:
    """
    Kiểm tra hit có thể nhập region hiện tại không.

    Input đã được sort theo:
        video_id
        pts_time

    nên chỉ cần xét:
        cùng video
        gap <= merge gap
        duration mới <= max duration
    """

    video_id = _safe_video_id(hit)

    if video_id != region.video_id:
        return False

    hit_time = _safe_time(hit)

    if hit_time is None:
        return False

    if hit_time < region.start_time:
        return False

    gap = (
        hit_time
        - region.end_time
    )

    if gap > config.region_merge_gap_seconds:
        return False

    new_end = max(
        region.end_time,
        hit_time,
    )

    duration = (
        new_end
        - region.start_time
    )

    if (
        duration
        > config.max_region_duration_seconds
    ):
        return False

    return True


def _append_hit(
    region: _RegionBuilder,
    hit: dict[str, Any],
) -> None:
    """
    Append hit vào region.
    """

    hit_time = _safe_time(hit)

    if hit_time is None:
        return

    region.end_time = max(
        region.end_time,
        hit_time,
    )

    region.hits.append(hit)


# ============================================================================
# REGION SCORE
# ============================================================================


def _normalize_unit_interval(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Normalize value về [0, 1].
    """

    if maximum <= minimum:
        return 1.0

    normalized = (
        value - minimum
    ) / (
        maximum - minimum
    )

    return max(
        0.0,
        min(
            1.0,
            normalized,
        ),
    )


def _region_peak_score(
    hits: list[dict[str, Any]],
) -> float:
    """
    Peak score của region.

    Cosine CLIP-L thường đã nằm trong một khoảng tương đối ổn định,
    nhưng không giả định cứng [-1, 1] ở đây.

    Ta normalize tương đối theo chính candidate pool ở bước finalize.
    """

    scores = [
        _safe_score(hit)
        for hit in hits
        if _safe_score(hit) != float("-inf")
    ]

    if not scores:
        return 0.0

    return max(scores)


def _region_support_score(
    hits: list[dict[str, Any]],
) -> float:
    """
    Support score.

    Không cộng cosine thô.

    Ý tưởng:
        - lấy các hit trong region
        - bỏ peak
        - các hit còn lại đóng góp giảm dần theo rank trong region

    Điều này thưởng cho region có nhiều evidence độc lập,
    nhưng không để region dài tự động thắng chỉ vì có nhiều frame.
    """

    scores = sorted(
        (
            _safe_score(hit)
            for hit in hits
            if _safe_score(hit) != float("-inf")
        ),
        reverse=True,
    )

    if len(scores) <= 1:
        return 0.0

    support = scores[1:]

    # Dùng top-3 support tối đa để tránh region dài có lợi thế vô hạn.
    support = support[:3]

    if not support:
        return 0.0

    # Reciprocal positional weighting.
    weights = [
        1.0,
        0.5,
        0.3333333333333333,
    ]

    weighted_sum = 0.0
    weight_sum = 0.0

    for idx, score in enumerate(support):
        weight = weights[idx]
        weighted_sum += (
            score * weight
        )
        weight_sum += weight

    if weight_sum <= 0:
        return 0.0

    return (
        weighted_sum
        / weight_sum
    )


def _region_density_score(
    region: _RegionBuilder,
) -> float:
    """
    Temporal density.

    Dùng số hit / span nhưng có saturation.

    Mục tiêu:
        2-3 hit tập trung tốt hơn 1 hit đơn độc,
        nhưng 20 hit không được thắng tuyệt đối chỉ vì số lượng.

    Density được chuẩn hóa bằng:
        hit_count / max(1, span_seconds + 1)

    rồi đưa qua saturation.
    """

    hit_count = len(
        region.hits
    )

    if hit_count <= 1:
        return 0.0

    span = max(
        0.0,
        region.end_time
        - region.start_time,
    )

    raw_density = (
        hit_count
        / (1.0 + span)
    )

    # Saturating transform:
    #
    # density=1 -> 0.50
    # density=2 -> 0.67
    # density=3 -> 0.75
    #
    # Không để count tăng tuyến tính vô hạn.
    return (
        raw_density
        / (
            raw_density
            + 1.0
        )
    )


def _score_regions(
    builders: list[_RegionBuilder],
    config: TRR1Config,
    *,
    video_scores: dict[str, float] | None = None,
) -> list[CoarseRegion]:
    """
    Chấm điểm toàn bộ candidate regions.

    Quan trọng:
        peak/support được normalize tương đối trong candidate pool.

    Sau đó:

        local_score =
            peak_weight    * peak
          + support_weight * support
          + density_weight * density

        score =
            (1 - video_consensus_weight) * local_score
          + video_consensus_weight * video_prior

    ``video_prior`` chỉ có khi API nhiều-event truyền vào và được tính
    hoàn toàn từ retrieval hits của chuỗi event.
    """

    if not builders:
        return []

    builders = [
        builder
        for builder in builders
        if len(builder.hits)
        >= config.min_hits_per_region
    ]

    if not builders:
        return []

    raw_peaks = [
        _region_peak_score(
            builder.hits
        )
        for builder in builders
    ]

    raw_supports = [
        _region_support_score(
            builder.hits
        )
        for builder in builders
    ]

    raw_densities = [
        _region_density_score(
            builder
        )
        for builder in builders
    ]

    peak_min = min(
        raw_peaks
    )
    peak_max = max(
        raw_peaks
    )

    support_min = min(
        raw_supports
    )
    support_max = max(
        raw_supports
    )

    density_min = min(
        raw_densities
    )
    density_max = max(
        raw_densities
    )

    weight_sum = (
        config.peak_weight
        + config.support_weight
        + config.density_weight
    )

    scored: list[
        tuple[
            float,
            _RegionBuilder,
        ]
    ] = []

    for (
        builder,
        peak,
        support,
        density,
    ) in zip(
        builders,
        raw_peaks,
        raw_supports,
        raw_densities,
    ):
        peak_norm = _normalize_unit_interval(
            peak,
            peak_min,
            peak_max,
        )

        support_norm = _normalize_unit_interval(
            support,
            support_min,
            support_max,
        )

        density_norm = _normalize_unit_interval(
            density,
            density_min,
            density_max,
        )

        local_score = (
            config.peak_weight
            * peak_norm
            + config.support_weight
            * support_norm
            + config.density_weight
            * density_norm
        ) / weight_sum

        consensus_weight = (
            config.video_consensus_weight
            if video_scores
            else 0.0
        )

        video_prior = (
            float(video_scores.get(builder.video_id, 0.0))
            if video_scores
            else 0.0
        )

        score = (
            (1.0 - consensus_weight) * local_score
            + consensus_weight * video_prior
        )

        scored.append(
            (
                float(score),
                builder,
            )
        )

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].video_id,
            item[1].start_time,
            item[1].end_time,
        )
    )

    output: list[
        CoarseRegion
    ] = []

    for score, builder in scored:
        hits = list(
            builder.hits
        )

        hits.sort(
            key=lambda hit: (
                -_safe_score(hit),
                _safe_time(hit)
                if _safe_time(hit) is not None
                else float("inf"),
            )
        )

        start_time = float(
            builder.start_time
        )

        end_time = float(
            builder.end_time
        )

        current_duration = end_time - start_time

        if current_duration < config.min_region_duration_seconds:
            center = (start_time + end_time) / 2.0
            half = config.min_region_duration_seconds / 2.0
            start_time = max(0.0, center - half)
            end_time = start_time + config.min_region_duration_seconds

        if config.region_padding_seconds > 0:
            start_time = max(
                0.0,
                start_time
                - config.region_padding_seconds,
            )

            end_time = (
                end_time
                + config.region_padding_seconds
            )

            # Padding không được làm region vượt max duration.
            if (
                end_time
                - start_time
                > config.max_region_duration_seconds
            ):
                center = (
                    builder.start_time
                    + builder.end_time
                ) / 2.0

                half = (
                    config.max_region_duration_seconds
                    / 2.0
                )

                start_time = max(
                    0.0,
                    center - half,
                )

                end_time = (
                    start_time
                    + config.max_region_duration_seconds
                )

        output.append(
            CoarseRegion(
                video_id=builder.video_id,
                start_time=float(
                    start_time
                ),
                end_time=float(
                    end_time
                ),
                score=float(
                    score
                ),
                hits=tuple(hits),
            )
        )

    return output


# ============================================================================
# TEMPORAL GROUPING
# ============================================================================
def _select_diverse_regions(
    regions: list[CoarseRegion],
    limit: int,
    *,
    max_per_video: int = 2,
    priority_videos: Iterable[str] = (),
    priority_video_limit: int = 0,
) -> list[CoarseRegion]:

    if limit <= 0:
        return []

    selected: list[
        CoarseRegion
    ] = []

    counts: dict[
        str,
        int,
    ] = {}

    selected_ids: set[int] = set()

    # ---------------------------------------------------------
    # Rescue quota:
    # giữ best region của các video có consensus toàn chuỗi cao.
    # Không dùng GT; priority được tính từ raw retrieval của mọi event.
    # ---------------------------------------------------------

    if priority_video_limit > 0:
        unique_priority = list(dict.fromkeys(str(v) for v in priority_videos))

        for video_id in unique_priority[:priority_video_limit]:
            region = next(
                (item for item in regions if item.video_id == video_id),
                None,
            )
            if region is None:
                continue

            selected.append(region)
            selected_ids.add(id(region))
            counts[video_id] = 1

            if len(selected) >= limit:
                return selected

    # ---------------------------------------------------------
    # Pass 1:
    # tối đa max_per_video để tránh một video chiếm hết top-N.
    # ---------------------------------------------------------

    for region in regions:

        if id(region) in selected_ids:
            continue

        count = counts.get(
            region.video_id,
            0,
        )

        if count >= max_per_video:
            continue

        selected.append(
            region
        )

        counts[
            region.video_id
        ] = count + 1

        selected_ids.add(id(region))

        if len(selected) >= limit:
            return selected

    # ---------------------------------------------------------
    # Pass 2:
    # nếu chưa đủ thì fill lại theo score.
    # ---------------------------------------------------------

    for region in regions:

        if id(region) in selected_ids:
            continue

        selected.append(
            region
        )

        if len(selected) >= limit:
            break

    return selected

def _gom_vung(
    hits: list[dict[str, Any]],
    config: TRR1Config,
    *,
    video_scores: dict[str, float] | None = None,
) -> list[CoarseRegion]:
    """
    Gộp raw hits thành coarse temporal regions.

    Quy trình:

        raw frame hits
            ↓
        normalize / validate
            ↓
        sort theo video + timestamp
            ↓
        temporal grouping
            ↓
        candidate region filtering
            ↓
        region-level scoring
            ↓
        global score sort
            ↓
        max_regions_per_event

    Khác bản cũ:
        region score KHÔNG còn đơn giản là max(frame score).
    """

    if not hits:
        return []

    normalized: list[
        dict[str, Any]
    ] = []

    for hit in hits:
        if not isinstance(
            hit,
            dict,
        ):
            continue

        video_id = _safe_video_id(hit)
        pts_time = _safe_time(hit)
        score = _safe_score(hit)

        if not video_id:
            continue

        if pts_time is None:
            continue

        if score == float("-inf"):
            continue

        normalized.append(
            {
                **hit,
                "video_id": video_id,
                "pts_time": float(
                    pts_time
                ),
                "score": float(
                    score
                ),
            }
        )

    if not normalized:
        return []

    # Temporal grouping phải dựa trên timestamp,
    # không dựa trên retrieval rank.
    normalized.sort(
        key=lambda hit: (
            str(
                hit["video_id"]
            ),
            float(
                hit["pts_time"]
            ),
            -float(
                hit["score"]
            ),
        )
    )

    builders: list[
        _RegionBuilder
    ] = []

    current: (
        _RegionBuilder | None
    ) = None

    for hit in normalized:
        video_id = str(
            hit["video_id"]
        )

        hit_time = float(
            hit["pts_time"]
        )

        if current is None:
            current = _RegionBuilder(
                video_id=video_id,
                start_time=hit_time,
                end_time=hit_time,
                hits=[hit],
            )
            continue

        if _can_merge(
            current,
            hit,
            config,
        ):
            _append_hit(
                current,
                hit,
            )
            continue

        builders.append(
            current
        )

        current = _RegionBuilder(
            video_id=video_id,
            start_time=hit_time,
            end_time=hit_time,
            hits=[hit],
        )

    if current is not None:
        builders.append(
            current
        )

    regions = _score_regions(
        builders,
        config,
        video_scores=video_scores,
    )

    priority_videos = (
        sorted(
            video_scores,
            key=lambda video_id: (-float(video_scores[video_id]), video_id),
        )
        if video_scores
        else []
    )

    return _select_diverse_regions(
        regions,
        config.max_regions_per_event,
        max_per_video=2,
        priority_videos=priority_videos,
        priority_video_limit=config.consensus_rescue_videos,
    )


# ============================================================================
# RETRIEVER ADAPTER
# ============================================================================


def _default_retriever(
    text: str,
    top_k: int,
    *,
    use_query_expansion: bool = True,
    max_query_variants: int = 2,
) -> list[dict[str, Any]]:
    """
    Production retriever mặc định.

    Dùng:
        Marian
        +
        CLIP-L
        +
        FAISS CLIP-L

    tim_vung_tho() chỉ gọi retriever một lần/event.
    """

    return tim_hit_clip_l(
        text,
        top_k=top_k,
        use_query_expansion=use_query_expansion,
        max_query_variants=max_query_variants,
    )


def _retrieve_hits(
    text: str,
    config: TRR1Config,
    retriever: Callable[
        [str, int],
        list[dict[str, Any]],
    ]
    | None,
) -> list[dict[str, Any]]:
    """Gọi retriever một lần và luôn tôn trọng cấu hình query expansion."""

    if retriever is None:
        hits = _default_retriever(
            text,
            config.top_k,
            use_query_expansion=config.use_query_expansion,
            max_query_variants=config.max_query_variants,
        )
    else:
        hits = retriever(text, config.top_k)

    if hits is None:
        return []

    if isinstance(hits, list):
        return hits

    return list(hits)


# ============================================================================
# PUBLIC API — ONE EVENT
# ============================================================================


def tim_vung_tho(
    event: dict[str, Any],
    *,
    config: TRR1Config | None = None,
    retriever: Callable[
        [str, int],
        list[dict[str, Any]],
    ]
    | None = None,
) -> TRR1Result:
    """
    Tìm coarse regions cho MỘT event.

    Parameters
    ----------
    event:
        Dict bắt buộc có:
            event_id
            text

        relation là optional.

    config:
        TR-R1 config.

    retriever:
        Dependency injection cho unit test.

        Signature:
            retriever(text, top_k) -> list[hit]

    Returns
    -------
    TRR1Result
    """

    if config is None:
        config = TRR1Config()

    event_id, text, relation = _parse_event(
        event
    )

    hits = _retrieve_hits(
        text,
        config,
        retriever,
    )

    regions = _gom_vung(
        hits,
        config=config,
    )

    return TRR1Result(
        event_id=event_id,
        text=text,
        relation=relation,
        regions=tuple(
            regions
        ),
    )


# ============================================================================
# PUBLIC API — MULTIPLE EVENTS
# ============================================================================


def tim_nhieu_su_kien(
    events: Iterable[dict[str, Any]],
    *,
    config: TRR1Config | None = None,
    retriever: Callable[
        [str, int],
        list[dict[str, Any]],
    ]
    | None = None,
) -> tuple[TRR1Result, ...]:
    """
    Tìm coarse regions cho nhiều event.

    Thứ tự output giữ nguyên thứ tự QueryPlan.
    """

    if config is None:
        config = TRR1Config()

    parsed_events = [
        _parse_event(event)
        for event in events
    ]

    hits_by_event = [
        _retrieve_hits(text, config, retriever)
        for _event_id, text, _relation in parsed_events
    ]

    consensus_scores = _video_consensus_scores(
        hits_by_event,
        rrf_k=config.video_rrf_k,
    )
    video_scores = consensus_scores

    if config.video_sequence_weight > 0:
        sequence_scores = _video_sequence_scores(
            hits_by_event,
            rrf_k=config.video_rrf_k,
            span_weight=config.video_sequence_span_weight,
            span_scale_seconds=config.video_sequence_span_scale_seconds,
        )
        video_ids = set(consensus_scores) | set(sequence_scores)
        sequence_weight = config.video_sequence_weight
        video_scores = {
            video_id: float(
                (1.0 - sequence_weight) * consensus_scores.get(video_id, 0.0)
                + sequence_weight * sequence_scores.get(video_id, 0.0)
            )
            for video_id in video_ids
        }

    results: list[TRR1Result] = []

    for (
        event_id,
        text,
        relation,
    ), hits in zip(parsed_events, hits_by_event):
        regions = _gom_vung(
            hits,
            config=config,
            video_scores=video_scores,
        )

        results.append(
            TRR1Result(
                event_id=event_id,
                text=text,
                relation=relation,
                regions=tuple(regions),
            )
        )

    return tuple(
        results
    )


def _trr1_results_from_hits(
    parsed_events: list[tuple[str, str, str]],
    hits_by_event: list[list[dict[str, Any]]],
    *,
    config: TRR1Config,
) -> tuple[TRR1Result, ...]:
    """Group một raw-hit snapshot với một cấu hình TR-R1 xác định."""

    consensus_scores = _video_consensus_scores(
        hits_by_event,
        rrf_k=config.video_rrf_k,
    )
    video_scores = consensus_scores

    if config.video_sequence_weight > 0:
        sequence_scores = _video_sequence_scores(
            hits_by_event,
            rrf_k=config.video_rrf_k,
            span_weight=config.video_sequence_span_weight,
            span_scale_seconds=config.video_sequence_span_scale_seconds,
        )
        video_ids = set(consensus_scores) | set(sequence_scores)
        sequence_weight = config.video_sequence_weight
        video_scores = {
            video_id: float(
                (1.0 - sequence_weight) * consensus_scores.get(video_id, 0.0)
                + sequence_weight * sequence_scores.get(video_id, 0.0)
            )
            for video_id in video_ids
        }

    return tuple(
        TRR1Result(
            event_id=event_id,
            text=text,
            relation=relation,
            regions=tuple(
                _gom_vung(
                    hits,
                    config=config,
                    video_scores=video_scores,
                )
            ),
        )
        for (event_id, text, relation), hits in zip(
            parsed_events,
            hits_by_event,
        )
    )


def fuse_video_profile_lists_rrf(
    source_lists: dict[str, Iterable[str]],
    *,
    source_weights: dict[str, float],
    rrf_k: float = 100.0,
    limit: int | None = None,
) -> list[str]:
    """Weighted-RRF nhiều video list, deduplicate và tie-break ổn định."""

    if rrf_k < 0:
        raise ValueError("rrf_k phải >= 0")
    if limit is not None and limit <= 0:
        raise ValueError("limit phải > 0 hoặc None")
    if set(source_lists) != set(source_weights):
        raise ValueError("source_lists và source_weights phải cùng key")
    if any(float(weight) < 0 for weight in source_weights.values()):
        raise ValueError("source weight phải >= 0")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for source_name, raw_values in source_lists.items():
        seen: set[str] = set()
        rank = 0
        for raw_video_id in raw_values:
            video_id = str(raw_video_id or "").strip()
            if not video_id or video_id in seen:
                continue
            seen.add(video_id)
            rank += 1
            scores[video_id] = scores.get(video_id, 0.0) + (
                float(source_weights[source_name]) / (rrf_k + rank)
            )
            best_rank[video_id] = min(best_rank.get(video_id, rank), rank)

    ranked = sorted(
        scores,
        key=lambda video_id: (
            -scores[video_id],
            best_rank[video_id],
            video_id,
        ),
    )
    return ranked[:limit] if limit is not None else ranked


def tim_nhieu_su_kien_profile_union(
    events: Iterable[dict[str, Any]],
    *,
    config: TRR1Config | None = None,
    retriever: Callable[[str, int], list[dict[str, Any]]] | None = None,
    video_beam_size: int = 12,
) -> dict[str, Any]:
    """Tạo TR-R1 beam union đã chốt từ sequence/visual/consensus profiles.

    Cấu hình không dùng GT và phản ánh đúng ablation thắng:
    sequence weight 1.0, visual sequence weight 0.5, rescue 4; sau đó
    weighted-RRF ba list với weights 1.5/0.5/1.0 và k=100. Mỗi source giữ
    ít nhất top-12; khi beam lớn hơn 12, source cũng phải mở cùng mức để
    candidate ở rank 13+ thực sự có cơ hội đi vào fusion.
    """

    if config is None:
        config = TRR1Config()
    if video_beam_size <= 0:
        raise ValueError("video_beam_size phải > 0")

    parsed_events = [_parse_event(event) for event in events]
    if not parsed_events:
        raise ValueError("events không được rỗng")

    base_hits = [
        _retrieve_hits(text, config, retriever)
        for _event_id, text, _relation in parsed_events
    ]
    visual_hits: list[list[dict[str, Any]]] = []
    visual_variants_by_event: dict[str, list[str]] = {}

    for (event_id, text, _relation), event_hits in zip(parsed_events, base_hits):
        variants = _trr1_visual_action_variants(text)
        visual_variants_by_event[event_id] = variants
        extra_hits: list[list[dict[str, Any]]] = []
        for variant in variants:
            if retriever is None:
                hits = _tim_hit_clip_l_mot_query(variant, config.top_k)
            else:
                raw_hits = retriever(variant, config.top_k)
                hits = list(raw_hits or [])
            extra_hits.append(hits)
        visual_hits.append(
            _merge_query_variant_hits([event_hits, *extra_hits])
            if extra_hits
            else event_hits
        )

    shared = dict(
        max_regions_per_event=10,
        video_consensus_weight=0.45,
        video_rrf_k=60.0,
        video_sequence_span_weight=0.0,
        video_sequence_span_scale_seconds=60.0,
        consensus_rescue_videos=4,
    )
    sequence_config = replace(
        config,
        **shared,
        video_sequence_weight=1.0,
    )
    visual_config = replace(
        config,
        **shared,
        video_sequence_weight=0.5,
    )
    sequence_results = _trr1_results_from_hits(
        parsed_events,
        base_hits,
        config=sequence_config,
    )
    visual_results = _trr1_results_from_hits(
        parsed_events,
        visual_hits,
        config=visual_config,
    )

    from aic2026.trake_r2_windows import rank_video_candidates_rrf

    def event_regions(results: tuple[TRR1Result, ...]) -> dict[str, list[dict[str, Any]]]:
        return {
            result.event_id: [coarse_region_to_dict(region) for region in result.regions]
            for result in results
        }

    source_limit = max(12, int(video_beam_size))
    sequence_beam = rank_video_candidates_rrf(
        event_regions(sequence_results),
        k=60,
        limit=source_limit,
    )
    visual_beam = rank_video_candidates_rrf(
        event_regions(visual_results),
        k=60,
        limit=source_limit,
    )
    visual_consensus = _video_consensus_scores(visual_hits, rrf_k=60.0)
    consensus_beam = sorted(
        visual_consensus,
        key=lambda video_id: (-visual_consensus[video_id], video_id),
    )[:source_limit]
    source_lists = {
        "sequence": sequence_beam,
        "visual": visual_beam,
        "consensus": consensus_beam,
    }
    candidate_video_ids = fuse_video_profile_lists_rrf(
        source_lists,
        source_weights={
            "sequence": 1.5,
            "visual": 0.5,
            "consensus": 1.0,
        },
        rrf_k=100.0,
        limit=video_beam_size,
    )

    return {
        "results": sequence_results,
        "candidate_video_ids": candidate_video_ids,
        "source_video_ids": source_lists,
        "visual_variants_by_event": visual_variants_by_event,
        "config_key": "rrf|sequence=1.5|visual=0.5|consensus=1|k=100",
    }


# ============================================================================
# SERIALIZATION
# ============================================================================


def coarse_region_to_dict(
    region: CoarseRegion,
) -> dict[str, Any]:
    """
    Serialize CoarseRegion.

    Contract CHỈ gồm:

        video_id
        start_time
        end_time
        score
        hits

    Không có frame_idx/best_frame_idx ở top-level.
    """

    return {
        "video_id": region.video_id,
        "start_time": float(
            region.start_time
        ),
        "end_time": float(
            region.end_time
        ),
        "score": float(
            region.score
        ),
        "hits": [
            dict(hit)
            for hit in region.hits
        ],
    }


def trr1_result_to_dict(
    result: TRR1Result,
) -> dict[str, Any]:
    """
    Serialize TRR1Result.
    """

    return {
        "event_id": result.event_id,
        "text": result.text,
        "relation": result.relation,
        "regions": [
            coarse_region_to_dict(
                region
            )
            for region in result.regions
        ],
    }


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "TRR1Config",
    "CoarseRegion",
    "TRR1Result",
    "_build_test_hit",
    "_gom_vung",
    "_trr1_visual_action_variants",
    "_video_consensus_scores",
    "_video_sequence_scores",
    "coarse_region_to_dict",
    "fuse_video_profile_lists_rrf",
    "tim_vung_tho",
    "tim_nhieu_su_kien",
    "tim_nhieu_su_kien_profile_union",
    "trr1_result_to_dict",
]
